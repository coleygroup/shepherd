import open3d 
from shepherd_score_utils.generate_point_cloud import (
    get_atom_coords, 
    get_atomic_vdw_radii, 
    get_molecular_surface,
    get_electrostatics,
    get_electrostatics_given_point_charges,
)
from shepherd_score_utils.pharm_utils.pharmacophore import get_pharmacophores
from shepherd_score_utils.conformer_generation import update_mol_coordinates

print('importing rdkit')
import rdkit
from rdkit.Chem import rdDetermineBonds

import numpy as np
import matplotlib.pyplot as plt

print('importing torch')
import torch
import torch_geometric
from torch_geometric.nn import radius_graph
import torch_scatter

import pickle
from copy import deepcopy
import os
import multiprocessing
from tqdm import tqdm

import sys
sys.path.insert(-1, "model/")
sys.path.insert(-1, "model/equiformer_v2")

print('importing lightning')
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from lightning_module import LightningModule
from datasets import HeteroDataset

import importlib

from inference import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

chkpt = 'shepherd_chkpts/x1x3x4_diffusion_mosesaq_20240824_submission.ckpt'

model_pl = LightningModule.load_from_checkpoint(chkpt)
params = model_pl.params
model_pl.to(device)
model_pl.model.device = device

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("batch_size", type=int) # should be set to 1 if using cpus
parser.add_argument("num_batches", type=int)
parser.add_argument("n_atoms", type=int)
parser.add_argument("N_x4", type=int)
parser.add_argument("file_index", type=int)
args = parser.parse_args()

batch_size = args.batch_size
n_atoms = args.n_atoms
num_batches = args.num_batches
file_index = args.file_index
#N_x4 = args.N_x4 # we sample this from the empirical distribution P(n_pharmacophores | n_atoms)

distributions = np.load('conformers/distributions/atom_pharm_count.npz')

generated_samples = []
for n in range(num_batches):
    
    prob = distributions['moses_aq'][n_atoms,:]
    if prob.sum() > 0.0:
        prob = prob / prob.sum()
        N_x4 = int(np.random.multinomial(1, pvals=prob).argmax())
    else:
        prob = distributions['moses_aq'].sum(axis = 0)
        prob = prob / prob.sum()
        N_x4 = int(np.random.multinomial(1, pvals=prob).argmax())
        
    
    # only use to break symmetry during unconditional generation
    T = params['noise_schedules']['x1']['T']
    
    inject_noise_at_ts = list(np.arange(130, 80, -1)) # [150]
    inject_noise_scales = [1.0] * len(inject_noise_at_ts)
    harmonize = True
    harmonize_ts = [80]
    harmonize_jumps = [20]
    
    generated_samples_batch = inference_sample(
        model_pl,
        batch_size = batch_size,
        
        N_x1 = n_atoms,
        N_x4 = N_x4, # must equal len(pharm_types) if inpainting
        
        unconditional = True,
        
        prior_noise_scale = 1.0,
        denoising_noise_scale = 1.0,
        
        # only use to break symmetry during unconditional generation
        inject_noise_at_ts = inject_noise_at_ts, #[],
        inject_noise_scales = inject_noise_scales, #[],    
        harmonize = harmonize, # False
        harmonize_ts = harmonize_ts, #[],
        harmonize_jumps = harmonize_jumps, #[],
        
        
        # all the below options are only relevant if unconditional is False
        
        inpaint_x2_pos = False,
        
        inpaint_x3_pos = False,
        inpaint_x3_x = False,
        
        inpaint_x4_pos = False,
        inpaint_x4_direction = False,
        inpaint_x4_type = False,
        
        stop_inpainting_at_time_x2 = 0.0, # range from 0.0 to 1.0 (fraction of T)
        add_noise_to_inpainted_x2_pos = 0.0,
        
        stop_inpainting_at_time_x3 = 0.0, # range from 0.0 to 1.0 (fraction of T)
        add_noise_to_inpainted_x3_pos = 0.0,
        add_noise_to_inpainted_x3_x = 0.0,
        
        stop_inpainting_at_time_x4 = 0.0, # range from 0.0 to 1.0 (fraction of T)
        add_noise_to_inpainted_x4_pos = 0.0,
        add_noise_to_inpainted_x4_direction = 0.0,
        add_noise_to_inpainted_x4_type = 0.0,
        
        # these are the inpainting targets
        center_of_mass = np.zeros(3), # center of mass of x1; already centered to zero above
        surface = np.zeros((75,3)),
        electrostatics = np.zeros(75),
        pharm_types = np.zeros(5, dtype = int),
        pharm_pos = np.zeros((5, 3)),
        pharm_direction = np.zeros((5, 3)),
       
    )
    generated_samples = generated_samples + generated_samples_batch

with open(f"samples/mosesaq_unconditional/x1x3x4/samples_{file_index}.pickle", "wb") as f:
    pickle.dump(generated_samples, f)
