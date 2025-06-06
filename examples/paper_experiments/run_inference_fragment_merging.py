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

# loading model
chkpt = 'shepherd_chkpts/x1x3x4_diffusion_mosesaq_20240824_submission.ckpt'
model_pl = LightningModule.load_from_checkpoint(chkpt)
params = model_pl.params
model_pl.to(device)
model_pl.model.device = device


# getting interaction profiles of target
with open('conformers/fragment_merging/fragment_merge_condition.pickle', 'rb') as f:
    fragment_merge_features = pickle.load(f)
COM = fragment_merge_features['x3']['positions'].mean(0)
fragment_merge_features['x2']['positions'] = fragment_merge_features['x2']['positions'] - COM
fragment_merge_features['x3']['positions'] = fragment_merge_features['x3']['positions'] - COM
fragment_merge_features['x4']['positions'] = fragment_merge_features['x4']['positions'] - COM

fragment_mols = []
for i in range(0,13):
    fmol = f'conformers/fragment_merging/fragments/mol_{i}.mol'
    fmol = rdkit.Chem.MolFromMolFile(fmol, removeHs = False)
    coordinates = fmol.GetConformer().GetPositions()
    coordinates = coordinates - COM
    fmol = update_mol_coordinates(fmol, coordinates)
    fragment_mols.append(rdkit.Chem.MolToMolBlock(fmol))
    
surface_fragments = deepcopy(fragment_merge_features['x3']['positions'])
electrostatics_fragments = deepcopy(fragment_merge_features['x3']['charges'])
pharm_types_fragments = deepcopy(fragment_merge_features['x4']['types'])
pharm_pos_fragments = deepcopy(fragment_merge_features['x4']['positions'])
pharm_direction_fragments = deepcopy(fragment_merge_features['x4']['directions'])


# Running Inference
# -> sampling from conditional distribution P(x1|x3,x4) via inpainting P(x1,x3,x4)

batch_size = 5

n_atoms_list = list(range(50,90)) # 40 total
n_atoms_list = n_atoms_list * 5 # 5 batches per choice of n_atoms, 200 batches total, 1000 samples total with batch size of 5

num_batches = len(n_atoms_list)

generated_samples = []
for n in range(num_batches):
    print(n)
    
    n_atoms = int(n_atoms_list[n])
    
    surface_fragments = deepcopy(fragment_merge_features['x3']['positions'])
    electrostatics_fragments = deepcopy(fragment_merge_features['x3']['charges'])
    pharm_types_fragments = deepcopy(fragment_merge_features['x4']['types'])
    pharm_pos_fragments = deepcopy(fragment_merge_features['x4']['positions'])
    pharm_direction_fragments = deepcopy(fragment_merge_features['x4']['directions'])

    generated_samples_batch = inference_sample(
        model_pl,
        batch_size = batch_size,
        
        N_x1 = n_atoms,
        N_x4 = len(pharm_types_fragments),
        
        unconditional = False,
        
        prior_noise_scale = 1.0,
        denoising_noise_scale = 1.0,
        
        inject_noise_at_ts = [],
        inject_noise_scales = [],
        
        harmonize = False,
        harmonize_ts = [],
        harmonize_jumps = [],
        
        # all the below options are only relevant if unconditional is False
        
        inpaint_x2_pos = False,
        
        inpaint_x3_pos = True,
        inpaint_x3_x = True,
        
        inpaint_x4_pos = True,
        inpaint_x4_direction = True,
        inpaint_x4_type = True,
        
        stop_inpainting_at_time_x2 = 0.0,
        add_noise_to_inpainted_x2_pos = 0.0,
        
        stop_inpainting_at_time_x3 = 0.0,
        add_noise_to_inpainted_x3_pos = 0.0,
        add_noise_to_inpainted_x3_x = 0.0,
        
        stop_inpainting_at_time_x4 = 0.0,
        add_noise_to_inpainted_x4_pos = 0.0,
        add_noise_to_inpainted_x4_direction = 0.0,
        add_noise_to_inpainted_x4_type = 0.0,
        
        # these are the inpainting targets
        center_of_mass = np.zeros(3), # center of mass of x1; already centered to zero above
        surface = surface_fragments,
        electrostatics = electrostatics_fragments,
        pharm_types = pharm_types_fragments,
        pharm_pos = pharm_pos_fragments,
        pharm_direction = pharm_direction_fragments,
        
    )
    
    generated_samples += generated_samples_batch


result = (
    fragment_mols,
    surface_fragments,
    electrostatics_fragments,
    pharm_types_fragments,
    pharm_pos_fragments,
    pharm_direction_fragments,
    generated_samples, # samples from shepherd
)

with open(f"samples/fragment_merging_samples/samples.pickle", "wb") as f:
    pickle.dump(result, f)
