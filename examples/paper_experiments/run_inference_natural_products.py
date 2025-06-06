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


with open('conformers/np/molblock_charges_NPs.pkl', 'rb') as f:
    molblocks_and_charges = pickle.load(f)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("NP_index", type=int) # 0, 1, 2
args = parser.parse_args()
NP_index = args.NP_index

results = {}
#for k in range(len(molblocks_and_charges)):
for k in [NP_index]:
    
    ##### Extracting target interaction profiles from reference molecule #####
    
    mol = rdkit.Chem.MolFromMolBlock(molblocks_and_charges[k][0], removeHs = False)
    charges = np.array(molblocks_and_charges[k][1])
    
    # centering molecule coordinates
    mol_coordinates = np.array(mol.GetConformer().GetPositions())
    mol_coordinates = mol_coordinates - np.mean(mol_coordinates, axis = 0)
    mol = update_mol_coordinates(mol, mol_coordinates)
    
    centers = mol.GetConformer().GetPositions()
    radii = get_atomic_vdw_radii(mol)
    surface = get_molecular_surface(
        centers, 
        radii, 
        params['dataset']['x3']['num_points'], 
        probe_radius = params['dataset']['probe_radius'],
        num_samples_per_atom = 20,
    )
    
    pharm_types, pharm_pos, pharm_direction = get_pharmacophores(
        mol,
        multi_vector = params['dataset']['x4']['multivectors'],
        check_access = params['dataset']['x4']['check_accessibility'],
    )

    electrostatics = get_electrostatics_given_point_charges(
        charges, centers, surface,
    )
    
    ########################################################################
    
    
    # Running Inference
    # -> sampling from conditional distribution P(x1|x3,x4) via inpainting P(x1,x3,x4)
    
    batch_size = 10
    
    n_atoms_list = [36,38,40,42,44,46,48,50,51,52,53,54,55,56,57,58,59,60,62,64,66,68,70,75,80] # len(n_atoms_list) == 25
    n_atoms_list = n_atoms_list * 10 # 10 batches per choice of n_atoms, 2500 samples total
    
    num_batches = len(n_atoms_list)
    
    generated_samples = []
    for n in range(num_batches):
        
        n_atoms = n_atoms_list[n]
        
        generated_samples_batch = inference_sample(
            model_pl,
            batch_size = batch_size,
            
            N_x1 = n_atoms,
            N_x4 = len(pharm_types), # must equal pharm_pos.shape[0] if inpainting
            
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
            surface = surface,
            electrostatics = electrostatics,
            pharm_types = pharm_types,
            pharm_pos = pharm_pos,
            pharm_direction = pharm_direction,
        
        )
        
        generated_samples += generated_samples_batch
    
    result = (
        rdkit.Chem.MolToMolBlock(mol), 
        charges, 
        surface, 
        electrostatics, 
        pharm_types, 
        pharm_pos, 
        pharm_direction, 
        generated_samples, # samples from shepherd
    )
    
    results[k] = result

    with open(f"samples/NP_analogues/shepherd_samples_NP_{k}.pickle", "wb") as f:
        pickle.dump(result, f)
