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

import rdkit
from rdkit.Chem import rdDetermineBonds

import numpy as np
import matplotlib.pyplot as plt

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

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from lightning_module import LightningModule
from datasets import HeteroDataset

import importlib

# harmonization functions
def forward_jump_parameters(t_start_idx, jump, sigma_ts):
    t_end_idx = t_start_idx + jump
    sigma_ts_ = sigma_ts[t_start_idx : t_end_idx] # std deviation schedule
    alpha_ts_ = (1. - sigma_ts_** 2.0)**0.5 # how much (mean) signal is preserved at each noising step
    alpha_dash_ts_ = np.cumprod(alpha_ts_)
    var_dash_ts_ = 1. - alpha_dash_ts_**2.0
    sigma_dash_ts_ = var_dash_ts_**0.5
    return alpha_dash_ts_[-1], sigma_dash_ts_[-1], t_end_idx

def forward_jump(x, t_start, jump, sigma_ts, remove_COM_from_noise = False, batch = None, mask = None):
    assert t_start + jump <= sigma_ts.shape[0] + 1 # can't jump past t=T
    
    if mask is None:
        mask = torch.zeros(x.shape, dtype = torch.long) == 0
    
    t_start_idx = t_start - 1
    alpha_jump, sigma_jump, t_end_idx = forward_jump_parameters(t_start_idx, jump, sigma_ts)
    t_end = t_end_idx + 1
    
    noise = torch.randn(x.shape)
    if remove_COM_from_noise:
        assert batch is not None
        assert len(x.shape) == 2
        # noise must have shape (N, 3)
        assert noise.shape[1] == 3
        noise = noise - torch_scatter.scatter_mean(noise[mask], batch[mask], dim = 0)[batch]
    noise[~mask, ...] = 0.0
    
    x_jump = alpha_jump * x + sigma_jump * noise
    x_jump[~mask] = x[~mask]
    
    return x_jump, t_end

# simulate forward noising trajectory
def forward_trajectory(x, ts, alpha_ts, sigma_ts, remove_COM_from_noise = False, mask = None, deterministic = False):
    if mask is None:
        mask = torch.ones(x.shape[0]) == 1.0
    
    if remove_COM_from_noise:
        assert x.shape[1] == 3
    
    trajectory = {0:x}
    
    x_t = x
    for t_idx, t in enumerate(ts):
        
        alpha_t = alpha_ts[t_idx]
        sigma_t = sigma_ts[t_idx]
        
        noise = torch.randn(x.shape)
        if remove_COM_from_noise:
            noise = noise - torch.mean(noise[mask], dim = 0)
        noise[~mask, ...] = 0.0
        
        if deterministic:
            noise = 0.0
        
        x_t_plus_1 = alpha_t * x_t + sigma_t * noise
        x_t_plus_1[~mask, ...] = x_t[~mask, ...]
        
        trajectory[t] = x_t_plus_1
        
        x_t = x_t_plus_1
    
    return trajectory


def inference_sample(
    model_pl,
    batch_size,
    
    N_x1,
    N_x4,
    
    unconditional,
    
    prior_noise_scale = 1.0,
    denoising_noise_scale = 1.0,
    
    inject_noise_at_ts = [],
    inject_noise_scales = [],    
    
    harmonize = False,
    harmonize_ts = [],
    harmonize_jumps = [],
    
    
    # all the below options are only relevant if unconditional is False
    
    inpaint_x2_pos = False,
    inpaint_x3_pos = False,
    inpaint_x3_x = False,
    inpaint_x4_pos = False,
    inpaint_x4_direction = False,
    inpaint_x4_type = False,
    
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
    center_of_mass = np.zeros(3),
    surface = np.zeros((75,3)),
    electrostatics = np.zeros(75),
    pharm_types = np.zeros(5, dtype = int),
    pharm_pos = np.zeros((5,3)),
    pharm_direction = np.zeros((5,3)),
    
):
    
    params = model_pl.params
    
    T = params['noise_schedules']['x1']['ts'].max()

    N_x2 = params['dataset']['x2']['num_points']
    N_x3 = params['dataset']['x3']['num_points']
    
    
    ####### Defining inpainting targets ########
    
    # centering about provided center of mass (of x1)
    surface = surface - center_of_mass
    pharm_pos = pharm_pos - center_of_mass
    
    # adding small noise to pharm_pos to avoid overlapping points (causes error when encoding clean structure)
    pharm_pos = pharm_pos + np.random.randn(*pharm_pos.shape) * 0.01
    
    # accounting for virtual nodes
    surface = np.concatenate([np.array([[0.0, 0.0, 0.0]]), surface], axis = 0) # virtual node
    electrostatics = np.concatenate([np.array([0.0]), electrostatics], axis = 0) # virtual node
    pharm_types = pharm_types + 1 # accounting for virtual node as the zeroeth type
    pharm_types = np.concatenate([np.array([0]), pharm_types], axis = 0) # virtual node
    pharm_pos = np.concatenate([np.array([[0.0, 0.0, 0.0]]), pharm_pos], axis = 0) # virtual node
    pharm_direction = np.concatenate([np.array([[0.0, 0.0, 0.0]]), pharm_direction], axis = 0) # virtual node
    
    # one-hot-encodings
    pharm_types_one_hot = np.zeros((pharm_types.size, params['dataset']['x4']['max_node_types']))
    pharm_types_one_hot[np.arange(pharm_types.size), pharm_types] = 1
    pharm_types = pharm_types_one_hot
    
    # scaling features
    electrostatics = electrostatics * params['dataset']['x3']['scale_node_features']
    pharm_types = pharm_types * params['dataset']['x4']['scale_node_features']
    pharm_direction = pharm_direction * params['dataset']['x4']['scale_vector_features']
    
    # defining inpainting targets
    target_inpaint_x2_pos = torch.as_tensor(surface, dtype = torch.float)
    target_inpaint_x2_mask = torch.zeros(surface.shape[0], dtype = torch.long)
    target_inpaint_x2_mask[0] = 1
    target_inpaint_x2_mask = target_inpaint_x2_mask == 0

    target_inpaint_x3_x = torch.as_tensor(electrostatics, dtype = torch.float)
    target_inpaint_x3_pos = torch.as_tensor(surface, dtype = torch.float)
    target_inpaint_x3_mask = torch.zeros(electrostatics.shape[0], dtype = torch.long)
    target_inpaint_x3_mask[0] = 1
    target_inpaint_x3_mask = target_inpaint_x3_mask == 0

    target_inpaint_x4_x = torch.as_tensor(pharm_types, dtype = torch.float)
    target_inpaint_x4_pos = torch.as_tensor(pharm_pos, dtype = torch.float)
    target_inpaint_x4_direction = torch.as_tensor(pharm_direction, dtype = torch.float)
    target_inpaint_x4_mask = torch.zeros(pharm_types.shape[0], dtype = torch.long)
    target_inpaint_x4_mask[0] = 1
    target_inpaint_x4_mask = target_inpaint_x4_mask == 0
    
    deterministic_inpainting_x1 = False
    deterministic_inpainting_x2 = False
    deterministic_inpainting_x3 = False
    deterministic_inpainting_x4 = False
    
    x2_pos_inpainting_trajectory = forward_trajectory(
        x = target_inpaint_x2_pos,
        
        ts = params['noise_schedules']['x2']['ts'],
        alpha_ts = params['noise_schedules']['x2']['alpha_ts'],
        sigma_ts = params['noise_schedules']['x2']['sigma_ts'],
        remove_COM_from_noise = False,
        mask = target_inpaint_x2_mask,
        deterministic = deterministic_inpainting_x2,
    )
    
    x3_pos_inpainting_trajectory = forward_trajectory(
        x = target_inpaint_x3_pos,
        
        ts = params['noise_schedules']['x3']['ts'],
        alpha_ts = params['noise_schedules']['x3']['alpha_ts'],
        sigma_ts = params['noise_schedules']['x3']['sigma_ts'],
        remove_COM_from_noise = False,
        mask = target_inpaint_x3_mask,
        deterministic = deterministic_inpainting_x3,
    )
    x3_x_inpainting_trajectory = forward_trajectory(
        x = target_inpaint_x3_x,
        
        ts = params['noise_schedules']['x3']['ts'],
        alpha_ts = params['noise_schedules']['x3']['alpha_ts'],
        sigma_ts = params['noise_schedules']['x3']['sigma_ts'],
        remove_COM_from_noise = False,
        mask = target_inpaint_x3_mask,
        deterministic = deterministic_inpainting_x3,
    )
    
    x4_x_inpainting_trajectory = forward_trajectory(
        x = target_inpaint_x4_x,
        
        ts = params['noise_schedules']['x4']['ts'],
        alpha_ts = params['noise_schedules']['x4']['alpha_ts'],
        sigma_ts = params['noise_schedules']['x4']['sigma_ts'],
        remove_COM_from_noise = False,
        mask = target_inpaint_x4_mask,
        deterministic = deterministic_inpainting_x4,
    )
    x4_pos_inpainting_trajectory = forward_trajectory(
        x = target_inpaint_x4_pos,
        
        ts = params['noise_schedules']['x4']['ts'],
        alpha_ts = params['noise_schedules']['x4']['alpha_ts'],
        sigma_ts = params['noise_schedules']['x4']['sigma_ts'],
        remove_COM_from_noise = False,
        mask = target_inpaint_x4_mask,
        deterministic = deterministic_inpainting_x4,
    )
    x4_direction_inpainting_trajectory = forward_trajectory(
        x = target_inpaint_x4_direction,
        
        ts = params['noise_schedules']['x4']['ts'],
        alpha_ts = params['noise_schedules']['x4']['alpha_ts'],
        sigma_ts = params['noise_schedules']['x4']['sigma_ts'],
        remove_COM_from_noise = False,
        mask = target_inpaint_x4_mask,
        deterministic = deterministic_inpainting_x4,
    )

    ####################################

    
    # override conditioning options
    if unconditional:
        inpaint_x2_pos = False
        inpaint_x3_pos = False
        inpaint_x3_x = False
        inpaint_x4_pos = False
        inpaint_x4_direction = False
        inpaint_x4_type = False
        
    stop_inpainting_at_time_x2 = int(T*stop_inpainting_at_time_x2)
    stop_inpainting_at_time_x3 = int(T*stop_inpainting_at_time_x3)
    stop_inpainting_at_time_x4 = int(T*stop_inpainting_at_time_x4)
    
    
    ###########  Initializing states at t=T   ##############
    
    include_virtual_node = True

    num_atom_types = len(params['dataset']['x1']['atom_types']) + len(params['dataset']['x1']['charge_types'])
    num_pharm_types = params['dataset']['x4']['max_node_types']
    
    bond_adj = np.triu(1-np.diag(np.ones(N_x1, dtype = int))) # directed graph, to only include 1 edge per bond
    bond_edge_index = np.stack(bond_adj.nonzero(), axis = 0) # this doesn't include any edges to the virtual node
    bond_edge_index = bond_edge_index + int(include_virtual_node)
    bond_edge_index = torch.as_tensor(bond_edge_index, dtype = torch.long)
    
    x1_batch = torch.cat([
        torch.ones(N_x1 + int(include_virtual_node), dtype = torch.long) * i for i in range(batch_size)
    ])
    virtual_node_pos_x1 = torch.tensor([[0.,0.,0.]], dtype = torch.float)
    virtual_node_x_x1 = torch.zeros(num_atom_types, dtype = torch.float)
    virtual_node_x_x1[0] = 1. * params['dataset']['x1']['scale_atom_features'] # one-hot encoding, that remains unnoised
    virtual_node_x_x1 = virtual_node_x_x1[None, ...]
    
    
    x2_batch = torch.cat([
        torch.ones(N_x2 + int(include_virtual_node), dtype = torch.long) * i for i in range(batch_size)
    ])
    virtual_node_pos_x2 = torch.tensor([[0.,0.,0.]], dtype = torch.float) # same as virtual node for x1
    
    
    x3_batch = torch.cat([
        torch.ones(N_x3 + int(include_virtual_node), dtype = torch.long) * i for i in range(batch_size)
    ])
    virtual_node_pos_x3 = torch.tensor([[0.,0.,0.]], dtype = torch.float) # same as virtual node for x1
    virtual_node_x_x3 = torch.tensor([0.0], dtype = torch.float)
    
    
    x4_batch = torch.cat([
        torch.ones(N_x4 + int(include_virtual_node), dtype = torch.long) * i for i in range(batch_size)
    ])
    virtual_node_direction_x4 = torch.tensor([[0.,0.,0.]], dtype = torch.float) 
    virtual_node_pos_x4 = torch.tensor([[0.,0.,0.]], dtype = torch.float) # same as virtual node for x1
    virtual_node_x_x4 =  torch.zeros(num_pharm_types, dtype = torch.float)
    virtual_node_x_x4[0] = 1. * params['dataset']['x4']['scale_node_features'] # one-hot encoding, that remains unnoised
    virtual_node_x_x4 = virtual_node_x_x4[None, ...]
    
    
    # initial state
    virtual_node_mask_x1 = []
    pos_forward_noised_x1 = []
    x_forward_noised_x1 = []
    bond_edge_x_forward_noised_x1 = [] 
    bond_edge_index_x1 = []
    num_nodes_counter = 0
    for b in range(batch_size):
        # continuous gaussian noise for coordinates
        x1_pos_T = torch.randn(N_x1, 3) * prior_noise_scale # t = T
        x1_pos_T = x1_pos_T - torch.mean(x1_pos_T, dim = 0) # removing COM from starting structure
        
        # continuous gaussian noise for atom features
        x1_x_T = torch.randn(N_x1, num_atom_types)
        
        # continuous gaussian noise for bond features
        x1_bond_edge_x_T = torch.randn(bond_edge_index.shape[1], len(params['dataset']['x1']['bond_types']))
        
        x1_virtual_node_mask_ = torch.zeros(N_x1 + int(include_virtual_node), dtype = torch.long)
        if include_virtual_node:
            x1_virtual_node_mask_[0] = 1
            x1_virtual_node_mask_ = x1_virtual_node_mask_ == 1
    
            x1_pos_T = torch.cat([virtual_node_pos_x1, x1_pos_T], dim = 0)
            x1_x_T = torch.cat([virtual_node_x_x1, x1_x_T], dim = 0)
        
        pos_forward_noised_x1.append(x1_pos_T)
        x_forward_noised_x1.append(x1_x_T)
        virtual_node_mask_x1.append(x1_virtual_node_mask_)
        
        bond_edge_x_forward_noised_x1.append(x1_bond_edge_x_T)
        bond_edge_index_x1.append(bond_edge_index + num_nodes_counter)
        num_nodes_counter += N_x1 + int(include_virtual_node)
        
    pos_forward_noised_x1 = torch.cat(pos_forward_noised_x1, dim = 0)
    x_forward_noised_x1 = torch.cat(x_forward_noised_x1, dim = 0)
    virtual_node_mask_x1 = torch.cat(virtual_node_mask_x1, dim =0)
    bond_edge_x_forward_noised_x1 = torch.cat(bond_edge_x_forward_noised_x1, dim =0)
    bond_edge_index_x1 = torch.cat(bond_edge_index_x1, dim =1)
    
    
    virtual_node_mask_x2 = []
    pos_forward_noised_x2 = []
    x_forward_noised_x2 = [] # this is an unnoised one-hot embedding of real/virtual node
    for b in range(batch_size):
        
        # this remains unnoised
        x2_x_T = torch.zeros((N_x2 + int(include_virtual_node), 2))
        x2_x_T[:,0] = 1
        if include_virtual_node:
            x2_x_T[0,0] = 0
            x2_x_T[0,1] = 1
        
        
        x2_pos_T = torch.randn(N_x2, 3) * prior_noise_scale # t = T 
        # NOT removing COM
    
        x2_virtual_node_mask_ = torch.zeros(N_x2 + int(include_virtual_node), dtype = torch.long)
        if include_virtual_node:
            x2_virtual_node_mask_[0] = 1
            x2_virtual_node_mask_ = x2_virtual_node_mask_ == 1
    
            x2_pos_T = torch.cat([virtual_node_pos_x2, x2_pos_T], dim = 0)
        
        pos_forward_noised_x2.append(x2_pos_T)
        x_forward_noised_x2.append(x2_x_T)
        virtual_node_mask_x2.append(x2_virtual_node_mask_)
    pos_forward_noised_x2 = torch.cat(pos_forward_noised_x2, dim = 0)
    x_forward_noised_x2 = torch.cat(x_forward_noised_x2, dim = 0)
    virtual_node_mask_x2 = torch.cat(virtual_node_mask_x2, dim =0)
    
    
    virtual_node_mask_x3 = []
    pos_forward_noised_x3 = []
    x_forward_noised_x3 = []
    for b in range(batch_size):
        
        # continuous gaussian noise for electrostatic potential
        x3_x_T = torch.randn(N_x3, dtype = torch.float) * prior_noise_scale
        
        x3_pos_T = torch.randn(N_x3, 3) * prior_noise_scale # t = T 
        # NOT removing COM from x2/x3 starting structure
    
        x3_virtual_node_mask_ = torch.zeros(N_x3 + int(include_virtual_node), dtype = torch.long)
        if include_virtual_node:
            x3_virtual_node_mask_[0] = 1
            x3_virtual_node_mask_ = x3_virtual_node_mask_ == 1
    
            x3_pos_T = torch.cat([virtual_node_pos_x3, x3_pos_T], dim = 0)
            x3_x_T = torch.cat([virtual_node_x_x3, x3_x_T], dim = 0)
        
        pos_forward_noised_x3.append(x3_pos_T)
        x_forward_noised_x3.append(x3_x_T)
        virtual_node_mask_x3.append(x3_virtual_node_mask_)
    pos_forward_noised_x3 = torch.cat(pos_forward_noised_x3, dim = 0)
    x_forward_noised_x3 = torch.cat(x_forward_noised_x3, dim = 0)
    virtual_node_mask_x3 = torch.cat(virtual_node_mask_x3, dim =0)
    
    
    virtual_node_mask_x4 = []
    pos_forward_noised_x4 = []
    direction_forward_noised_x4 = []
    x_forward_noised_x4 = []
    for b in range(batch_size):
        
        # continuous gaussian noise for coordinates
        x4_pos_T = torch.randn(N_x4, 3) * prior_noise_scale # t = T
        # NOT removing COM from x4
        
        # continuous gaussian noise for directions
        x4_direction_T = torch.randn(N_x4, 3) * prior_noise_scale # t = T
        
        # continuous gaussian noise for atom features
        x4_x_T = torch.randn(N_x4, num_pharm_types)
        
        x4_virtual_node_mask_ = torch.zeros(N_x4 + int(include_virtual_node), dtype = torch.long)
        if include_virtual_node:
            x4_virtual_node_mask_[0] = 1
            x4_virtual_node_mask_ = x4_virtual_node_mask_ == 1
    
            x4_pos_T = torch.cat([virtual_node_pos_x4, x4_pos_T], dim = 0)
            x4_direction_T = torch.cat([virtual_node_direction_x4, x4_direction_T], dim = 0)
            x4_x_T = torch.cat([virtual_node_x_x4, x4_x_T], dim = 0)
        
        pos_forward_noised_x4.append(x4_pos_T)
        direction_forward_noised_x4.append(x4_direction_T)
        x_forward_noised_x4.append(x4_x_T)
        virtual_node_mask_x4.append(x4_virtual_node_mask_)
        
    pos_forward_noised_x4 = torch.cat(pos_forward_noised_x4, dim = 0)
    direction_forward_noised_x4 = torch.cat(direction_forward_noised_x4, dim = 0)
    x_forward_noised_x4 = torch.cat(x_forward_noised_x4, dim = 0)
    virtual_node_mask_x4 = torch.cat(virtual_node_mask_x4, dim =0)
    
    
    # renaming variables for consistency
    x1_pos_t = pos_forward_noised_x1
    x1_x_t = x_forward_noised_x1
    x1_bond_edge_x_t = bond_edge_x_forward_noised_x1
    
    x2_pos_t = pos_forward_noised_x2
    x2_x_t = x_forward_noised_x2
    
    x3_pos_t = pos_forward_noised_x3
    x3_x_t = x_forward_noised_x3
    
    x4_pos_t = pos_forward_noised_x4
    x4_direction_t = direction_forward_noised_x4
    x4_x_t = x_forward_noised_x4
    
    
    x1_batch_size_nodes = x1_pos_t.shape[0]
    x2_batch_size_nodes = x2_pos_t.shape[0]
    x3_batch_size_nodes = x3_pos_t.shape[0]
    x4_batch_size_nodes = x4_pos_t.shape[0]
    
    x1_t = params['noise_schedules']['x1']['ts'][::-1][0]
    x2_t = params['noise_schedules']['x2']['ts'][::-1][0]
    x3_t = params['noise_schedules']['x3']['ts'][::-1][0]
    x4_t = params['noise_schedules']['x4']['ts'][::-1][0]
    
    t = x1_t
    assert x1_t == x2_t
    assert x1_t == x3_t
    assert x1_t == x4_t
    
    if (x2_t > stop_inpainting_at_time_x2):
        if inpaint_x2_pos:
            x2_pos_t = torch.cat([x2_pos_inpainting_trajectory[x2_t] for _ in range(batch_size)], dim = 0)
    
    if (x3_t > stop_inpainting_at_time_x3):
        if inpaint_x3_pos:
            x3_pos_t = torch.cat([x3_pos_inpainting_trajectory[x3_t] for _ in range(batch_size)], dim = 0)
        if inpaint_x3_x:
            x3_x_t = torch.cat([x3_x_inpainting_trajectory[x3_t] for _ in range(batch_size)], dim = 0)
    
    if (x4_t > stop_inpainting_at_time_x4):
        if inpaint_x4_pos:
            x4_pos_t = torch.cat([x4_pos_inpainting_trajectory[x4_t] for _ in range(batch_size)], dim = 0)
        if inpaint_x4_direction:
            x4_direction_t = torch.cat([x4_direction_inpainting_trajectory[x4_t] for _ in range(batch_size)], dim = 0)
        if inpaint_x4_type:
            x4_x_t = torch.cat([x4_x_inpainting_trajectory[x4_t] for _ in range(batch_size)], dim = 0)
        
    
    
    ######## Main Denoising Loop #########
    
    pbar = tqdm(total= T + sum(harmonize_jumps) * int(harmonize))
    
    x1_t_x_list = []
    x1_t_bond_edge_x_list = []
    x1_t_pos_list = []
    
    x2_t_pos_list = []
    
    x3_t_x_list = []
    x3_t_pos_list = []
    
    x4_t_x_list = []
    x4_t_pos_list = []
    x4_t_direction_list = []
    
    while t > 0:
        
        # inputs
        x1_t = t
        x2_t = t
        x3_t = t
        x4_t = t
        
        # harmonize
        if (harmonize) and (t == harmonize_ts[0]):
            #print(f'harmonizing... at time {t}')
            harmonize_ts.pop(0)
            if len(harmonize_ts) == 0:
                harmonize = False
            harmonize_jump = harmonize_jumps.pop(0)
            
            x1_sigma_ts = params['noise_schedules']['x1']['sigma_ts']
            x2_sigma_ts = params['noise_schedules']['x2']['sigma_ts']
            x3_sigma_ts = params['noise_schedules']['x3']['sigma_ts']
            x4_sigma_ts = params['noise_schedules']['x4']['sigma_ts']
            
            x1_pos_t, x1_t_jump = forward_jump(x1_pos_t, x1_t, harmonize_jump, x1_sigma_ts, remove_COM_from_noise = True, batch = x1_batch, mask = ~virtual_node_mask_x1)
            x1_x_t, x1_t_jump = forward_jump(x1_x_t, x1_t, harmonize_jump, x1_sigma_ts, remove_COM_from_noise = False, batch = x1_batch, mask = ~virtual_node_mask_x1)
            x1_bond_edge_x_t, x1_t_jump = forward_jump(x1_bond_edge_x_t, x1_t, harmonize_jump, x1_sigma_ts, remove_COM_from_noise = False, batch = None, mask = None)
            
            x2_pos_t, x2_t_jump = forward_jump(x2_pos_t, x2_t, harmonize_jump, x2_sigma_ts, remove_COM_from_noise = False, batch = x2_batch, mask = ~virtual_node_mask_x2)
            
            x3_pos_t, x3_t_jump = forward_jump(x3_pos_t, x3_t, harmonize_jump, x3_sigma_ts, remove_COM_from_noise = False, batch = x3_batch, mask = ~virtual_node_mask_x3)
            x3_x_t, x3_t_jump = forward_jump(x3_x_t, x3_t, harmonize_jump, x3_sigma_ts, remove_COM_from_noise = False, batch = x3_batch, mask = ~virtual_node_mask_x3)
            
            x4_pos_t, x4_t_jump = forward_jump(x4_pos_t, x4_t, harmonize_jump, x4_sigma_ts, remove_COM_from_noise = False, batch = x4_batch, mask = ~virtual_node_mask_x4)
            x4_direction_t, x4_t_jump = forward_jump(x4_direction_t, x4_t, harmonize_jump, x4_sigma_ts, remove_COM_from_noise = False, batch = x4_batch, mask = ~virtual_node_mask_x4)
            x4_x_t, x4_t_jump = forward_jump(x4_x_t, x4_t, harmonize_jump, x4_sigma_ts, remove_COM_from_noise = False, batch = x4_batch, mask = ~virtual_node_mask_x4)
    
            
            x1_t = x1_t_jump
            x2_t = x2_t_jump
            x3_t = x3_t_jump
            x4_t = x4_t_jump
            
            assert x1_t == x2_t
            assert x1_t == x3_t
            assert x1_t == x4_t
            t = x1_t
        
        if (x2_t > stop_inpainting_at_time_x2) and inpaint_x2_pos:
            x2_pos_t = torch.cat([x2_pos_inpainting_trajectory[x2_t] for _ in range(batch_size)], dim = 0)        
            noise = torch.randn_like(x2_pos_t)
            noise[virtual_node_mask_x2] = 0.0
            x2_pos_t = x2_pos_t + add_noise_to_inpainted_x2_pos * noise
        
        if (x3_t > stop_inpainting_at_time_x3):
            if inpaint_x3_pos:
                x3_pos_t = torch.cat([x3_pos_inpainting_trajectory[x3_t] for _ in range(batch_size)], dim = 0)        
                noise = torch.randn_like(x3_pos_t)
                noise[virtual_node_mask_x3] = 0.0
                x3_pos_t = x3_pos_t + add_noise_to_inpainted_x3_pos * noise
            if inpaint_x3_x:
                x3_x_t = torch.cat([x3_x_inpainting_trajectory[x3_t] for _ in range(batch_size)], dim = 0)
                noise = torch.randn_like(x3_x_t)
                noise[virtual_node_mask_x3] = 0.0
                x3_x_t = x3_x_t + add_noise_to_inpainted_x3_x * noise
            
        if (x4_t > stop_inpainting_at_time_x4):
            if inpaint_x4_pos:
                x4_pos_t = torch.cat([x4_pos_inpainting_trajectory[x4_t] for _ in range(batch_size)], dim = 0)
                noise = torch.randn_like(x4_pos_t)
                noise[virtual_node_mask_x4] = 0.0
                x4_pos_t = x4_pos_t + add_noise_to_inpainted_x4_pos * noise
            if inpaint_x4_direction:
                x4_direction_t = torch.cat([x4_direction_inpainting_trajectory[x4_t] for _ in range(batch_size)], dim = 0)
                noise = torch.randn_like(x4_direction_t)
                noise[virtual_node_mask_x4] = 0.0
                x4_direction_t = x4_direction_t + add_noise_to_inpainted_x4_direction * noise
            if inpaint_x4_type:
                x4_x_t = torch.cat([x4_x_inpainting_trajectory[x4_t] for _ in range(batch_size)], dim = 0)
                noise = torch.randn_like(x4_x_t)
                noise[virtual_node_mask_x4] = 0.0
                x4_x_t = x4_x_t + add_noise_to_inpainted_x4_type * noise
        
        
        # get noise parameters for current timestep
        x1_t_idx = np.where(params['noise_schedules']['x1']['ts'] == x1_t)[0][0]
        x1_alpha_t = params['noise_schedules']['x1']['alpha_ts'][x1_t_idx]
        x1_sigma_t = params['noise_schedules']['x1']['sigma_ts'][x1_t_idx]
        x1_alpha_dash_t = params['noise_schedules']['x1']['alpha_dash_ts'][x1_t_idx]
        x1_var_dash_t = params['noise_schedules']['x1']['var_dash_ts'][x1_t_idx]
        x1_sigma_dash_t = params['noise_schedules']['x1']['sigma_dash_ts'][x1_t_idx]
        
        # get noise parameters for next timestep
        if x1_t_idx > 0:
            x1_t_1 = x1_t - 1
            x1_t_1_idx = x1_t_idx - 1
            x1_alpha_t_1 = params['noise_schedules']['x1']['alpha_ts'][x1_t_1_idx]
            x1_sigma_t_1 = params['noise_schedules']['x1']['sigma_ts'][x1_t_1_idx]
            x1_alpha_dash_t_1 = params['noise_schedules']['x1']['alpha_dash_ts'][x1_t_1_idx]
            x1_var_dash_t_1 = params['noise_schedules']['x1']['var_dash_ts'][x1_t_1_idx]
            x1_sigma_dash_t_1 = params['noise_schedules']['x1']['sigma_dash_ts'][x1_t_1_idx]
            
        
        # get noise parameters for current timestep
        x2_t_idx = np.where(params['noise_schedules']['x2']['ts'] == x2_t)[0][0]
        x2_alpha_t = params['noise_schedules']['x2']['alpha_ts'][x2_t_idx]
        x2_sigma_t = params['noise_schedules']['x2']['sigma_ts'][x2_t_idx]
        x2_alpha_dash_t = params['noise_schedules']['x2']['alpha_dash_ts'][x2_t_idx]
        x2_var_dash_t = params['noise_schedules']['x2']['var_dash_ts'][x2_t_idx]
        x2_sigma_dash_t = params['noise_schedules']['x2']['sigma_dash_ts'][x2_t_idx]
        
        # get noise parameters for next timestep
        if x2_t_idx > 0:
            x2_t_1 = x2_t - 1
            x2_t_1_idx = x2_t_idx - 1
            x2_alpha_t_1 = params['noise_schedules']['x2']['alpha_ts'][x2_t_1_idx]
            x2_sigma_t_1 = params['noise_schedules']['x2']['sigma_ts'][x2_t_1_idx]
            x2_alpha_dash_t_1 = params['noise_schedules']['x2']['alpha_dash_ts'][x2_t_1_idx]
            x2_var_dash_t_1 = params['noise_schedules']['x2']['var_dash_ts'][x2_t_1_idx]
            x2_sigma_dash_t_1 = params['noise_schedules']['x2']['sigma_dash_ts'][x2_t_1_idx]
        
        
        # get noise parameters for current timestep
        x3_t_idx = np.where(params['noise_schedules']['x3']['ts'] == x3_t)[0][0]
        x3_alpha_t = params['noise_schedules']['x3']['alpha_ts'][x3_t_idx]
        x3_sigma_t = params['noise_schedules']['x3']['sigma_ts'][x3_t_idx]
        x3_alpha_dash_t = params['noise_schedules']['x3']['alpha_dash_ts'][x3_t_idx]
        x3_var_dash_t = params['noise_schedules']['x3']['var_dash_ts'][x3_t_idx]
        x3_sigma_dash_t = params['noise_schedules']['x3']['sigma_dash_ts'][x3_t_idx]
        
        # get noise parameters for next timestep
        if x3_t_idx > 0:
            x3_t_1 = x3_t - 1
            x3_t_1_idx = x3_t_idx - 1
            x3_alpha_t_1 = params['noise_schedules']['x3']['alpha_ts'][x3_t_1_idx]
            x3_sigma_t_1 = params['noise_schedules']['x3']['sigma_ts'][x3_t_1_idx]
            x3_alpha_dash_t_1 = params['noise_schedules']['x3']['alpha_dash_ts'][x3_t_1_idx]
            x3_var_dash_t_1 = params['noise_schedules']['x3']['var_dash_ts'][x3_t_1_idx]
            x3_sigma_dash_t_1 = params['noise_schedules']['x3']['sigma_dash_ts'][x3_t_1_idx]
        
        # get noise parameters for current timestep
        x4_t_idx = np.where(params['noise_schedules']['x4']['ts'] == x4_t)[0][0]
        x4_alpha_t = params['noise_schedules']['x4']['alpha_ts'][x4_t_idx]
        x4_sigma_t = params['noise_schedules']['x4']['sigma_ts'][x4_t_idx]
        x4_alpha_dash_t = params['noise_schedules']['x4']['alpha_dash_ts'][x4_t_idx]
        x4_var_dash_t = params['noise_schedules']['x4']['var_dash_ts'][x4_t_idx]
        x4_sigma_dash_t = params['noise_schedules']['x4']['sigma_dash_ts'][x4_t_idx]
        
        # get noise parameters for next timestep
        if x4_t_idx > 0:
            x4_t_1 = x4_t - 1
            x4_t_1_idx = x4_t_idx - 1
            x4_alpha_t_1 = params['noise_schedules']['x4']['alpha_ts'][x4_t_1_idx]
            x4_sigma_t_1 = params['noise_schedules']['x4']['sigma_ts'][x4_t_1_idx]
            x4_alpha_dash_t_1 = params['noise_schedules']['x4']['alpha_dash_ts'][x4_t_1_idx]
            x4_var_dash_t_1 = params['noise_schedules']['x4']['var_dash_ts'][x4_t_1_idx]
            x4_sigma_dash_t_1 = params['noise_schedules']['x4']['sigma_dash_ts'][x4_t_1_idx]
        
        
        
        # get current data
        x1_timestep = torch.tensor([x1_t] * batch_size)
        x2_timestep = torch.tensor([x2_t] * batch_size)
        x3_timestep = torch.tensor([x3_t] * batch_size)
        x4_timestep = torch.tensor([x4_t] * batch_size)
        
        x1_sigma_dash_t_ = torch.tensor([x1_sigma_dash_t] * batch_size, dtype = torch.float)
        x1_alpha_dash_t_ = torch.tensor([x1_alpha_dash_t] * batch_size, dtype = torch.float)
        
        x2_sigma_dash_t_ = torch.tensor([x2_sigma_dash_t] * batch_size, dtype = torch.float)
        x2_alpha_dash_t_ = torch.tensor([x2_alpha_dash_t] * batch_size, dtype = torch.float)
        
        x3_sigma_dash_t_ = torch.tensor([x3_sigma_dash_t] * batch_size, dtype = torch.float)
        x3_alpha_dash_t_ = torch.tensor([x3_alpha_dash_t] * batch_size, dtype = torch.float)
        
        x4_sigma_dash_t_ = torch.tensor([x4_sigma_dash_t] * batch_size, dtype = torch.float)
        x4_alpha_dash_t_ = torch.tensor([x4_alpha_dash_t] * batch_size, dtype = torch.float)
        
        
        
        input_dict = {}
        input_dict['device'] = model_pl.model.device
        input_dict['dtype'] = torch.float32
        input_dict['x1'] =  {
            
            # the decoder uses the forward-noised structures
            'decoder': {
                'pos': x1_pos_t.to(input_dict['device']), # this is the structure after forward-noising
                'x': x1_x_t.to(input_dict['device']), # this is the structure after forward-noising
                'batch': x1_batch.to(input_dict['device']),
                
                'bond_edge_x': x1_bond_edge_x_t.to(input_dict['device']), # this is the structure after forward-noising
                'bond_edge_index': bond_edge_index_x1.to(input_dict['device']),
                
                'timestep': x1_timestep.to(input_dict['device']),
                'sigma_dash_t': x1_sigma_dash_t_.to(input_dict['device']),
                'alpha_dash_t': x1_alpha_dash_t_.to(input_dict['device']),
    
                'virtual_node_mask': virtual_node_mask_x1.to(input_dict['device']),
                
            },
        }    
        
        input_dict['x2'] =  {
            
            # the decoder uses the forward-noised structures
            'decoder': {
                'pos': x2_pos_t.to(input_dict['device']), # this is the structure after forward-noising
                'x': x2_x_t.to(input_dict['device']), # this is the structure after forward-noising
                'batch': x2_batch.to(input_dict['device']),
                
                'timestep': x2_timestep.to(input_dict['device']),
                'sigma_dash_t': x2_sigma_dash_t_.to(input_dict['device']),
                'alpha_dash_t': x2_alpha_dash_t_.to(input_dict['device']),
                
                'virtual_node_mask': virtual_node_mask_x2.to(input_dict['device']),
                
            },
        }
        
        input_dict['x3'] =  {
            
            # the decoder uses the forward-noised structures
            'decoder': {
                'pos': x3_pos_t.to(input_dict['device']), # this is the structure after forward-noising
                'x': x3_x_t.to(input_dict['device']), # this is the structure after forward-noising
                'batch': x3_batch.to(input_dict['device']),
                
                'timestep': x3_timestep.to(input_dict['device']),
                'sigma_dash_t': x3_sigma_dash_t_.to(input_dict['device']),
                'alpha_dash_t': x3_alpha_dash_t_.to(input_dict['device']),
                
                'virtual_node_mask': virtual_node_mask_x3.to(input_dict['device']),
                
            },
        }
        
        input_dict['x4'] =  {
            
            # the decoder uses the forward-noised structures
            'decoder': {
                'x': x4_x_t.to(input_dict['device']), # this is the structure after forward-noising
                'pos': x4_pos_t.to(input_dict['device']), # this is the structure after forward-noising
                'direction': x4_direction_t.to(input_dict['device']), # this is the structure after forward-noising
                'batch': x4_batch.to(input_dict['device']),
                
                'timestep': x4_timestep.to(input_dict['device']),
                'sigma_dash_t': x4_sigma_dash_t_.to(input_dict['device']),
                'alpha_dash_t': x4_alpha_dash_t_.to(input_dict['device']),
                
                'virtual_node_mask': virtual_node_mask_x4.to(input_dict['device']),
                
            },
        }
        

        # predict noise with neural network    
        with torch.no_grad():
            _, output_dict = model_pl.model.forward(input_dict)
        
        x1_x_out = output_dict['x1']['decoder']['denoiser']['x_out'].detach().cpu()
        x1_bond_edge_x_out = output_dict['x1']['decoder']['denoiser']['bond_edge_x_out'].detach().cpu()
        x1_pos_out = output_dict['x1']['decoder']['denoiser']['pos_out'].detach().cpu()
        x1_pos_out = x1_pos_out - torch_scatter.scatter_mean(x1_pos_out[~virtual_node_mask_x1], x1_batch[~virtual_node_mask_x1], dim = 0)[x1_batch] # removing COM from predicted noise 
        
        x1_x_out[virtual_node_mask_x1, :] = 0.0
        x1_pos_out[virtual_node_mask_x1, :] = 0.0
        
        
        x2_pos_out = output_dict['x2']['decoder']['denoiser']['pos_out']
        if x2_pos_out is not None:
            x2_pos_out = x2_pos_out.detach().cpu() # NOT removing COM from predicted positional noise for x3
            x2_pos_out[virtual_node_mask_x2, :] = 0.0
        else:
            x2_pos_out = torch.zeros_like(x2_pos_t)
            
        
        x3_pos_out = output_dict['x3']['decoder']['denoiser']['pos_out']
        x3_x_out = output_dict['x3']['decoder']['denoiser']['x_out']
        if x3_pos_out is not None:
            x3_pos_out = x3_pos_out.detach().cpu() # NOT removing COM from predicted positional noise for x3
            x3_pos_out[virtual_node_mask_x3, :] = 0.0
            
            x3_x_out = x3_x_out.detach().cpu()
            x3_x_out = x3_x_out.squeeze()
            x3_x_out[virtual_node_mask_x3] = 0.0
        else:
            x3_pos_out = torch.zeros_like(x3_pos_t)
            x3_x_out = torch.zeros_like(x3_x_t)
        
        
        x4_x_out = output_dict['x4']['decoder']['denoiser']['x_out']
        x4_pos_out = output_dict['x4']['decoder']['denoiser']['pos_out']
        x4_direction_out = output_dict['x4']['decoder']['denoiser']['direction_out']
        if x4_x_out is not None:
            x4_pos_out = x4_pos_out.detach().cpu() # NOT removing COM from predicted positional noise for x4
            x4_pos_out[virtual_node_mask_x4, :] = 0.0
            
            x4_direction_out = x4_direction_out.detach().cpu() # NOT removing COM from predicted positional noise for x4
            x4_direction_out[virtual_node_mask_x4, :] = 0.0
            
            x4_x_out = x4_x_out.detach().cpu()
            x4_x_out = x4_x_out.squeeze()
            x4_x_out[virtual_node_mask_x4] = 0.0
        
        else:
            x4_pos_out = torch.zeros_like(x4_pos_t)
            x4_direction_out = torch.zeros_like(x4_direction_t)
            x4_x_out = torch.zeros_like(x4_x_t)
        
        
        
        # get added noise - x1
        x1_pos_epsilon = torch.randn(x1_batch_size_nodes, 3)
        x1_pos_epsilon = x1_pos_epsilon - torch_scatter.scatter_mean(x1_pos_epsilon[~virtual_node_mask_x1], x1_batch[~virtual_node_mask_x1], dim = 0)[x1_batch] # removing COM from added noise
        x1_pos_epsilon[virtual_node_mask_x1, :] = 0.0
        
        x1_x_epsilon = torch.randn(x1_batch_size_nodes, num_atom_types)    
        x1_x_epsilon[virtual_node_mask_x1, :] = 0.0
        
        x1_bond_edge_x_epsilon = torch.randn_like(x1_bond_edge_x_out)
        
        x1_c_t = (x1_sigma_t * x1_sigma_dash_t_1) / (x1_sigma_dash_t) if x1_t_idx > 0 else 0
        x1_c_t = x1_c_t * denoising_noise_scale
        
        
        # get added noise - x2
        x2_pos_epsilon = torch.randn(x2_batch_size_nodes,3)
        x2_pos_epsilon[virtual_node_mask_x2, :] = 0.0
        
        x2_c_t = (x2_sigma_t * x2_sigma_dash_t_1) / (x2_sigma_dash_t) if x2_t_idx > 0 else 0
        x2_c_t = x2_c_t * denoising_noise_scale
        
        
        # get added noise - x3
        x3_pos_epsilon = torch.randn(x3_batch_size_nodes,3)
        x3_pos_epsilon[virtual_node_mask_x3, :] = 0.0
        
        x3_x_epsilon = torch.randn(x3_batch_size_nodes)    
        x3_x_epsilon[virtual_node_mask_x3, ...] = 0.0
       
        x3_c_t = (x3_sigma_t * x3_sigma_dash_t_1) / (x3_sigma_dash_t) if x3_t_idx > 0 else 0
        x3_c_t = x3_c_t * denoising_noise_scale
        
        
        # get added noise - x4
        x4_pos_epsilon = torch.randn(x4_batch_size_nodes,3)
        x4_pos_epsilon[virtual_node_mask_x4, :] = 0.0
        
        x4_direction_epsilon = torch.randn(x4_batch_size_nodes, 3)
        x4_direction_epsilon[virtual_node_mask_x4, :] = 0.0
        
        x4_x_epsilon = torch.randn(x4_batch_size_nodes, num_pharm_types)    
        x4_x_epsilon[virtual_node_mask_x4, ...] = 0.0
       
        x4_c_t = (x4_sigma_t * x4_sigma_dash_t_1) / (x4_sigma_dash_t) if x4_t_idx > 0 else 0
        x4_c_t = x4_c_t * denoising_noise_scale
        
        
        # (intended for symmetry breaking, but could also be used for increasing diversity of samples)
        x1_c_t_injected = x1_c_t
        x2_c_t_injected = x2_c_t
        x3_c_t_injected = x3_c_t
        x4_c_t_injected = x4_c_t
        if len(inject_noise_at_ts) > 0:
            if t == inject_noise_at_ts[0]:
                #print(f'injecting noise... at time {t}')
                inject_noise_at_ts.pop(0)
                inject_noise_scale = inject_noise_scales.pop(0)
                
                # extra noisy, only applied to positions to break symmetry
                x1_c_t_injected = x1_c_t + inject_noise_scale
                x2_c_t_injected = x2_c_t + inject_noise_scale
                x3_c_t_injected = x3_c_t + inject_noise_scale
                x4_c_t_injected = x4_c_t + inject_noise_scale
        
        
        # reverse denoising step - x1
        x1_pos_t_1 = ((1. / x1_alpha_t) * x1_pos_t)  - ((x1_var_dash_t/(x1_alpha_t * x1_sigma_dash_t)) * x1_pos_out)  +  (x1_c_t_injected * x1_pos_epsilon)
        x1_x_t_1 = ((1. / x1_alpha_t) * x1_x_t)  - ((x1_var_dash_t/(x1_alpha_t * x1_sigma_dash_t)) * x1_x_out)  +  (x1_c_t * x1_x_epsilon)
        x1_bond_edge_x_t_1 = ((1. / x1_alpha_t) * x1_bond_edge_x_t)  - ((x1_var_dash_t/(x1_alpha_t * x1_sigma_dash_t)) * x1_bond_edge_x_out)  +  (x1_c_t * x1_bond_edge_x_epsilon)
        
        # reverse denoising step - x2
        x2_pos_t_1 = ((1. / float(x2_alpha_t)) * x2_pos_t)  - ((x2_var_dash_t/(x2_alpha_t * x2_sigma_dash_t)) * x2_pos_out)  +  (x2_c_t_injected * x2_pos_epsilon)
        x2_x_t_1 = x2_x_t
    
        # reverse denoising step - x3
        x3_pos_t_1 = ((1. / float(x3_alpha_t)) * x3_pos_t)  - ((x3_var_dash_t/(x3_alpha_t * x3_sigma_dash_t)) * x3_pos_out)  +  (x3_c_t_injected * x3_pos_epsilon)
        x3_x_t_1 = ((1. / x3_alpha_t) * x3_x_t)  - ((x3_var_dash_t/(x3_alpha_t * x3_sigma_dash_t)) * x3_x_out)  +  (x3_c_t * x3_x_epsilon)
    
        # reverse denoising step - x4
        x4_pos_t_1 = ((1. / float(x4_alpha_t)) * x4_pos_t)  - ((x4_var_dash_t/(x4_alpha_t * x4_sigma_dash_t)) * x4_pos_out)  +  (x4_c_t_injected * x4_pos_epsilon)
        x4_direction_t_1 = ((1. / float(x4_alpha_t)) * x4_direction_t)  - ((x4_var_dash_t/(x4_alpha_t * x4_sigma_dash_t)) * x4_direction_out)  +  (x4_c_t * x4_direction_epsilon)
        x4_x_t_1 = ((1. / x4_alpha_t) * x4_x_t)  - ((x4_var_dash_t/(x4_alpha_t * x4_sigma_dash_t)) * x4_x_out)  +  (x4_c_t * x4_x_epsilon)
    
        
        # resetting virtual nodes
        x1_pos_t_1[virtual_node_mask_x1] = x1_pos_t[virtual_node_mask_x1]
        x1_x_t_1[virtual_node_mask_x1] = x1_x_t[virtual_node_mask_x1]
        
        x2_pos_t_1[virtual_node_mask_x2] = x2_pos_t[virtual_node_mask_x2]
        x2_x_t_1[virtual_node_mask_x2] = x2_x_t[virtual_node_mask_x2]
        
        x3_pos_t_1[virtual_node_mask_x3] = x3_pos_t[virtual_node_mask_x3]
        x3_x_t_1[virtual_node_mask_x3] = x3_x_t[virtual_node_mask_x3]
        
        x4_pos_t_1[virtual_node_mask_x4] = x4_pos_t[virtual_node_mask_x4]
        x4_direction_t_1[virtual_node_mask_x4] = x4_direction_t[virtual_node_mask_x4]
        x4_x_t_1[virtual_node_mask_x4] = x4_x_t[virtual_node_mask_x4]
        
        
        # saving intermediate states for visualization / tracking
        x1_t_x_list.append(x1_x_t.detach().cpu().numpy())
        x1_t_bond_edge_x_list.append(x1_bond_edge_x_t.detach().cpu().numpy())
        x1_t_pos_list.append(x1_pos_t.detach().cpu().numpy())
        
        x2_t_pos_list.append(x2_pos_t.detach().cpu().numpy())
            
        x3_t_pos_list.append(x3_pos_t.detach().cpu().numpy())
        x3_t_x_list.append(x3_x_t.detach().cpu().numpy())
        
        x4_t_pos_list.append(x4_pos_t.detach().cpu().numpy())
        x4_t_direction_list.append(x4_direction_t.detach().cpu().numpy())
        x4_t_x_list.append(x4_x_t.detach().cpu().numpy())
        
        
        # set next state and iterate
        x1_pos_t = x1_pos_t_1
        x1_x_t = x1_x_t_1
        x1_bond_edge_x_t = x1_bond_edge_x_t_1
        
        x2_pos_t = x2_pos_t_1
        x2_x_t = x2_x_t_1
        
        x3_pos_t = x3_pos_t_1
        x3_x_t = x3_x_t_1
        
        x4_pos_t = x4_pos_t_1
        x4_direction_t = x4_direction_t_1
        x4_x_t = x4_x_t_1
        
        t = t - 1
        x1_t = x1_t - 1
        x2_t = x2_t - 1
        x3_t = x3_t - 1
        x4_t = x4_t - 1
    
        pbar.update(1)
        
        # this is necessary for clearing CUDA memory
        del output_dict
        del input_dict    
        
    pbar.close()
    
    
    
    ####### Extracting final structures, and re-scaling ########
    
    x2_pos_final = x2_pos_t[~virtual_node_mask_x2].numpy()

    x3_pos_final = x3_pos_t[~virtual_node_mask_x3].numpy()
    x3_x_final = x3_x_t[~virtual_node_mask_x3].numpy()
    x3_x_final = x3_x_final / params['dataset']['x3']['scale_node_features']
    
    x4_x_final = np.argmin(np.abs(x4_x_t[~virtual_node_mask_x4] - params['dataset']['x4']['scale_node_features']), axis = -1)
    x4_x_final = x4_x_final - 1 # readjusting for the previous addition of the virtual node pharmacophore type
    x4_pos_final = x4_pos_t[~virtual_node_mask_x4].numpy()
    
    x4_direction_final = x4_direction_t[~virtual_node_mask_x4].numpy() / params['dataset']['x4']['scale_vector_features']
    x4_direction_final_norm = np.linalg.norm(x4_direction_final, axis = 1)
    x4_direction_final[x4_direction_final_norm < 0.5] = 0.0
    x4_direction_final[x4_direction_final_norm >= 0.5] = x4_direction_final[x4_direction_final_norm >= 0.5] / x4_direction_final_norm[x4_direction_final_norm >= 0.5][..., None]
    
    
    x1_x_t[~virtual_node_mask_x1, 0] = -np.inf # this masks out remaining probability assigned to virtual nodes
    x1_pos_final = x1_pos_t[~virtual_node_mask_x1].numpy()
    x1_x_final = np.argmin(np.abs(x1_x_t[~virtual_node_mask_x1, 0:-len(params['dataset']['x1']['charge_types'])] - params['dataset']['x1']['scale_atom_features']), axis = -1)
    x1_bond_edge_x_final = np.argmin(np.abs(x1_bond_edge_x_t - params['dataset']['x1']['scale_bond_features']), axis = -1)
    
    # need to remap the indices in x1_x_final to the list of atom types
    atomic_number_remapping = torch.tensor([0,1,6,7,8,9,17,35,53,16,15,14]) # [None, 'H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P', 'Si']
    x1_x_final = atomic_number_remapping[x1_x_final]
    
    
    # return generated structures
    generated_structures = []
    for b in range(batch_size):
        generated_dict = {
            'x1': {
                'atoms': np.split(x1_x_final.numpy(), batch_size)[b],
                #'formal_charges': None, # still need to extract from x1_x_t[~virtual_node_mask_x1, -len(params['dataset']['x1']['charge_types']):]
                'bonds': np.split(x1_bond_edge_x_final.numpy(), batch_size)[b],
                'positions': np.split(x1_pos_final, batch_size)[b],
            },
            'x2': {
                'positions': np.split(x2_pos_final, batch_size)[b],
            },
            'x3': {
                'charges': np.split(x3_x_final, batch_size)[b], # electrostatic potential
                'positions': np.split(x3_pos_final, batch_size)[b],
            },
            'x4': {
                'types': np.split(x4_x_final.numpy(), batch_size)[b],
                'positions': np.split(x4_pos_final, batch_size)[b],
                'directions': np.split(x4_direction_final, batch_size)[b],
            },
        }
        generated_structures.append(generated_dict)
    
    return generated_structures
