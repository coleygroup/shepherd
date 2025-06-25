"""
This module contains the inference sampler for the ShEPhERD model.
"""
import os
import sys
import pathlib
from tqdm import tqdm
import pickle
from copy import deepcopy
from functools import partial
from typing import Optional

import open3d
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch_geometric
import torch_scatter

import pytorch_lightning as pl
from shepherd.lightning_module import LightningModule
from shepherd.datasets import HeteroDataset

from shepherd.inference.initialization import (
    _initialize_x1_state,
    _initialize_x2_state,
    _initialize_x3_state,
    _initialize_x4_state
)
from shepherd.inference.noise import (
    forward_trajectory,
    _get_noise_params_for_timestep,
    forward_jump
)
from shepherd.inference.steps import (
    _perform_reverse_denoising_step,
    _prepare_model_input,
    _inference_step,
    _extract_generated_samples
)

def generate(
    model_pl: LightningModule,
    batch_size: int,

    N_x1: int,
    N_x4: int,

    unconditional: bool,

    prior_noise_scale: float = 1.0,
    denoising_noise_scale: float = 1.0,

    inject_noise_at_ts: list[int] = [],
    inject_noise_scales: list[int] = [],    

    harmonize: bool = False,
    harmonize_ts: list[int] = [],
    harmonize_jumps: list[int] = [],

    # all the below options are only relevant if unconditional is False

    inpaint_x1_pos: bool = False,
    inpaint_x1_x: bool = False,

    inpaint_x2_pos: bool = False,
    inpaint_x3_pos: bool = False,
    inpaint_x3_x: bool = False,
    inpaint_x4_pos: bool = False,
    inpaint_x4_direction: bool = False,
    inpaint_x4_type: bool = False,

    stop_inpainting_at_time_x1: float = 0.0,

    stop_inpainting_at_time_x2: float = 0.0,
    add_noise_to_inpainted_x2_pos: float = 0.0,

    stop_inpainting_at_time_x3: float = 0.0,
    add_noise_to_inpainted_x3_pos: float = 0.0,
    add_noise_to_inpainted_x3_x: float = 0.0,

    stop_inpainting_at_time_x4: float = 0.0,
    add_noise_to_inpainted_x4_pos: float = 0.0,
    add_noise_to_inpainted_x4_direction: float = 0.0,
    add_noise_to_inpainted_x4_type: float = 0.0,

    # these are the inpainting targets
    atom_types: Optional[list[int]] = None,
    atom_pos: Optional[np.ndarray] = None,
    center_of_mass: np.ndarray = np.zeros(3),
    surface: np.ndarray = np.zeros((75,3)),
    electrostatics: np.ndarray = np.zeros(75),
    pharm_types: np.ndarray = np.zeros(5, dtype = int),
    pharm_pos: np.ndarray = np.zeros((5,3)),
    pharm_direction: np.ndarray = np.zeros((5,3)),
    verbose: bool = True
    ) -> list[dict]:
    """
    Runs inference of ShEPhERD to sample `batch_size` number of molecules.

    Arguments
    ---------
    model_pl : PyTorch Lightning module.

    batch_size : int Number of molecules to sample in a single batch.

    N_x1 : int Number of atoms to diffuse.
    N_x4 : int Number of pharmacophores to diffuse.
        If inpainting, can be greater than len(pharm_types) for partial pharmacophore conditioning.

    unconditional : bool to toggle unconditional generation.

    prior_noise_scale : float (default = 1.0) Noise scale of the prior distribution.
    denoising_noise_scale : float (default = 1.0) Noise scale for each denoising step.
    
    inject_noise_at_ts : list[int] (default = []) Time steps to inject extra noise.
    inject_noise_scales : list[int] (default = []) Scale of noise to inject at above time steps.
     
    harmonize : bool (default=False) Whether to use harmonization.
    harmonize_ts : list[int] (default = []) Time steps to to harmonization.
    harmonize_jumps : list[int] (default = []) Length of time to harmonize (in time steps).

    *all the below options are only relevant if unconditional is False*

    inpaint_x1_pos : bool (default=False)
    inpaint_x1_x : bool (default=False)

    inpaint_x2_pos : bool (default=False) Toggle inpainting.
        Note that x2 is implicitly modeled via x3.

    inpaint_x3_pos : bool (default=False)
    inpaint_x3_x : bool (default=False)

    inpaint_x4_pos : bool (default=False)
    inpaint_x4_direction : bool (default=False)
    inpaint_x4_type : bool (default=False)
    
    stop_inpainting_at_time_x1 : float (default = 0.0) Time step to stop inpainting.
        t=0.0 implies that inpainting doesn't stop.

    stop_inpainting_at_time_x2 : float (default = 0.0) Time step to stop inpainting.
        t=0.0 implies that inpainting doesn't stop.
    add_noise_to_inpainted_x2_pos : float (default = 0.0) Scale of noise to add to inpainted
        values.
    
    stop_inpainting_at_time_x3 : float (default = 0.0)
    add_noise_to_inpainted_x3_pos : float (default = 0.0)
    add_noise_to_inpainted_x3_x : float (default = 0.0)
    
    stop_inpainting_at_time_x4 : float (default = 0.0)
    add_noise_to_inpainted_x4_pos : float (default = 0.0)
    add_noise_to_inpainted_x4_direction : float (default = 0.0)
    add_noise_to_inpainted_x4_type : float (default = 0.0)
    
    *these are the inpainting targets*
    atom_types : Optional[list[int]] (default = None) Atom elements expected as a list of atomic numbers.
    atom_pos : Optional[np.ndarray] (default = None) Atom positions as coordinates.
    center_of_mass : np.ndarray (3,) (default = np.zeros(3)) Must be supplied if target molecule is
        not already centered.
    surface : np.ndarray (75,3) (default = np.zeros((75,3)) Surface point coordinates.
    electrostatics : np.ndarray (75,) (default = np.zeros(75)) Electrostatics at each surface point.
    pharm_types : np.ndarray (<=N_x4,) (default = np.zeros(5, dtype = int)) Pharmacophore types.
    pharm_pos : np.ndarray (<=N_x4,3) (default = np.zeros((5,3))) Pharmacophore positions as
        coordinates.
    pharm_direction : np.ndarray (<=N_x4,3) (default = np.zeros((5,3))) Pharmacophore directions as
        unit vectors.

    save_intermediate : bool (default=False)
        whether to save intermediates --> not implemented

    start_t_ind : int (default = 0)
        Index of the time step to start from (default is from pure noise).
        Requires xi_initial_dict to be provided.
    xi_initial_dict : Optional[dict] (default = None)
        Dictionary containing the initial states of x1, x2, x3, and x4.
        If None, the states are initialized randomly.
    noise_dict: Optional[dict] (default = None)
        Dictionary containing the noise parameters for the *first* inference step.
        After the first inference step, the noise will be sampled randomly.
        If None, the noises will be sampled randomly.

    do_property_cfg : bool (default = False) Whether to use property conditioning.
    cfg_weight : float (default = 3.0) Weight of property conditioning.
    sa_score : float (default = 1.0) SA score of the target molecule.
        Range is 0-10: 10 is difficult to synthesize.

    verbose : bool (default = True) Whether to print progress bar.

    Returns
    -------
    generated_structures : List[Dict]
        Output dictionary is structured as:
        'x1': {
                'atoms': np.ndarray (N_x1,) of ints for atomic numbers.
                'bonds': np.ndarray of bond types between every atom pair.
                'positions': np.ndarray (N_x1, 3) Coordinates of atoms.
            },
            'x2': {
                'positions': np.ndarray (75, 3) Coordinates of surface points.
            },
            'x3': {
                'charges': np.ndarray (75, 3) ESP at surface points.
                'positions': np.ndarray (75, 3) Coordinates of surface points.
            },
            'x4': {
                'types': np.ndarray (N_x4,) of ints for pharmacophore types.
                'positions': np.ndarray (N_x4, 3) Coordinates of pharmacophores.
                'directions': np.ndarray (N_x4, 3) Unit vectors of pharmacophores.
            },
        }
    """
    params = model_pl.params

    T = params['noise_schedules']['x1']['ts'].max()
    time_steps = np.arange(T, 0, -1) # Full sequence [T, T-1, ..., 1]

    N_x2 = params['dataset']['x2']['num_points']
    N_x3 = params['dataset']['x3']['num_points']

    ####### Defining inpainting targets ########

    # override conditioning options
    if unconditional:
        inpaint_x1_pos = False
        inpaint_x1_x = False
        inpaint_x2_pos = False
        inpaint_x3_pos = False
        inpaint_x3_x = False
        inpaint_x4_pos = False
        inpaint_x4_direction = False
        inpaint_x4_type = False

    do_partial_pharm_inpainting = False
    assert len(pharm_direction) == len(pharm_pos) and len(pharm_pos) == len(pharm_types)
    assert N_x4 >= len(pharm_pos)
    if N_x4 > len(pharm_pos):
        do_partial_pharm_inpainting = True

    do_partial_atom_inpainting = False
    if atom_pos is not None:
        try:
            assert N_x1 >= len(atom_pos)
        except AssertionError:
            raise ValueError(f"Number of atoms in the target molecule ({len(atom_pos)}) must be less than or equal to the number of atoms in the model ({N_x1}).")
        if N_x1 > len(atom_pos):
            do_partial_atom_inpainting = True
        else:
            do_partial_atom_inpainting = False
    if atom_types is not None:
        try:
            assert len(atom_types) == len(atom_pos)
        except AssertionError:
            raise ValueError(f"Number of atom types in the target molecule ({len(atom_types)}) must be equal to the number of atom positions to inpaint ({len(atom_pos)}).")
        ptable = Chem.GetPeriodicTable()
        # convert atomic numbers to symbols
        atomic_number_to_symbol = {
            ptable.GetAtomicNumber(symbol): symbol for symbol in params['dataset']['x1']['atom_types'] if isinstance(symbol, str)
        }
        atom_types = [atomic_number_to_symbol[z] for z in atom_types]

    if atom_pos is None:
        # initialize dummy atom positions
        atom_pos = np.zeros((N_x1, 3))
    if atom_types is None:
        # initialize dummy atom types
        atom_types = np.zeros(N_x1, dtype = int)

    # centering about provided center of mass (of x1)
    surface = surface - center_of_mass
    pharm_pos = pharm_pos - center_of_mass
    atom_pos = atom_pos - center_of_mass

    # adding small noise to pharm_pos to avoid overlapping points (causes error when encoding clean structure)
    pharm_pos = pharm_pos + np.random.randn(*pharm_pos.shape) * 0.01

    # accounting for virtual nodes
    surface = np.concatenate([np.array([[0.0, 0.0, 0.0]]), surface], axis = 0) # virtual node
    electrostatics = np.concatenate([np.array([0.0]), electrostatics], axis = 0) # virtual node
    pharm_types = pharm_types + 1 # accounting for virtual node as the zeroeth type
    pharm_types = np.concatenate([np.array([0]), pharm_types], axis = 0) # virtual node
    pharm_pos = np.concatenate([np.array([[0.0, 0.0, 0.0]]), pharm_pos], axis = 0) # virtual node
    pharm_direction = np.concatenate([np.array([[0.0, 0.0, 0.0]]), pharm_direction], axis = 0) # virtual node
    
    # TODO EXPECT ATOMIC SYMBOLS: ADJUST INPUT
    atom_pos = np.concatenate([np.array([[0.0, 0.0, 0.0]]), atom_pos], axis = 0)
    # atom types are converted to one-hot encoding
    atom_type_map = {atomic_symbol: i for i, atomic_symbol in enumerate(params['dataset']['x1']['atom_types'])}
    # virtual node type is 0 (from param file) so don't need to add +1 like in the pharm_types case
    atom_type_indices = np.array([atom_type_map[z] for z in atom_types])
    atom_type_indices = np.concatenate([np.array([0]), atom_type_indices], axis = 0) # virtual node

    num_atom_types = len(params['dataset']['x1']['atom_types']) + len(params['dataset']['x1']['charge_types'])
    atom_types_one_hot = np.zeros((atom_type_indices.size, num_atom_types))
    atom_types_one_hot[np.arange(atom_type_indices.size), atom_type_indices] = 1
    atom_types = atom_types_one_hot

    # one-hot encodings
    pharm_types_one_hot = np.zeros((pharm_types.size, params['dataset']['x4']['max_node_types']))
    pharm_types_one_hot[np.arange(pharm_types.size), pharm_types] = 1
    pharm_types = pharm_types_one_hot

    # scaling features
    electrostatics = electrostatics * params['dataset']['x3']['scale_node_features']
    pharm_types = pharm_types * params['dataset']['x4']['scale_node_features']
    pharm_direction = pharm_direction * params['dataset']['x4']['scale_vector_features']
    atom_types = atom_types * params['dataset']['x1']['scale_atom_features']

    # defining inpainting targets
    target_inpaint_x1_x = torch.as_tensor(atom_types, dtype=torch.float)
    target_inpaint_x1_pos = torch.as_tensor(atom_pos, dtype=torch.float)
    target_inpaint_x1_mask = torch.zeros(atom_pos.shape[0], dtype=torch.long)
    target_inpaint_x1_mask[0] = 1
    target_inpaint_x1_mask = target_inpaint_x1_mask == 0

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

    x1_pos_inpainting_trajectory = None
    x1_x_inpainting_trajectory = None
    x2_pos_inpainting_trajectory = None
    x3_pos_inpainting_trajectory = None
    x3_x_inpainting_trajectory = None
    x4_pos_inpainting_trajectory = None
    x4_direction_inpainting_trajectory = None
    x4_x_inpainting_trajectory = None

    if inpaint_x1_pos:
        x1_pos_inpainting_trajectory = forward_trajectory(
            x = target_inpaint_x1_pos,

            ts = params['noise_schedules']['x1']['ts'],
            alpha_ts = params['noise_schedules']['x1']['alpha_ts'],
            sigma_ts = params['noise_schedules']['x1']['sigma_ts'],
            remove_COM_from_noise = True, # only removes COM from noise, not the x1_pos
            mask = target_inpaint_x1_mask,
            deterministic = False,
        )
    
    if inpaint_x1_x:
        x1_x_inpainting_trajectory = forward_trajectory(
            x = target_inpaint_x1_x,

            ts = params['noise_schedules']['x1']['ts'],
            alpha_ts = params['noise_schedules']['x1']['alpha_ts'],
            sigma_ts = params['noise_schedules']['x1']['sigma_ts'],
            remove_COM_from_noise = False,
            mask = target_inpaint_x1_mask,
            deterministic = False,
        )

    if inpaint_x2_pos:
        x2_pos_inpainting_trajectory = forward_trajectory(
            x = target_inpaint_x2_pos,

            ts = params['noise_schedules']['x2']['ts'],
            alpha_ts = params['noise_schedules']['x2']['alpha_ts'],
            sigma_ts = params['noise_schedules']['x2']['sigma_ts'],
            remove_COM_from_noise = False,
            mask = target_inpaint_x2_mask,
            deterministic = False,
        )

    if inpaint_x3_pos:
        x3_pos_inpainting_trajectory = forward_trajectory(
            x = target_inpaint_x3_pos,

            ts = params['noise_schedules']['x3']['ts'],
            alpha_ts = params['noise_schedules']['x3']['alpha_ts'],
            sigma_ts = params['noise_schedules']['x3']['sigma_ts'],
            remove_COM_from_noise = False,
            mask = target_inpaint_x3_mask,
            deterministic = False,
        )
    if inpaint_x3_x:
        x3_x_inpainting_trajectory = forward_trajectory(
            x = target_inpaint_x3_x,

            ts = params['noise_schedules']['x3']['ts'],
            alpha_ts = params['noise_schedules']['x3']['alpha_ts'],
            sigma_ts = params['noise_schedules']['x3']['sigma_ts'],
            remove_COM_from_noise = False,
            mask = target_inpaint_x3_mask,
            deterministic = False,
        )

    if inpaint_x4_type:
        x4_x_inpainting_trajectory = forward_trajectory(
            x = target_inpaint_x4_x,
            
            ts = params['noise_schedules']['x4']['ts'],
            alpha_ts = params['noise_schedules']['x4']['alpha_ts'],
            sigma_ts = params['noise_schedules']['x4']['sigma_ts'],
            remove_COM_from_noise = False,
            mask = target_inpaint_x4_mask,
            deterministic = False,
        )
    if inpaint_x4_pos:
        x4_pos_inpainting_trajectory = forward_trajectory(
            x = target_inpaint_x4_pos,
            
            ts = params['noise_schedules']['x4']['ts'],
            alpha_ts = params['noise_schedules']['x4']['alpha_ts'],
            sigma_ts = params['noise_schedules']['x4']['sigma_ts'],
            remove_COM_from_noise = False,
            mask = target_inpaint_x4_mask,
            deterministic = False,
        )
    if inpaint_x4_direction:
        x4_direction_inpainting_trajectory = forward_trajectory(
            x = target_inpaint_x4_direction,
            
            ts = params['noise_schedules']['x4']['ts'],
            alpha_ts = params['noise_schedules']['x4']['alpha_ts'],
            sigma_ts = params['noise_schedules']['x4']['sigma_ts'],
            remove_COM_from_noise = False,
            mask = target_inpaint_x4_mask,
            deterministic = False,
        )

    ####################################

    stop_inpainting_at_time_x1 = int(T*stop_inpainting_at_time_x1)
    stop_inpainting_at_time_x2 = int(T*stop_inpainting_at_time_x2)
    stop_inpainting_at_time_x3 = int(T*stop_inpainting_at_time_x3)
    stop_inpainting_at_time_x4 = int(T*stop_inpainting_at_time_x4)

    ###########  Initializing states at t=T   ##############

    include_virtual_node = True
    num_atom_types = len(params['dataset']['x1']['atom_types']) + len(params['dataset']['x1']['charge_types'])
    num_pharm_types = params['dataset']['x4']['max_node_types'] # needed later for inpainting

    # Initialize x1 state
    (pos_forward_noised_x1, x_forward_noised_x1, bond_edge_x_forward_noised_x1, 
    x1_batch, virtual_node_mask_x1, bond_edge_index_x1) = _initialize_x1_state(
        batch_size, N_x1, params, prior_noise_scale, include_virtual_node
    )

    # Initialize x2 state
    pos_forward_noised_x2, x_forward_noised_x2, x2_batch, virtual_node_mask_x2 = _initialize_x2_state(
        batch_size, N_x2, params, prior_noise_scale, include_virtual_node
    )

    # Initialize x3 state
    pos_forward_noised_x3, x_forward_noised_x3, x3_batch, virtual_node_mask_x3 = _initialize_x3_state(
        batch_size, N_x3, params, prior_noise_scale, include_virtual_node
    )

    # Initialize x4 state
    (pos_forward_noised_x4, direction_forward_noised_x4, x_forward_noised_x4, 
    x4_batch, virtual_node_mask_x4) = _initialize_x4_state(
        batch_size, N_x4, params, prior_noise_scale, include_virtual_node
    )

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

    if (x1_t > stop_inpainting_at_time_x1):
        if inpaint_x1_pos:
            x1_pos_t_inpaint = x1_pos_inpainting_trajectory[x1_t].repeat(batch_size, 1, 1)
            if do_partial_atom_inpainting:
                x1_pos_t = x1_pos_t.reshape(batch_size, -1, 3)
                x1_pos_t[:, :x1_pos_t_inpaint.shape[1]] = x1_pos_t_inpaint
                x1_pos_t = x1_pos_t.reshape(-1, 3)
            else:
                x1_pos_t = x1_pos_t_inpaint.reshape(-1, 3)

        if inpaint_x1_x:
            x1_x_t_inpaint = x1_x_inpainting_trajectory[x1_t].repeat(batch_size, 1, 1)
            if do_partial_atom_inpainting:
                x1_x_t = x1_x_t.reshape(batch_size, -1, num_atom_types)
                x1_x_t[:, :x1_x_t_inpaint.shape[1]] = x1_x_t_inpaint
                x1_x_t = x1_x_t.reshape(-1, num_atom_types)
            else:
                x1_x_t = x1_x_t_inpaint.reshape(-1, num_atom_types)

    if (x2_t > stop_inpainting_at_time_x2):
        if inpaint_x2_pos:
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
            x4_pos_t_inpaint = x4_pos_inpainting_trajectory[x4_t].repeat(batch_size, 1, 1)
            if do_partial_pharm_inpainting:
                x4_pos_t = x4_pos_t.reshape(batch_size, -1, 3)
                x4_pos_t[:, :x4_pos_t_inpaint.shape[1]] = x4_pos_t_inpaint
                x4_pos_t = x4_pos_t.reshape(-1, 3)
            else:
                x4_pos_t = x4_pos_t_inpaint.reshape(-1,3)
        if inpaint_x4_direction:
            x4_direction_t_inpaint = x4_direction_inpainting_trajectory[x4_t].repeat(batch_size, 1, 1)
            if do_partial_pharm_inpainting:
                x4_direction_t = x4_direction_t.reshape(batch_size, -1, 3)
                x4_direction_t[:, :x4_direction_t_inpaint.shape[1]] = x4_direction_t_inpaint
                x4_direction_t = x4_direction_t.reshape(-1, 3)
            else:
                x4_direction_t = x4_direction_t_inpaint.reshape(-1,3)
        if inpaint_x4_type:
            x4_x_t_inpaint = x4_x_inpainting_trajectory[x4_t].repeat(batch_size, 1, 1)
            if do_partial_pharm_inpainting:
                x4_x_t = x4_x_t.reshape(batch_size, -1, num_pharm_types)
                x4_x_t[:, :x4_x_t_inpaint.shape[1]] = x4_x_t_inpaint
                x4_x_t = x4_x_t.reshape(-1, num_pharm_types)
            else:
                x4_x_t = x4_x_t_inpaint.reshape(-1, num_pharm_types)


    ######## Main Denoising Loop #########

    # packing inpainting dict
    inpainting_dict = None
    if not unconditional:
        inpainting_dict = {
            'inpaint_x1_pos': inpaint_x1_pos,
            'inpaint_x1_x': inpaint_x1_x,
            'inpaint_x2_pos': inpaint_x2_pos,
            'inpaint_x3_pos': inpaint_x3_pos,
            'inpaint_x3_x': inpaint_x3_x,
            'inpaint_x4_pos': inpaint_x4_pos,
            'inpaint_x4_direction': inpaint_x4_direction,
            'inpaint_x4_type': inpaint_x4_type,
            'x1_pos_inpainting_trajectory': x1_pos_inpainting_trajectory,
            'x1_x_inpainting_trajectory': x1_x_inpainting_trajectory,
            'x2_pos_inpainting_trajectory': x2_pos_inpainting_trajectory,
            'x3_pos_inpainting_trajectory': x3_pos_inpainting_trajectory,
            'x3_x_inpainting_trajectory': x3_x_inpainting_trajectory,
            'x4_pos_inpainting_trajectory': x4_pos_inpainting_trajectory,
            'x4_direction_inpainting_trajectory': x4_direction_inpainting_trajectory,
            'x4_x_inpainting_trajectory': x4_x_inpainting_trajectory,
            'stop_inpainting_at_time_x1': stop_inpainting_at_time_x1,
            'stop_inpainting_at_time_x2': stop_inpainting_at_time_x2,
            'stop_inpainting_at_time_x3': stop_inpainting_at_time_x3,
            'stop_inpainting_at_time_x4': stop_inpainting_at_time_x4,
            'add_noise_to_inpainted_x2_pos': add_noise_to_inpainted_x2_pos,
            'add_noise_to_inpainted_x3_pos': add_noise_to_inpainted_x3_pos,
            'add_noise_to_inpainted_x3_x': add_noise_to_inpainted_x3_x,
            'add_noise_to_inpainted_x4_pos': add_noise_to_inpainted_x4_pos,
            'add_noise_to_inpainted_x4_direction': add_noise_to_inpainted_x4_direction,
            'add_noise_to_inpainted_x4_type': add_noise_to_inpainted_x4_type,
            'do_partial_pharm_inpainting': do_partial_pharm_inpainting,
            'do_partial_atom_inpainting': do_partial_atom_inpainting,
        }

    inference_step = partial(
        _inference_step,
        model_pl=model_pl,
        params=params,
        time_steps=time_steps,
        harmonize=harmonize, harmonize_ts=harmonize_ts, harmonize_jumps=harmonize_jumps,
        batch_size=batch_size,
        denoising_noise_scale=denoising_noise_scale,
        inject_noise_at_ts=inject_noise_at_ts, inject_noise_scales=inject_noise_scales,
        virtual_node_mask_x1=virtual_node_mask_x1,
        virtual_node_mask_x2=virtual_node_mask_x2,
        virtual_node_mask_x3=virtual_node_mask_x3,
        virtual_node_mask_x4=virtual_node_mask_x4,
        inpainting_dict=inpainting_dict,
    )
    if verbose:
        pbar = tqdm(total=len(time_steps) -1 + sum(harmonize_jumps) * int(harmonize), position=0, leave=True, miniters=50, maxinterval=1000)
    else:
        pbar = None

    current_time_idx = 0
    while current_time_idx < len(time_steps) - 1:
        current_time_idx, next_state = inference_step(
            current_time_idx=current_time_idx,
            x1_pos_t=x1_pos_t, x1_x_t=x1_x_t, x1_bond_edge_x_t=x1_bond_edge_x_t, x1_batch=x1_batch,
            bond_edge_index_x1=bond_edge_index_x1,
            x2_pos_t=x2_pos_t, x2_x_t=x2_x_t, x2_batch=x2_batch,
            x3_pos_t=x3_pos_t, x3_x_t=x3_x_t, x3_batch=x3_batch,
            x4_pos_t=x4_pos_t, x4_direction_t=x4_direction_t, x4_x_t=x4_x_t, x4_batch=x4_batch,
            pbar=pbar,
            include_x0_pred=False,
        )

        # update states: x_{t-1} -> x_{t}
        x1_pos_t = next_state['x1_pos_t_1']
        x1_x_t = next_state['x1_x_t_1']
        x1_bond_edge_x_t = next_state['x1_bond_edge_x_t_1']
        x2_pos_t = next_state['x2_pos_t_1']
        x2_x_t = next_state['x2_x_t_1']
        x3_pos_t = next_state['x3_pos_t_1']
        x3_x_t = next_state['x3_x_t_1']
        x4_pos_t = next_state['x4_pos_t_1']
        x4_direction_t = next_state['x4_direction_t_1']
        x4_x_t = next_state['x4_x_t_1']

        # TODO store trajectories

    if pbar is not None:
        pbar.close()
    
    del next_state

    generated_structures = _extract_generated_samples(
        x1_x_t, x1_pos_t, x1_bond_edge_x_t, virtual_node_mask_x1,
        x2_pos_t, virtual_node_mask_x2,
        x3_pos_t, x3_x_t, virtual_node_mask_x3,
        x4_pos_t, x4_direction_t, x4_x_t, virtual_node_mask_x4,
        params, batch_size)

    return generated_structures
