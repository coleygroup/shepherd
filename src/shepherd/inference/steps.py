"""
Contains the process to perform one step of reverse denoising.
"""
from typing import Optional
from copy import deepcopy
from tqdm import tqdm
import torch
import numpy as np
import torch_scatter

from shepherd.inference.noise import (
    _get_noise_params_for_timestep,
    forward_jump
)

# helper function to perform one step of reverse denoising
def _perform_reverse_denoising_step(
    current_t: int, # current timestep
    batch_size: int,
    noise_params_current: dict, # dict of params for current_t
    # current states (x_t)
    x1_pos_t: torch.Tensor, x1_x_t: torch.Tensor, x1_bond_edge_x_t: torch.Tensor, x1_batch: torch.Tensor, virtual_node_mask_x1: torch.Tensor,
    x2_pos_t: torch.Tensor, x2_x_t: torch.Tensor, x2_batch: torch.Tensor, virtual_node_mask_x2: torch.Tensor,
    x3_pos_t: torch.Tensor, x3_x_t: torch.Tensor, x3_batch: torch.Tensor, virtual_node_mask_x3: torch.Tensor,
    x4_pos_t: torch.Tensor, x4_direction_t: torch.Tensor, x4_x_t: torch.Tensor, x4_batch: torch.Tensor, virtual_node_mask_x4: torch.Tensor,
    # model outputs (predicted noise eps_theta(x_t, t))
    x1_pos_out: torch.Tensor, x1_x_out: torch.Tensor, x1_bond_edge_x_out: torch.Tensor,
    x2_pos_out: torch.Tensor,
    x3_pos_out: torch.Tensor, x3_x_out: torch.Tensor,
    x4_pos_out: torch.Tensor, x4_direction_out: torch.Tensor, x4_x_out: torch.Tensor,
    denoising_noise_scale: float, inject_noise_at_ts: list, inject_noise_scales: list):
    """
    Perform one step of reverse denoising.

    Arguments
    ---------
    current_t: The current timestep.
    batch_size: The batch size.
    noise_params_current: The noise parameters for the current timestep.

    x1_pos_t: The x1 position tensor.
    x1_x_t: The x1 feature tensor.
    x1_bond_edge_x_t: The x1 bond edge feature tensor.
    x1_batch: The x1 batch tensor.
    virtual_node_mask_x1: The x1 virtual node mask tensor.

    x2_pos_t: The x2 position tensor.
    x2_x_t: The x2 feature tensor.
    x2_batch: The x2 batch tensor.
    virtual_node_mask_x2: The x2 virtual node mask tensor.

    x3_pos_t: The x3 position tensor.
    x3_x_t: The x3 feature tensor.
    x3_batch: The x3 batch tensor.
    virtual_node_mask_x3: The x3 virtual node mask tensor.

    x4_pos_t: The x4 position tensor.
    x4_direction_t: The x4 direction tensor.
    x4_x_t: The x4 feature tensor.
    x4_batch: The x4 batch tensor.
    virtual_node_mask_x4: The x4 virtual node mask tensor.

    denoising_noise_scale: The denoising noise scale.
    inject_noise_at_ts: The noise injection time steps.
    inject_noise_scales: The noise injection scales.

    Returns
    -------
    dict: The next state.
        x1_pos_t_1: The x1 position tensor for the next step.
        x1_x_t_1: The x1 feature tensor for the next step.
        x1_bond_edge_x_t_1: The x1 bond edge feature tensor for the next step.
        x2_pos_t_1: The x2 position tensor for the next step.
        x2_x_t_1: The x2 feature tensor for the next step.
        x3_pos_t_1: The x3 position tensor for the next step.
        x3_x_t_1: The x3 feature tensor for the next step.
        x4_pos_t_1: The x4 position tensor for the next step.
        x4_direction_t_1: The x4 direction tensor for the next step.
        x4_x_t_1: The x4 feature tensor for the next step.
    """
    # extract parameters for convenience
    # current time
    x1_alpha_dash_t = noise_params_current['x1']['alpha_dash_t']
    x1_var_dash_t = noise_params_current['x1']['var_dash_t']
    x2_alpha_dash_t = noise_params_current['x2']['alpha_dash_t']
    x2_var_dash_t = noise_params_current['x2']['var_dash_t']
    x3_alpha_dash_t = noise_params_current['x3']['alpha_dash_t']
    x3_var_dash_t = noise_params_current['x3']['var_dash_t']
    x4_alpha_dash_t = noise_params_current['x4']['alpha_dash_t']
    x4_var_dash_t = noise_params_current['x4']['var_dash_t']
        
    # get batch sizes and feature dims
    num_atom_types = x1_x_t.shape[-1]
    num_pharm_types = x4_x_t.shape[-1]
    x1_batch_size_nodes = x1_pos_t.shape[0]
    x2_batch_size_nodes = x2_pos_t.shape[0]
    x3_batch_size_nodes = x3_pos_t.shape[0]
    x4_batch_size_nodes = x4_pos_t.shape[0]

    # get added noise - x1
    x1_pos_epsilon = torch.randn(x1_batch_size_nodes, 3)
    x1_x_epsilon = torch.randn(x1_batch_size_nodes, num_atom_types)
    x1_bond_edge_x_epsilon = torch.randn_like(x1_bond_edge_x_out)
    x1_pos_epsilon = x1_pos_epsilon - torch_scatter.scatter_mean(x1_pos_epsilon[~virtual_node_mask_x1], x1_batch[~virtual_node_mask_x1], dim=0)[x1_batch]
    x1_pos_epsilon[virtual_node_mask_x1, :] = 0.0
    x1_x_epsilon[virtual_node_mask_x1, :] = 0.0
    x1_c_t = (noise_params_current['x1']['sigma_t'] * noise_params_current['x1']['sigma_dash_t_1']) / (noise_params_current['x1']['sigma_dash_t'] + 1e-9) if noise_params_current['x1']['t_idx'] > 0 else 0
    x1_c_t = x1_c_t * denoising_noise_scale

    # get added noise - x2
    x2_pos_epsilon = torch.randn(x2_batch_size_nodes, 3)
    x2_pos_epsilon[virtual_node_mask_x2, :] = 0.0
    x2_c_t = (noise_params_current['x2']['sigma_t'] * noise_params_current['x2']['sigma_dash_t_1']) / (noise_params_current['x2']['sigma_dash_t'] + 1e-9) if noise_params_current['x2']['t_idx'] > 0 else 0
    x2_c_t = x2_c_t * denoising_noise_scale

    # get added noise - x3
    x3_pos_epsilon = torch.randn(x3_batch_size_nodes, 3)
    x3_x_epsilon = torch.randn(x3_batch_size_nodes)
    x3_pos_epsilon[virtual_node_mask_x3, :] = 0.0
    x3_x_epsilon[virtual_node_mask_x3, ...] = 0.0
    x3_c_t = (noise_params_current['x3']['sigma_t'] * noise_params_current['x3']['sigma_dash_t_1']) / (noise_params_current['x3']['sigma_dash_t'] + 1e-9) if noise_params_current['x3']['t_idx'] > 0 else 0
    x3_c_t = x3_c_t * denoising_noise_scale

    # get added noise - x4
    x4_pos_epsilon = torch.randn(x4_batch_size_nodes, 3)
    x4_direction_epsilon = torch.randn(x4_batch_size_nodes, 3)
    x4_x_epsilon = torch.randn(x4_batch_size_nodes, num_pharm_types)
    x4_pos_epsilon[virtual_node_mask_x4, :] = 0.0
    x4_direction_epsilon[virtual_node_mask_x4, :] = 0.0
    x4_x_epsilon[virtual_node_mask_x4, ...] = 0.0
    x4_c_t = (noise_params_current['x4']['sigma_t'] * noise_params_current['x4']['sigma_dash_t_1']) / (noise_params_current['x4']['sigma_dash_t'] + 1e-9) if noise_params_current['x4']['t_idx'] > 0 else 0
    x4_c_t = x4_c_t * denoising_noise_scale

    # --- Conditional Sampler Logic --- 
    # fetch necessary single-step and cumulative params for current_t
    x1_t_idx = noise_params_current['x1']['t_idx']
    x1_alpha_t = noise_params_current['x1']['alpha_t'] # single step alpha
    x1_sigma_t = noise_params_current['x1']['sigma_t']
    x1_sigma_dash_t = noise_params_current['x1']['sigma_dash_t']
    x1_sigma_dash_t_1 = noise_params_current['x1']['sigma_dash_t_1'] # from prev step in full schedule
    # Note: DDPM formula uses single-step alpha_t and cumulative vars/sigmas
    # var_dash_t is needed from current params
    x1_var_dash_t = noise_params_current['x1']['var_dash_t'] 
    
    x2_t_idx = noise_params_current['x2']['t_idx']
    x2_alpha_t = noise_params_current['x2']['alpha_t']
    x2_sigma_t = noise_params_current['x2']['sigma_t']
    x2_sigma_dash_t = noise_params_current['x2']['sigma_dash_t']
    x2_sigma_dash_t_1 = noise_params_current['x2']['sigma_dash_t_1']
    x2_var_dash_t = noise_params_current['x2']['var_dash_t']

    x3_t_idx = noise_params_current['x3']['t_idx']
    x3_alpha_t = noise_params_current['x3']['alpha_t']
    x3_sigma_t = noise_params_current['x3']['sigma_t']
    x3_sigma_dash_t = noise_params_current['x3']['sigma_dash_t']
    x3_sigma_dash_t_1 = noise_params_current['x3']['sigma_dash_t_1']
    x3_var_dash_t = noise_params_current['x3']['var_dash_t']

    x4_t_idx = noise_params_current['x4']['t_idx']
    x4_alpha_t = noise_params_current['x4']['alpha_t']
    x4_sigma_t = noise_params_current['x4']['sigma_t']
    x4_sigma_dash_t = noise_params_current['x4']['sigma_dash_t']
    x4_sigma_dash_t_1 = noise_params_current['x4']['sigma_dash_t_1']
    x4_var_dash_t = noise_params_current['x4']['var_dash_t']

    # calculate noise scale factor 'c_t' using current schedule params
    x1_c_t = (x1_sigma_t * x1_sigma_dash_t_1) / (x1_sigma_dash_t + 1e-9) if x1_t_idx > 0 else 0
    x1_c_t = x1_c_t * denoising_noise_scale
    x2_c_t = (x2_sigma_t * x2_sigma_dash_t_1) / (x2_sigma_dash_t + 1e-9) if x2_t_idx > 0 else 0
    x2_c_t = x2_c_t * denoising_noise_scale
    x3_c_t = (x3_sigma_t * x3_sigma_dash_t_1) / (x3_sigma_dash_t + 1e-9) if x3_t_idx > 0 else 0
    x3_c_t = x3_c_t * denoising_noise_scale
    x4_c_t = (x4_sigma_t * x4_sigma_dash_t_1) / (x4_sigma_dash_t + 1e-9) if x4_t_idx > 0 else 0
    x4_c_t = x4_c_t * denoising_noise_scale

    # apply noise injection logic using current_t
    x1_c_t_injected = x1_c_t
    x2_c_t_injected = x2_c_t
    x3_c_t_injected = x3_c_t
    x4_c_t_injected = x4_c_t
    # use copies to avoid modifying lists if function is somehow called in a loop over t
    current_inject_ts = list(inject_noise_at_ts)
    current_inject_scales = list(inject_noise_scales)
    if current_t in current_inject_ts:
        idx_to_pop = current_inject_ts.index(current_t)
        inject_noise_scale = current_inject_scales[idx_to_pop]
        # Don't pop from original lists 
        x1_c_t_injected = x1_c_t + inject_noise_scale
        x2_c_t_injected = x2_c_t + inject_noise_scale
        x3_c_t_injected = x3_c_t + inject_noise_scale
        x4_c_t_injected = x4_c_t + inject_noise_scale

    # DDPM update
    # xt-1 = (xt - (1-alpha_t)/sqrt(1-alpha_dash_t) * eps_theta) / sqrt(alpha_t) + sigma_t * eps
    x1_pos_t_1 = ((1. / x1_alpha_t) * x1_pos_t) - ((x1_var_dash_t / (x1_alpha_t * x1_sigma_dash_t + 1e-9)) * x1_pos_out) + (x1_c_t_injected * x1_pos_epsilon)
    x1_x_t_1 = ((1. / x1_alpha_t) * x1_x_t) - ((x1_var_dash_t / (x1_alpha_t * x1_sigma_dash_t + 1e-9)) * x1_x_out) + (x1_c_t * x1_x_epsilon)
    x1_bond_edge_x_t_1 = ((1. / x1_alpha_t) * x1_bond_edge_x_t) - ((x1_var_dash_t / (x1_alpha_t * x1_sigma_dash_t + 1e-9)) * x1_bond_edge_x_out) + (x1_c_t * x1_bond_edge_x_epsilon)

    x2_pos_t_1 = ((1. / float(x2_alpha_t)) * x2_pos_t) - ((x2_var_dash_t / (x2_alpha_t * x2_sigma_dash_t + 1e-9)) * x2_pos_out) + (x2_c_t_injected * x2_pos_epsilon)
    x2_x_t_1 = x2_x_t # Not diffused

    x3_pos_t_1 = ((1. / float(x3_alpha_t)) * x3_pos_t) - ((x3_var_dash_t / (x3_alpha_t * x3_sigma_dash_t + 1e-9)) * x3_pos_out) + (x3_c_t_injected * x3_pos_epsilon)
    x3_x_t_1 = ((1. / x3_alpha_t) * x3_x_t) - ((x3_var_dash_t / (x3_alpha_t * x3_sigma_dash_t + 1e-9)) * x3_x_out) + (x3_c_t * x3_x_epsilon)

    x4_pos_t_1 = ((1. / float(x4_alpha_t)) * x4_pos_t) - ((x4_var_dash_t / (x4_alpha_t * x4_sigma_dash_t + 1e-9)) * x4_pos_out) + (x4_c_t_injected * x4_pos_epsilon)
    x4_direction_t_1 = ((1. / float(x4_alpha_t)) * x4_direction_t) - ((x4_var_dash_t / (x4_alpha_t * x4_sigma_dash_t + 1e-9)) * x4_direction_out) + (x4_c_t * x4_direction_epsilon)
    x4_x_t_1 = ((1. / x4_alpha_t) * x4_x_t) - ((x4_var_dash_t / (x4_alpha_t * x4_sigma_dash_t + 1e-9)) * x4_x_out) + (x4_c_t * x4_x_epsilon)

    # reset virtual nodes (common to both paths)
    x1_pos_t_1[virtual_node_mask_x1] = x1_pos_t[virtual_node_mask_x1]
    x1_x_t_1[virtual_node_mask_x1] = x1_x_t[virtual_node_mask_x1]
    x2_pos_t_1[virtual_node_mask_x2] = x2_pos_t[virtual_node_mask_x2]
    x2_x_t_1[virtual_node_mask_x2] = x2_x_t[virtual_node_mask_x2]
    x3_pos_t_1[virtual_node_mask_x3] = x3_pos_t[virtual_node_mask_x3]
    x3_x_t_1[virtual_node_mask_x3] = x3_x_t[virtual_node_mask_x3]
    x4_pos_t_1[virtual_node_mask_x4] = x4_pos_t[virtual_node_mask_x4]
    x4_direction_t_1[virtual_node_mask_x4] = x4_direction_t[virtual_node_mask_x4]
    x4_x_t_1[virtual_node_mask_x4] = x4_x_t[virtual_node_mask_x4]

    return {
        'x1_pos_t_1': x1_pos_t_1, 'x1_x_t_1': x1_x_t_1, 'x1_bond_edge_x_t_1': x1_bond_edge_x_t_1,
        'x2_pos_t_1': x2_pos_t_1, 'x2_x_t_1': x2_x_t_1, 
        'x3_pos_t_1': x3_pos_t_1, 'x3_x_t_1': x3_x_t_1,
        'x4_pos_t_1': x4_pos_t_1, 'x4_direction_t_1': x4_direction_t_1, 'x4_x_t_1': x4_x_t_1,
    }


def _prepare_model_input(device, dtype, batch_size, t, 
                         x1_pos_t, x1_x_t, x1_batch, x1_bond_edge_x_t, x1_bond_edge_index, x1_virtual_node_mask, x1_params, 
                         x2_pos_t, x2_x_t, x2_batch, x2_virtual_node_mask, x2_params, 
                         x3_pos_t, x3_x_t, x3_batch, x3_virtual_node_mask, x3_params, 
                         x4_pos_t, x4_direction_t, x4_x_t, x4_batch, x4_virtual_node_mask, x4_params
                         ):
    """
    Prepare the model input dictionary.

    Arguments
    ---------
    device: The device.
    dtype: The data type.
    batch_size: The batch size.
    t: The current timestep.

    all x_t: tensors
    all x_params: dict

    Returns
    -------
    dict: The model input dictionary.
        device: torch.device
        dtype: torch.dtype
        x1-x4: dict
            decoder: dict
                various features: torch.Tensor
                batch: torch.Tensor
                timestep: torch.Tensor
                sigma_dash_t: torch.Tensor
                alpha_dash_t: torch.Tensor
                virtual_node_mask: torch.Tensor
    """
    x1_timestep = torch.tensor([t] * batch_size)
    x2_timestep = torch.tensor([t] * batch_size)
    x3_timestep = torch.tensor([t] * batch_size)
    x4_timestep = torch.tensor([t] * batch_size)

    x1_sigma_dash_t_ = torch.tensor([x1_params['sigma_dash_t']] * batch_size, dtype=dtype)
    x1_alpha_dash_t_ = torch.tensor([x1_params['alpha_dash_t']] * batch_size, dtype=dtype)
    
    x2_sigma_dash_t_ = torch.tensor([x2_params['sigma_dash_t']] * batch_size, dtype=dtype)
    x2_alpha_dash_t_ = torch.tensor([x2_params['alpha_dash_t']] * batch_size, dtype=dtype)
    
    x3_sigma_dash_t_ = torch.tensor([x3_params['sigma_dash_t']] * batch_size, dtype=dtype)
    x3_alpha_dash_t_ = torch.tensor([x3_params['alpha_dash_t']] * batch_size, dtype=dtype)
    
    x4_sigma_dash_t_ = torch.tensor([x4_params['sigma_dash_t']] * batch_size, dtype=dtype)
    x4_alpha_dash_t_ = torch.tensor([x4_params['alpha_dash_t']] * batch_size, dtype=dtype)

    input_dict = {
        'device': device,
        'dtype': dtype,
        'x1': {
            'decoder': {
                'pos': x1_pos_t.to(device),
                'x': x1_x_t.to(device),
                'batch': x1_batch.to(device),
                'bond_edge_x': x1_bond_edge_x_t.to(device),
                'bond_edge_index': x1_bond_edge_index.to(device),
                'timestep': x1_timestep.to(device),
                'sigma_dash_t': x1_sigma_dash_t_.to(device),
                'alpha_dash_t': x1_alpha_dash_t_.to(device),
                'virtual_node_mask': x1_virtual_node_mask.to(device),
            },
        },
        'x2': {
            'decoder': {
                'pos': x2_pos_t.to(device),
                'x': x2_x_t.to(device),
                'batch': x2_batch.to(device),
                'timestep': x2_timestep.to(device),
                'sigma_dash_t': x2_sigma_dash_t_.to(device),
                'alpha_dash_t': x2_alpha_dash_t_.to(device),
                'virtual_node_mask': x2_virtual_node_mask.to(device),
            },
        },
        'x3': {
            'decoder': {
                'pos': x3_pos_t.to(device),
                'x': x3_x_t.to(device),
                'batch': x3_batch.to(device),
                'timestep': x3_timestep.to(device),
                'sigma_dash_t': x3_sigma_dash_t_.to(device),
                'alpha_dash_t': x3_alpha_dash_t_.to(device),
                'virtual_node_mask': x3_virtual_node_mask.to(device),
            },
        },
        'x4': {
            'decoder': {
                'x': x4_x_t.to(device),
                'pos': x4_pos_t.to(device),
                'direction': x4_direction_t.to(device),
                'batch': x4_batch.to(device),
                'timestep': x4_timestep.to(device),
                'sigma_dash_t': x4_sigma_dash_t_.to(device),
                'alpha_dash_t': x4_alpha_dash_t_.to(device),
                'virtual_node_mask': x4_virtual_node_mask.to(device),
            },
        },
    }

    return input_dict


def _inference_step(
    model_pl, params,
    # times
    time_steps, current_time_idx,
    harmonize, harmonize_ts, harmonize_jumps,
    batch_size,
    denoising_noise_scale, inject_noise_at_ts, inject_noise_scales,
    # current states
    x1_pos_t, x1_x_t, x1_bond_edge_x_t, x1_batch, bond_edge_index_x1, virtual_node_mask_x1,
    x2_pos_t, x2_x_t, x2_batch, virtual_node_mask_x2,
    x3_pos_t, x3_x_t, x3_batch, virtual_node_mask_x3,
    x4_pos_t, x4_direction_t, x4_x_t, x4_batch, virtual_node_mask_x4,
    # inpainting
    inpainting_dict: Optional[dict] = None,
    # progress bar
    pbar: Optional[tqdm] = None,
    include_x0_pred=False,
    ):
    """
    Inner loop for the denoising process.
    This function is called by the main denoising function and handles the
    denoising steps for each time step in the sequence.

    Arguments
    ---------
    model_pl: The model.
    params: The parameters.
    time_steps: The time steps.
    current_time_idx: The current time index.
    harmonize: Whether to harmonize.
    harmonize_ts: The harmonize time steps.
    harmonize_jumps: The harmonize jumps.
    batch_size: The batch size.
    denoising_noise_scale: The denoising noise scale.
    inject_noise_at_ts: The noise injection time steps.
    inject_noise_scales: The noise injection scales.
    x1_pos_t: The x1 position tensor.
    x1_x_t: The x1 feature tensor.
    x1_bond_edge_x_t: The x1 bond edge feature tensor.
    """
    current_t = time_steps[current_time_idx]
    prev_t = time_steps[current_time_idx + 1] # The time we are calculating state FOR
    
    # t passed to helpers/model should be current time
    t = current_t 

    # inputs (these might be redundant now, t is the main driver)
    x1_t = t
    x2_t = t
    x3_t = t
    x4_t = t

    stop_inpainting_at_time_x1 = 0.0
    stop_inpainting_at_time_x2 = 0.0
    stop_inpainting_at_time_x3 = 0.0
    stop_inpainting_at_time_x4 = 0.0

    if inpainting_dict is None: # unconditional
        inpaint_x1_pos = False
        inpaint_x1_x = False
        inpaint_x2_pos = False
        inpaint_x3_pos = False
        inpaint_x3_x = False
        inpaint_x4_pos = False
        inpaint_x4_direction = False
        inpaint_x4_type = False

    if inpainting_dict is not None: # conditional
        inpaint_x1_pos = inpainting_dict['inpaint_x1_pos']
        inpaint_x1_x = inpainting_dict['inpaint_x1_x']
        inpaint_x2_pos = inpainting_dict['inpaint_x2_pos']
        inpaint_x3_pos = inpainting_dict['inpaint_x3_pos']
        inpaint_x3_x = inpainting_dict['inpaint_x3_x']
        inpaint_x4_pos = inpainting_dict['inpaint_x4_pos']
        inpaint_x4_direction = inpainting_dict['inpaint_x4_direction']
        inpaint_x4_type = inpainting_dict['inpaint_x4_type']

        if inpaint_x1_pos:
            x1_pos_inpainting_trajectory = inpainting_dict['x1_pos_inpainting_trajectory']
            stop_inpainting_at_time_x1 = inpainting_dict['stop_inpainting_at_time_x1']

        if inpaint_x1_x:
            x1_x_inpainting_trajectory = inpainting_dict['x1_x_inpainting_trajectory']
            stop_inpainting_at_time_x1 = inpainting_dict['stop_inpainting_at_time_x1']

        if inpaint_x2_pos:
            x2_pos_inpainting_trajectory = inpainting_dict['x2_pos_inpainting_trajectory']
            stop_inpainting_at_time_x2 = inpainting_dict['stop_inpainting_at_time_x2']
            add_noise_to_inpainted_x2_pos = inpainting_dict['add_noise_to_inpainted_x2_pos']
        else:
            stop_inpainting_at_time_x2 = 0.0
        if inpaint_x3_pos:
            x3_pos_inpainting_trajectory = inpainting_dict['x3_pos_inpainting_trajectory']
        if inpaint_x3_x:
            x3_x_inpainting_trajectory = inpainting_dict['x3_x_inpainting_trajectory']
        if inpaint_x3_pos or inpaint_x3_x:
            stop_inpainting_at_time_x3 = inpainting_dict['stop_inpainting_at_time_x3']
            add_noise_to_inpainted_x3_pos = inpainting_dict['add_noise_to_inpainted_x3_pos']
            add_noise_to_inpainted_x3_x = inpainting_dict['add_noise_to_inpainted_x3_x']
        else:
            stop_inpainting_at_time_x3 = 0.0
        if inpaint_x4_pos:
            x4_pos_inpainting_trajectory = inpainting_dict['x4_pos_inpainting_trajectory']
        if inpaint_x4_direction:
            x4_direction_inpainting_trajectory = inpainting_dict['x4_direction_inpainting_trajectory']
        if inpaint_x4_type:
            x4_x_inpainting_trajectory = inpainting_dict['x4_x_inpainting_trajectory']
        if inpaint_x4_pos or inpaint_x4_direction or inpaint_x4_type:
            stop_inpainting_at_time_x4 = inpainting_dict['stop_inpainting_at_time_x4']
            add_noise_to_inpainted_x4_pos = inpainting_dict['add_noise_to_inpainted_x4_pos']
            add_noise_to_inpainted_x4_direction = inpainting_dict['add_noise_to_inpainted_x4_direction']
            add_noise_to_inpainted_x4_type = inpainting_dict['add_noise_to_inpainted_x4_type']
        else:
            stop_inpainting_at_time_x4 = 0.0
        do_partial_pharm_inpainting = inpainting_dict['do_partial_pharm_inpainting']
        do_partial_atom_inpainting = inpainting_dict['do_partial_atom_inpainting']
    
    # harmonize
    # harmonization needs careful consideration with subsequenced timesteps
    # for now, we only harmonize if t is exactly in harmonize_ts
    # a jump might skip over a harmonize_ts value in DDIM
    # if harmonization is used with DDIM, ensure harmonize_ts align with time_steps
    perform_harmonization_jump = False
    harmonize_jump_len = 0
    if (harmonize) and (len(harmonize_ts) > 0) and (t == harmonize_ts[0]):
        if pbar is not None:
            print(f'Harmonizing... at time {t}')
        harmonize_ts.pop(0)
        if len(harmonize_ts) == 0:
            harmonize = False # use up harmonization steps
        harmonize_jump_len = harmonize_jumps.pop(0)
        perform_harmonization_jump = True
        
    if perform_harmonization_jump:
        x1_sigma_ts = params['noise_schedules']['x1']['sigma_ts']
        x2_sigma_ts = params['noise_schedules']['x2']['sigma_ts']
        x3_sigma_ts = params['noise_schedules']['x3']['sigma_ts']
        x4_sigma_ts = params['noise_schedules']['x4']['sigma_ts']
        
        x1_pos_t, x1_t_jump = forward_jump(x1_pos_t, x1_t, harmonize_jump_len, x1_sigma_ts, remove_COM_from_noise = True, batch = x1_batch, mask = ~virtual_node_mask_x1)
        x1_x_t, x1_t_jump = forward_jump(x1_x_t, x1_t, harmonize_jump_len, x1_sigma_ts, remove_COM_from_noise = False, batch = x1_batch, mask = ~virtual_node_mask_x1)
        x1_bond_edge_x_t, x1_t_jump = forward_jump(x1_bond_edge_x_t, x1_t, harmonize_jump_len, x1_sigma_ts, remove_COM_from_noise = False, batch = None, mask = None)
        
        x2_pos_t, x2_t_jump = forward_jump(x2_pos_t, x2_t, harmonize_jump_len, x2_sigma_ts, remove_COM_from_noise = False, batch = x2_batch, mask = ~virtual_node_mask_x2)
        
        x3_pos_t, x3_t_jump = forward_jump(x3_pos_t, x3_t, harmonize_jump_len, x3_sigma_ts, remove_COM_from_noise = False, batch = x3_batch, mask = ~virtual_node_mask_x3)
        x3_x_t, x3_t_jump = forward_jump(x3_x_t, x3_t, harmonize_jump_len, x3_sigma_ts, remove_COM_from_noise = False, batch = x3_batch, mask = ~virtual_node_mask_x3)
        
        x4_pos_t, x4_t_jump = forward_jump(x4_pos_t, x4_t, harmonize_jump_len, x4_sigma_ts, remove_COM_from_noise = False, batch = x4_batch, mask = ~virtual_node_mask_x4)
        x4_direction_t, x4_t_jump = forward_jump(x4_direction_t, x4_t, harmonize_jump_len, x4_sigma_ts, remove_COM_from_noise = False, batch = x4_batch, mask = ~virtual_node_mask_x4)
        x4_x_t, x4_t_jump = forward_jump(x4_x_t, x4_t, harmonize_jump_len, x4_sigma_ts, remove_COM_from_noise = False, batch = x4_batch, mask = ~virtual_node_mask_x4)

        # after jumping forward, we need to find the corresponding index in our time_steps
        # this simple implementation assumes the jump lands exactly on a future step in the sequence
        # more robust: find the closest step in the sequence
        jumped_to_t = x1_t_jump # assuming all jumps are same length
        try:
            # find where the jumped-to time occurs in the sequence
            jump_to_idx = np.where(time_steps == jumped_to_t)[0][0]
            # reset the loop index to continue from the jumped-to time
            current_time_idx = jump_to_idx
            if pbar is not None:
                pbar.update(harmonize_jump_len) # update progress bar for the jumped steps
                print(f"Harmonization jumped from t={t} to t={jumped_to_t}, resuming loop.")
            t = jumped_to_t # update t for the next iteration start
            # need to re-fetch noise params for the new 't' before proceeding if the loop continued immediately,
            # but we will recalculate at the start of the next iteration anyway
            next_state = {
                'x1_pos_t_1': x1_pos_t, 'x1_x_t_1': x1_x_t, 'x1_bond_edge_x_t_1': x1_bond_edge_x_t,
                'x2_pos_t_1': x2_pos_t, 'x2_x_t_1': x2_x_t, 
                'x3_pos_t_1': x3_pos_t, 'x3_x_t_1': x3_x_t,
                'x4_pos_t_1': x4_pos_t, 'x4_direction_t_1': x4_direction_t, 'x4_x_t_1': x4_x_t,
            }
            return current_time_idx, next_state # skip the rest of the current loop iteration (denoising step)
        except IndexError:
            print(f"Warning: Harmonization jumped from t={t} to t={jumped_to_t}, which is not in the planned time_steps sequence {time_steps}. Stopping Harmonization.")
            harmonize = False # disable future harmonization if jump is incompatible
            # continue the loop from the *next* scheduled step after the original t

    # inpainting logic
    ## HAVE TO RECENTER SOMEWHERE
    if (x1_t > stop_inpainting_at_time_x1):
        num_atom_types = len(params['dataset']['x1']['atom_types']) + len(params['dataset']['x1']['charge_types'])
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
            x4_pos_t_inpaint = torch.cat([x4_pos_inpainting_trajectory[x4_t] for _ in range(batch_size)], dim = 0)
            noise = torch.randn_like(x4_pos_t)
            noise[virtual_node_mask_x4] = 0.0
            if do_partial_pharm_inpainting:
                x4_pos_t_inpaint = x4_pos_t_inpaint.reshape(batch_size, -1, 3)
                noise = noise.reshape(batch_size, -1, 3)[:, :x4_pos_t_inpaint.shape[1]]

                x4_pos_t_inpaint = x4_pos_t_inpaint + add_noise_to_inpainted_x4_pos * noise
                x4_pos_t = x4_pos_t.reshape(batch_size, -1, 3)
                x4_pos_t[:, :x4_pos_t_inpaint.shape[1]] = x4_pos_t_inpaint
                x4_pos_t = x4_pos_t.reshape(-1, 3)
            else:
                x4_pos_t_inpaint = x4_pos_t_inpaint + add_noise_to_inpainted_x4_pos * noise
                x4_pos_t = x4_pos_t_inpaint

        if inpaint_x4_direction:
            x4_direction_t_inpaint = torch.cat([x4_direction_inpainting_trajectory[x4_t] for _ in range(batch_size)], dim = 0)
            noise = torch.randn_like(x4_direction_t)
            noise[virtual_node_mask_x4] = 0.0
            if do_partial_pharm_inpainting:
                x4_direction_t_inpaint = x4_direction_t_inpaint.reshape(batch_size, -1, 3)
                noise = noise.reshape(batch_size, -1, 3)[:, :x4_direction_t_inpaint.shape[1]]

                x4_direction_t_inpaint = x4_direction_t_inpaint + add_noise_to_inpainted_x4_direction * noise
                x4_direction_t = x4_direction_t.reshape(batch_size, -1, 3)
                x4_direction_t[:, :x4_direction_t_inpaint.shape[1]] = x4_direction_t_inpaint
                x4_direction_t = x4_direction_t.reshape(-1, 3)
            else:
                x4_direction_t_inpaint = x4_direction_t_inpaint + add_noise_to_inpainted_x4_direction * noise
                x4_direction_t = x4_direction_t_inpaint
        if inpaint_x4_type:
            num_pharm_types = params['dataset']['x4']['max_node_types']
            x4_x_t_inpaint = torch.cat([x4_x_inpainting_trajectory[x4_t] for _ in range(batch_size)], dim = 0)
            noise = torch.randn_like(x4_x_t)
            noise[virtual_node_mask_x4] = 0.0
            if do_partial_pharm_inpainting:
                x4_x_t_inpaint = x4_x_t_inpaint.reshape(batch_size, -1, num_pharm_types)
                noise = noise.reshape(batch_size, -1, num_pharm_types)[:, :x4_x_t_inpaint.shape[1]]

                x4_x_t_inpaint = x4_x_t_inpaint + add_noise_to_inpainted_x4_type * noise
                x4_x_t = x4_x_t.reshape(batch_size, -1, num_pharm_types)
                x4_x_t[:, :x4_x_t_inpaint.shape[1]] = x4_x_t_inpaint
                x4_x_t = x4_x_t.reshape(-1, num_pharm_types)
            else:
                x4_x_t_inpaint = x4_x_t_inpaint + add_noise_to_inpainted_x4_type * noise
                x4_x_t = x4_x_t_inpaint


    # get noise parameters for current timestep t and previous timestep prev_t
    noise_params_current = _get_noise_params_for_timestep(params, current_t)

    # pass only current params to model input preparation
    x1_params_current = noise_params_current['x1']
    x2_params_current = noise_params_current['x2']
    x3_params_current = noise_params_current['x3']
    x4_params_current = noise_params_current['x4']

    # get current data
    input_dict = _prepare_model_input(
        model_pl.model.device, torch.float32, batch_size, current_t,
        x1_pos_t, x1_x_t, x1_batch, x1_bond_edge_x_t, bond_edge_index_x1, virtual_node_mask_x1, x1_params_current,
        x2_pos_t, x2_x_t, x2_batch, virtual_node_mask_x2, x2_params_current,
        x3_pos_t, x3_x_t, x3_batch, virtual_node_mask_x3, x3_params_current,
        x4_pos_t, x4_direction_t, x4_x_t, x4_batch, virtual_node_mask_x4, x4_params_current,
    )

    # predict noise with neural network    
    with torch.no_grad():
        _, output_dict = model_pl.model.forward(input_dict)


    x1_x_out = output_dict['x1']['decoder']['denoiser']['x_out'].detach().cpu()
    x1_bond_edge_x_out = output_dict['x1']['decoder']['denoiser']['bond_edge_x_out'].detach().cpu()
    x1_pos_out = output_dict['x1']['decoder']['denoiser']['pos_out'].detach().cpu()

    # Correct the COM during inpainting
    if inpaint_x1_pos and do_partial_atom_inpainting and (x1_t > stop_inpainting_at_time_x1):
        x1_alpha_dash_t = x1_params_current['alpha_dash_t']
        x1_sigma_dash_t = x1_params_current['sigma_dash_t']

        # Predict x0 from model output
        x1_pos_0_pred = (x1_pos_t - x1_sigma_dash_t * x1_pos_out) / x1_alpha_dash_t

        # Get ground truth x0 for inpainted atoms
        x1_pos_0_inpaint_reshaped = inpainting_dict['x1_pos_inpainting_trajectory'][0].repeat(batch_size, 1, 1)

        # Correct the x0 prediction by inserting the ground truth for inpainted atoms
        x1_pos_0_corrected = x1_pos_0_pred.clone()
        x1_pos_0_corrected_reshaped = x1_pos_0_corrected.reshape(batch_size, -1, 3)
        num_inpaint_atoms_this_batch = x1_pos_0_inpaint_reshaped.shape[1]
        x1_pos_0_corrected_reshaped[:, :num_inpaint_atoms_this_batch, :] = x1_pos_0_inpaint_reshaped
        x1_pos_0_corrected = x1_pos_0_corrected_reshaped.reshape(-1, 3)

        # Recenter the corrected x0 prediction to maintain translational invariance
        x1_pos_0_corrected_com = torch_scatter.scatter_mean(x1_pos_0_corrected[~virtual_node_mask_x1], x1_batch[~virtual_node_mask_x1], dim=0)
        x1_pos_0_corrected = x1_pos_0_corrected - x1_pos_0_corrected_com[x1_batch]

        # Derive the corrected noise (eps_theta) from the corrected x0
        x1_pos_out = (x1_pos_t - x1_alpha_dash_t * x1_pos_0_corrected) / x1_sigma_dash_t
    else:
        # Original logic for non-inpainting case
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

    if include_x0_pred:
        x1_sigma_dash_t = x1_params_current['sigma_dash_t']
        x1_sigma_dash_t_1 = x1_params_current['sigma_dash_t_1']
        x2_sigma_dash_t = x2_params_current['sigma_dash_t']
        x3_sigma_dash_t = x3_params_current['sigma_dash_t']
        x4_sigma_dash_t = x4_params_current['sigma_dash_t']

        x1_alpha_dash_t = x1_params_current['alpha_dash_t']
        x2_alpha_dash_t = x2_params_current['alpha_dash_t']
        x3_alpha_dash_t = x3_params_current['alpha_dash_t']
        x4_alpha_dash_t = x4_params_current['alpha_dash_t']

        x1_pos_0 = (x1_pos_t - x1_sigma_dash_t * x1_pos_out) / x1_alpha_dash_t
        x1_x_0 = (x1_x_t - x1_sigma_dash_t * x1_x_out) / x1_alpha_dash_t
        x1_bond_edge_x_0 = (x1_bond_edge_x_t - x1_sigma_dash_t * x1_bond_edge_x_out) / x1_alpha_dash_t

        x2_pos_0 = (x2_pos_t - x2_sigma_dash_t * x2_pos_out) / x2_alpha_dash_t
        x2_x_0 = x2_x_t

        x3_pos_0 = (x3_pos_t - x3_sigma_dash_t * x3_pos_out) / x3_alpha_dash_t
        x3_x_0 = (x3_x_t - x3_sigma_dash_t * x3_x_out) / x3_alpha_dash_t

        x4_pos_0 = (x4_pos_t - x4_sigma_dash_t * x4_pos_out) / x4_alpha_dash_t
        x4_direction_0 = (x4_direction_t - x4_sigma_dash_t * x4_direction_out) / x4_alpha_dash_t
        x4_x_0 = (x4_x_t - x4_sigma_dash_t * x4_x_out) / x4_alpha_dash_t

        # reset virtual nodes (common to both paths)
        x1_pos_0[virtual_node_mask_x1] = x1_pos_t[virtual_node_mask_x1]
        x1_x_0[virtual_node_mask_x1] = x1_x_t[virtual_node_mask_x1]
        x2_pos_0[virtual_node_mask_x2] = x2_pos_t[virtual_node_mask_x2]
        x2_x_0[virtual_node_mask_x2] = x2_x_t[virtual_node_mask_x2]
        x3_pos_0[virtual_node_mask_x3] = x3_pos_t[virtual_node_mask_x3]
        x3_x_0[virtual_node_mask_x3] = x3_x_t[virtual_node_mask_x3]
        x4_pos_0[virtual_node_mask_x4] = x4_pos_t[virtual_node_mask_x4]
        x4_direction_0[virtual_node_mask_x4] = x4_direction_t[virtual_node_mask_x4]
        x4_x_0[virtual_node_mask_x4] = x4_x_t[virtual_node_mask_x4]

        x0_pred = {
            'x1_pos_0': x1_pos_0, 'x1_x_0': x1_x_0, 'x1_bond_edge_x_0': x1_bond_edge_x_0,
            'x2_pos_0': x2_pos_0, 'x2_x_0': x2_x_0, 
            'x3_pos_0': x3_pos_0, 'x3_x_0': x3_x_0,
            'x4_pos_0': x4_pos_0, 'x4_direction_0': x4_direction_0, 'x4_x_0': x4_x_0,
        }


    # Perform reverse denoising step using helper function
    next_state = _perform_reverse_denoising_step(
        current_t, # Pass current time t (tau_i)
        batch_size, 
        noise_params_current, # Pass params for current time t
        # Current states (x_t)
        x1_pos_t, x1_x_t, x1_bond_edge_x_t, x1_batch, virtual_node_mask_x1, 
        x2_pos_t, x2_x_t, x2_batch, virtual_node_mask_x2, 
        x3_pos_t, x3_x_t, x3_batch, virtual_node_mask_x3, 
        x4_pos_t, x4_direction_t, x4_x_t, x4_batch, virtual_node_mask_x4, 
        # Model outputs (predicted noise or x0)
        x1_pos_out, x1_x_out, x1_bond_edge_x_out,
        x2_pos_out, 
        x3_pos_out, x3_x_out,
        x4_pos_out, x4_direction_out, x4_x_out,
        denoising_noise_scale, inject_noise_at_ts, inject_noise_scales,
    )

    if pbar is not None:
        pbar.update(1)

    del output_dict
    del input_dict

    current_time_idx += 1 # Move to next index in time_steps sequence

    if include_x0_pred:
        next_state['x0_pred'] = x0_pred

    return current_time_idx, next_state # next_state is a dictionary with updated states


def _extract_generated_samples(
        x1_x_t, x1_pos_t, x1_bond_edge_x_t, virtual_node_mask_x1,
        x2_pos_t, virtual_node_mask_x2,
        x3_pos_t, x3_x_t, virtual_node_mask_x3,
        x4_pos_t, x4_direction_t, x4_x_t, virtual_node_mask_x4,
        params, batch_size,
        ):
    """
    Extract final structures, and re-scale.

    Arguments
    ---------
    x1_x_t: The x1 feature tensor.
    x1_pos_t: The x1 position tensor.
    x1_bond_edge_x_t: The x1 bond edge feature tensor.
    virtual_node_mask_x1: The x1 virtual node mask tensor.
    x2_pos_t: The x2 position tensor.
    virtual_node_mask_x2: The x2 virtual node mask tensor.
    x3_pos_t: The x3 position tensor.
    x3_x_t: The x3 feature tensor.
    virtual_node_mask_x3: The x3 virtual node mask tensor.
    x4_pos_t: The x4 position tensor.
    x4_direction_t: The x4 direction tensor.
    x4_x_t: The x4 feature tensor.
    virtual_node_mask_x4: The x4 virtual node mask tensor.
    params: The parameters.
    batch_size: The batch size.

    Returns
    -------
    list: The generated structures.
        dict:
            x1: dict
                    atoms: torch.Tensor
                    bonds: torch.Tensor
                    positions: torch.Tensor
                x2: dict
                    positions: torch.Tensor
                x3: dict
                    charges: torch.Tensor
                    positions: torch.Tensor
                x4: dict
                    types: torch.Tensor
                    positions: torch.Tensor
                    directions: torch.Tensor
    """
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
