import math
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch_scatter
import torch_geometric
from torch_geometric.nn import radius_graph

from .egnn.egnn import EGNN, MultiLayerPerceptron, GaussianSmearing
from .equiformer_v2_encoder import EquiformerV2

from equiformer_v2.nets.equiformer_v2.so3 import SO3_Embedding, SO3_Grid
from equiformer_v2.nets.equiformer_v2.transformer_block import FeedForwardNetwork
from equiformer_v2.nets.equiformer_v2.module_list import ModuleListInfo


from utils.add_virtual_edges_to_edge_index import add_virtual_edges_to_edge_index
from utils.positional_encoding import positional_encoding

import e3nn
from equiformer_operations import FeedForwardNetwork_equiformer, convert_e3nn_to_equiformerv2, convert_equiformerv2_to_e3nn


def remap_values(remapping_tuple, input_tensor):
    """
    # credit to: https://discuss.pytorch.org/t/cv2-remap-in-pytorch/99354/8
    
    Maps integer values in input_tensor to new integer values specified by the map remapping_tuple[0]:remapping_tuple[1]
    
    Args:
        remapping_tuple (Tuple(torch.LongTensor, torch.LongTensor))
        input_tensor (torch.LongTensor)
    Returns:
        (torch.LongTensor) with new values
    """
    index = torch.bucketize(input_tensor.ravel(), remapping_tuple[0])
    return remapping_tuple[1][index].reshape(input_tensor.shape)


# useful for debugging
def display_dict(d, indent=''):
    for key in d:
        print(indent + key)
        value = d[key]
        if isinstance(value, dict):
            print_dict_key_structure(value, indent = indent + '    ')


class Model(torch.nn.Module):
    
    def __init__(self, params):
        super(Model, self).__init__()
        
        self.params = params
        self.device = 'cpu'
                
        self.x1_bond_diffusion = params['x1_bond_diffusion'] 
        
        lmax_list = params['lmax_list']
        grid_resolution = params['grid_resolution']
        self.joint_SO3_grid = ModuleListInfo('({}, {})'.format(max(lmax_list), max(lmax_list)))
        for l in range(max(lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, 
                        m, 
                        resolution=grid_resolution, 
                        normalization='component'
                    )
                )
            self.joint_SO3_grid.append(SO3_m_grid)
            
        
        
        # Joint Module

        self.explicit_diffusion_variables = params['explicit_diffusion_variables']
        self.exclude_variables_from_decoder_heterogeneous_graph = params['exclude_variables_from_decoder_heterogeneous_graph']
        decoder_heterogeneous_variables = deepcopy([x_ for x_ in self.explicit_diffusion_variables if x_ not in self.exclude_variables_from_decoder_heterogeneous_graph])
        
        
        # heterogeneous graph encoding (in decoder, before joint global code processing)
            # change to only encode a specified subset of the variables (must be in the same absolute reference frame, and exclude COM variable)
            # we could project any re-centered variables back into the absolute frame if we wish, but this complicates things
        self.decoder_joint_heterogeneous_graph_encoder = None
        if 'decoder_heterogeneous_graph_encoder' in params:
            if params['decoder_heterogeneous_graph_encoder']['use']:
                assert len(decoder_heterogeneous_variables) > 1
                
                for x_ in decoder_heterogeneous_variables:
                    assert params[x_]['decoder']['encoder']['sphere_channels'] == params['decoder_heterogeneous_graph_encoder']['sphere_channels']
                
                hetero_params = params['decoder_heterogeneous_graph_encoder']
                self.decoder_joint_heterogeneous_graph_encoder = EquiformerV2(
                    
                    final_block_channels = 0,
                    
                    num_layers = hetero_params['num_layers'],
                    input_sphere_channels = hetero_params['sphere_channels'],
                    sphere_channels = hetero_params['sphere_channels'],
                    attn_hidden_channels = hetero_params['attn_hidden_channels'],
                    num_heads = hetero_params['num_heads'], 
                    attn_alpha_channels = hetero_params['attn_alpha_channels'],
                    attn_value_channels = hetero_params['attn_value_channels'],
                    ffn_hidden_channels = hetero_params['ffn_hidden_channels'],
                    
                    norm_type='layer_norm_sh',
                    
                    lmax_list = hetero_params['lmax_list'],
                    mmax_list = hetero_params['mmax_list'],
                    grid_resolution = hetero_params['grid_resolution'],
                    cutoff = hetero_params['cutoff'],
                    
                    num_sphere_samples=hetero_params['num_sphere_samples'],
                    edge_channels=hetero_params['edge_channels'],
                    
                    use_atom_edge_embedding=True,
                    share_atom_edge_embedding=False,
                    use_m_share_rad=False,
                    distance_function="gaussian",
                    num_distance_basis=600, # not used; hard-coded by Equiformer-V2 to 600
                    
                    attn_activation='silu',
                    use_s2_act_attn=False, 
                    use_attn_renorm=True,
                    ffn_activation='silu',
                    use_gate_act=False,
                    use_grid_mlp=True, 
                    use_sep_s2_act=True,
                    alpha_drop=0.0,
                    drop_path_rate=0.0, 
                    proj_drop=0.0, 
                    weight_init='normal',
                )
        
        
        # Joint processing of global latent representations
        
        # these COULD share parameters, but for now, they are initialized separately for each explicit diffusion variable
        if 'x1' in self.explicit_diffusion_variables:
            self.x1_decoder_global_timestep_embedding = torch.nn.Linear(
                sum([params[x_]['decoder']['time_embedding_size'] for x_ in self.explicit_diffusion_variables]), # in ['x1', 'x2', ...]
                params['x1']['decoder']['node_channels'],
            )
        if 'x2' in self.explicit_diffusion_variables:
            self.x2_decoder_global_timestep_embedding = torch.nn.Linear(
                sum([params[x_]['decoder']['time_embedding_size'] for x_ in self.explicit_diffusion_variables]), 
                params['x2']['decoder']['node_channels'],
            )
        if 'x3' in self.explicit_diffusion_variables:
            self.x3_decoder_global_timestep_embedding = torch.nn.Linear(
                sum([params[x_]['decoder']['time_embedding_size'] for x_ in self.explicit_diffusion_variables]),
                params['x3']['decoder']['node_channels'],
            )
        if 'x4' in self.explicit_diffusion_variables:
            self.x4_decoder_global_timestep_embedding = torch.nn.Linear(
                sum([params[x_]['decoder']['time_embedding_size'] for x_ in self.explicit_diffusion_variables]), 
                params['x4']['decoder']['node_channels'],
            )
        
        # these could also all share parameters, but for now, they are initialized separately for each explicit diffusion variable
        if 'x1' in self.explicit_diffusion_variables:
            self.x1_decoder_global_l1_embedding = FeedForwardNetwork(
                sphere_channels = sum([params[x_]['decoder']['node_channels'] for x_ in self.explicit_diffusion_variables]),
                hidden_channels = params['ffn_hidden_channels'], 
                output_channels = params['x1']['decoder']['node_channels'],
                lmax_list = params['lmax_list'],
                mmax_list = params['mmax_list'],
                SO3_grid = self.joint_SO3_grid,  
                activation = 'silu',
                use_gate_act = False,
                use_grid_mlp = True,
                use_sep_s2_act = True,
            )
        if 'x2' in self.explicit_diffusion_variables:
            # self.x2_decoder_global_l1_embedding = self.x1_decoder_global_l1_embedding # share parameters ?
            self.x2_decoder_global_l1_embedding = FeedForwardNetwork(
                sphere_channels = sum([params[x_]['decoder']['node_channels'] for x_ in self.explicit_diffusion_variables]),
                hidden_channels = params['ffn_hidden_channels'], 
                output_channels = params['x2']['decoder']['node_channels'], 
                lmax_list = params['lmax_list'], # same as x1
                mmax_list = params['mmax_list'], # same as x1
                SO3_grid = self.joint_SO3_grid, # shared; same as x1
                activation = 'silu',
                use_gate_act = False,
                use_grid_mlp = True,
                use_sep_s2_act = True,
            )
        if 'x3' in self.explicit_diffusion_variables:
            # self.x3_decoder_global_l1_embedding = self.x1_decoder_global_l1_embedding # share parameters ?
            self.x3_decoder_global_l1_embedding = FeedForwardNetwork(
                sphere_channels = sum([params[x_]['decoder']['node_channels'] for x_ in self.explicit_diffusion_variables]),
                hidden_channels = params['ffn_hidden_channels'], 
                output_channels = params['x3']['decoder']['node_channels'], 
                lmax_list = params['lmax_list'], # same as x1
                mmax_list = params['mmax_list'], # same as x1
                SO3_grid = self.joint_SO3_grid, # shared; same as x1
                activation = 'silu',
                use_gate_act = False,
                use_grid_mlp = True,
                use_sep_s2_act = True,
            )
        if 'x4' in self.explicit_diffusion_variables:
            # self.x4_decoder_global_l1_embedding = self.x1_decoder_global_l1_embedding # share parameters ?
            self.x4_decoder_global_l1_embedding = FeedForwardNetwork(
                sphere_channels = sum([params[x_]['decoder']['node_channels'] for x_ in self.explicit_diffusion_variables]),
                hidden_channels = params['ffn_hidden_channels'], 
                output_channels = params['x4']['decoder']['node_channels'], 
                lmax_list = params['lmax_list'], # same as x1
                mmax_list = params['mmax_list'], # same as x1
                SO3_grid = self.joint_SO3_grid, # shared; same as x1
                activation = 'silu',
                use_gate_act = False,
                use_grid_mlp = True,
                use_sep_s2_act = True,
            )
        
        
        # for mixing l=0 and l=1 channels of the joint embeddings prior to denoising
                # these could also all share parameters, but for now, they are initialized separately for each explicit diffusion variable
        if 'x1' in self.explicit_diffusion_variables:
            lmax = 1
            num_channels = params['x1']['decoder']['node_channels']
            self.x1_decoder_equiformer_tensor_product = FeedForwardNetwork_equiformer(
                irreps_node_input = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]), 
                irreps_node_attr = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                irreps_node_output = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]), 
                irreps_mlp_mid = e3nn.o3.Irreps(''.join([f'{num_channels//(2*(i+1))}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                proj_drop=0.0,
            )
        if 'x2' in self.explicit_diffusion_variables:
            lmax = 1
            num_channels = params['x2']['decoder']['node_channels']
            self.x2_decoder_equiformer_tensor_product = FeedForwardNetwork_equiformer(
                irreps_node_input = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]), 
                irreps_node_attr = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                irreps_node_output = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]), 
                irreps_mlp_mid = e3nn.o3.Irreps(''.join([f'{num_channels//(2*(i+1))}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                proj_drop=0.0,
            )
        if 'x3' in self.explicit_diffusion_variables:
            lmax = 1
            num_channels = params['x3']['decoder']['node_channels']
            self.x3_decoder_equiformer_tensor_product = FeedForwardNetwork_equiformer(
                irreps_node_input = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]), 
                irreps_node_attr = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                irreps_node_output = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]), 
                irreps_mlp_mid = e3nn.o3.Irreps(''.join([f'{num_channels//(2*(i+1))}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                proj_drop=0.0,
            )
        if 'x4' in self.explicit_diffusion_variables:
            lmax = 1
            num_channels = params['x4']['decoder']['node_channels']
            self.x4_decoder_equiformer_tensor_product = FeedForwardNetwork_equiformer(
                irreps_node_input = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]), 
                irreps_node_attr = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                irreps_node_output = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]), 
                irreps_mlp_mid = e3nn.o3.Irreps(''.join([f'{num_channels//(2*(i+1))}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                proj_drop=0.0,
            )
        
        
        
        # Denoising Modules
        
        if 'x1' in self.explicit_diffusion_variables:
            
            #params['x1']['decoder']['encoder']
            self.x1_decoder_encoder_embedding = torch.nn.Linear(
                params['x1']['decoder']['input_node_channels'], # (noised) one hot atomic code embedding
                params['x1']['decoder']['node_channels'], # linear embedding
            )
            
            self.x1_decoder_local_timestep_embedding = torch.nn.Linear(
                params['x1']['decoder']['time_embedding_size'],
                params['x1']['decoder']['node_channels'],
            )
    
            
            x1_decoder_encoder_params = params['x1']['decoder']['encoder']
            assert params['x1']['decoder']['node_channels'] == x1_decoder_encoder_params['input_sphere_channels']
            assert x1_decoder_encoder_params['input_sphere_channels'] == x1_decoder_encoder_params['sphere_channels']
            
            self.x1_decoder_encoder_bond_edge_embedding = None
            if self.x1_bond_diffusion:
                self.x1_decoder_encoder_bond_edge_embedding = torch.nn.Linear(
                    x1_decoder_encoder_params['input_bond_channels'], # (noised) one hot bond type embedding
                    x1_decoder_encoder_params['edge_attr_channels'], # linear embedding
                )

            
            self.x1_decoder_encoder = EquiformerV2(
                    
                    final_block_channels = 0,
                    
                    num_layers = x1_decoder_encoder_params['num_layers'],
                    input_sphere_channels = x1_decoder_encoder_params['input_sphere_channels'],
                    sphere_channels = x1_decoder_encoder_params['sphere_channels'],
                    attn_hidden_channels = x1_decoder_encoder_params['attn_hidden_channels'],
                    num_heads = x1_decoder_encoder_params['num_heads'], 
                    attn_alpha_channels = x1_decoder_encoder_params['attn_alpha_channels'],
                    attn_value_channels = x1_decoder_encoder_params['attn_value_channels'],
                    ffn_hidden_channels = x1_decoder_encoder_params['ffn_hidden_channels'],
                    
                    norm_type='layer_norm_sh',
                    
                    lmax_list = x1_decoder_encoder_params['lmax_list'],
                    mmax_list = x1_decoder_encoder_params['mmax_list'],
                    grid_resolution = x1_decoder_encoder_params['grid_resolution'],
                    cutoff = x1_decoder_encoder_params['cutoff'],
                
                    num_sphere_samples=x1_decoder_encoder_params['num_sphere_samples'],
                
                    edge_attr_input_channels = x1_decoder_encoder_params['edge_attr_channels'] if self.x1_bond_diffusion else 0,

                
                    edge_channels=x1_decoder_encoder_params['edge_channels'],
                
                    use_atom_edge_embedding=True,
                    share_atom_edge_embedding=False,
                    use_m_share_rad=False,
                    distance_function="gaussian",
                    num_distance_basis=600, # not used; hard-coded by Equiformer-V2 to 600
                
                    attn_activation='silu',
                    use_s2_act_attn=False, 
                    use_attn_renorm=True,
                    ffn_activation='silu',
                    use_gate_act=False,
                    use_grid_mlp=True, 
                    use_sep_s2_act=True,
                    alpha_drop=0.0,
                    drop_path_rate=0.0, 
                    proj_drop=0.0, 
                    weight_init='normal',
            )
            
            
            assert params['x1']['decoder']['node_channels'] == params['x1']['decoder']['encoder']['sphere_channels']
            
            self.x1_decoder_denoiser_MLP = MultiLayerPerceptron(
                input_dim = params['x1']['decoder']['node_channels'],
                hidden_dim = params['x1']['decoder']['denoiser']['MLP_hidden_dim'], 
                output_dim = params['x1']['decoder']['denoiser']['output_node_channels'],
                num_hidden_layers = params['x1']['decoder']['denoiser']['num_MLP_hidden_layers'], 
                activation=torch.nn.LeakyReLU(0.2),
                include_final_activation = False,
            )
            
            self.x1_decoder_denoiser_bond_MLP = None
            if self.x1_bond_diffusion:
                bond_distance_expansion_dim = 32
                self.x1_decoder_denoiser_bond_MLP =  MultiLayerPerceptron(
                    input_dim = 2 * params['x1']['decoder']['node_channels'] + params['x1']['decoder']['encoder']['input_bond_channels'] + bond_distance_expansion_dim,
                    hidden_dim = params['x1']['decoder']['denoiser']['MLP_hidden_dim'], 
                    output_dim = params['x1']['decoder']['denoiser']['output_bond_channels'],
                    num_hidden_layers = params['x1']['decoder']['denoiser']['num_MLP_hidden_layers'], 
                    activation=torch.nn.LeakyReLU(0.2),
                    include_final_activation = False,
                )
                self.x1_decoder_denoiser_bond_distance_scalar_expansion = GaussianSmearing(
                    start = 0.0,
                    stop = 5.0,
                    num_gaussians = bond_distance_expansion_dim,
                )
            
            
            if params['x1']['decoder']['denoiser']['use_e3nn']:
                
                #self.SO3_grid = self.x1_decoder_encoder.SO3_grid
                lmax_list = params['x1']['decoder']['denoiser']['e3nn']['lmax_list']
                grid_resolution = params['x1']['decoder']['denoiser']['e3nn']['grid_resolution']
                self.x1_denoiser_SO3_grid = ModuleListInfo('({}, {})'.format(max(lmax_list), max(lmax_list)))
                for l in range(max(lmax_list) + 1):
                    SO3_m_grid = nn.ModuleList()
                    for m in range(max(lmax_list) + 1):
                        SO3_m_grid.append(
                            SO3_Grid(
                                l, 
                                m, 
                                resolution=grid_resolution, 
                                normalization='component'
                            )
                        )
                    self.x1_denoiser_SO3_grid.append(SO3_m_grid)
                
                self.x1_decoder_denoiser_E3NN = FeedForwardNetwork(
                    sphere_channels = params['x1']['decoder']['node_channels'],
                    hidden_channels = params['x1']['decoder']['denoiser']['e3nn']['ffn_hidden_channels'], 
                    output_channels = 1,
                    lmax_list = params['x1']['decoder']['denoiser']['e3nn']['lmax_list'],
                    mmax_list = params['x1']['decoder']['denoiser']['e3nn']['mmax_list'],
                    SO3_grid = self.x1_denoiser_SO3_grid,  
                    activation = 'silu',
                    use_gate_act = False,
                    use_grid_mlp = True,
                    use_sep_s2_act = True,
                )
            
            if params['x1']['decoder']['denoiser']['use_egnn_positions_update'] == True:
                self.x1_decoder_denoiser_EGNN = EGNN(
                    node_embedding_dim = params['x1']['decoder']['node_channels'], 
                    node_output_embedding_dim = params['x1']['decoder']['denoiser']['output_node_channels'], # ignored
                    edge_attr_dim = 0, 
                    distance_expansion_dim = params['x1']['decoder']['denoiser']['egnn']['distance_expansion_dim'], 
                    normalize_distance_vectors = params['x1']['decoder']['denoiser']['egnn']['normalize_egnn_vectors'], 
                    num_MLP_hidden_layers = params['x1']['decoder']['denoiser']['egnn']['num_MLP_hidden_layers'],
                    MLP_hidden_dim = params['x1']['decoder']['denoiser']['egnn']['MLP_hidden_dim'],
                )        
        

        
        
        if 'x2' in self.explicit_diffusion_variables:
        
            self.x2_decoder_encoder_embedding = torch.nn.Linear(
                params['x2']['decoder']['input_node_channels'], # one hot embedding of real node vs virtual node
                params['x2']['decoder']['node_channels'], # linear embedding
            )
            
            self.x2_decoder_local_timestep_embedding = torch.nn.Linear(
                params['x2']['decoder']['time_embedding_size'],
                params['x2']['decoder']['node_channels'],
            )

            x2_decoder_encoder_params = params['x2']['decoder']['encoder']
            assert params['x2']['decoder']['node_channels'] == x2_decoder_encoder_params['input_sphere_channels']
            assert x2_decoder_encoder_params['input_sphere_channels'] == x2_decoder_encoder_params['sphere_channels']
            self.x2_decoder_encoder = EquiformerV2(
                    
                    final_block_channels = params['x2']['decoder']['encoder']['sphere_channels'] + params['x3']['decoder']['encoder']['sphere_channels'] if (self.combine_x2_x3_convolution_decoder) else 0,
                    
                    num_layers = x2_decoder_encoder_params['num_layers'],
                    input_sphere_channels = x2_decoder_encoder_params['input_sphere_channels'],
                    sphere_channels = x2_decoder_encoder_params['sphere_channels'],
                    attn_hidden_channels = x2_decoder_encoder_params['attn_hidden_channels'],
                    num_heads = x2_decoder_encoder_params['num_heads'], 
                    attn_alpha_channels = x2_decoder_encoder_params['attn_alpha_channels'],
                    attn_value_channels = x2_decoder_encoder_params['attn_value_channels'],
                    ffn_hidden_channels = x2_decoder_encoder_params['ffn_hidden_channels'],
                    
                    norm_type='layer_norm_sh',
                    
                    lmax_list = x2_decoder_encoder_params['lmax_list'],
                    mmax_list = x2_decoder_encoder_params['mmax_list'],
                    grid_resolution = x2_decoder_encoder_params['grid_resolution'],
                    cutoff = x2_decoder_encoder_params['cutoff'],
                
                    num_sphere_samples=x2_decoder_encoder_params['num_sphere_samples'],
                    edge_channels=x2_decoder_encoder_params['edge_channels'],
                
                    use_atom_edge_embedding=True,
                    share_atom_edge_embedding=False,
                    use_m_share_rad=False,
                    distance_function="gaussian",
                    num_distance_basis=600, # not used; hard-coded by Equiformer-V2 to 600
                
                    attn_activation='silu',
                    use_s2_act_attn=False, 
                    use_attn_renorm=True,
                    ffn_activation='silu',
                    use_gate_act=False,
                    use_grid_mlp=True, 
                    use_sep_s2_act=True,
                    alpha_drop=0.0,
                    drop_path_rate=0.0, 
                    proj_drop=0.0, 
                    weight_init='normal',
            )
            
            
            assert params['x2']['decoder']['node_channels'] == params['x2']['decoder']['encoder']['sphere_channels']
            
            if params['x2']['decoder']['denoiser']['use_e3nn']:
                
                #self.SO3_grid = self.x2_decoder_encoder.SO3_grid
                lmax_list = params['x2']['decoder']['denoiser']['e3nn']['lmax_list']
                grid_resolution = params['x2']['decoder']['denoiser']['e3nn']['grid_resolution']
                self.x2_denoiser_SO3_grid = ModuleListInfo('({}, {})'.format(max(lmax_list), max(lmax_list)))
                for l in range(max(lmax_list) + 1):
                    SO3_m_grid = nn.ModuleList()
                    for m in range(max(lmax_list) + 1):
                        SO3_m_grid.append(
                            SO3_Grid(
                                l, 
                                m, 
                                resolution=grid_resolution, 
                                normalization='component'
                            )
                        )
                    self.x2_denoiser_SO3_grid.append(SO3_m_grid)
                
                self.x2_decoder_denoiser_E3NN = FeedForwardNetwork(
                    sphere_channels = params['x2']['decoder']['node_channels'],
                    hidden_channels = params['x2']['decoder']['denoiser']['e3nn']['ffn_hidden_channels'],
                    output_channels = 1,
                    lmax_list = params['x2']['decoder']['denoiser']['e3nn']['lmax_list'],
                    mmax_list = params['x2']['decoder']['denoiser']['e3nn']['mmax_list'],
                    SO3_grid = self.x2_denoiser_SO3_grid,  
                    activation = 'silu',
                    use_gate_act = False,
                    use_grid_mlp = True,
                    use_sep_s2_act = True,
                )
            
            if self.params['x2']['decoder']['denoiser']['use_egnn_positions_update']:
                self.x2_decoder_denoiser_EGNN = EGNN(
                    node_embedding_dim = params['x2']['decoder']['node_channels'], 
                    node_output_embedding_dim = params['x2']['decoder']['denoiser']['output_node_channels'], # node output embeddings are ignored
                    edge_attr_dim = 0, 
                    distance_expansion_dim = params['x2']['decoder']['denoiser']['egnn']['distance_expansion_dim'], 
                    normalize_distance_vectors = params['x2']['decoder']['denoiser']['egnn']['normalize_egnn_vectors'], 
                    num_MLP_hidden_layers = params['x2']['decoder']['denoiser']['egnn']['num_MLP_hidden_layers'],
                    MLP_hidden_dim = params['x2']['decoder']['denoiser']['egnn']['MLP_hidden_dim'],
                )
        
        

        
        if 'x3' in self.explicit_diffusion_variables:
            
            self.x3_decoder_scalar_expansion = GaussianSmearing(
                start = params['x3']['decoder']['scalar_expansion_min'],
                stop = params['x3']['decoder']['scalar_expansion_max'],
                num_gaussians = params['x3']['decoder']['input_node_channels'],
            )
            self.x3_decoder_encoder_embedding = torch.nn.Linear(
                params['x3']['decoder']['input_node_channels'], # dimension of RBF expansion of coulombic potentials / partial charges
                params['x3']['decoder']['node_channels'], # linear embedding
            )
            self.x3_decoder_local_timestep_embedding = torch.nn.Linear(
                params['x3']['decoder']['time_embedding_size'],
                params['x3']['decoder']['node_channels'],
            )

            
            x3_decoder_encoder_params = params['x3']['decoder']['encoder']
            assert params['x3']['decoder']['node_channels'] == x3_decoder_encoder_params['input_sphere_channels']
            assert x3_decoder_encoder_params['input_sphere_channels'] == x3_decoder_encoder_params['sphere_channels']
            self.x3_decoder_encoder = EquiformerV2(
                    
                    final_block_channels = 0,
                    
                    num_layers = x3_decoder_encoder_params['num_layers'],
                    input_sphere_channels = x3_decoder_encoder_params['input_sphere_channels'],
                    sphere_channels = x3_decoder_encoder_params['sphere_channels'],
                    attn_hidden_channels = x3_decoder_encoder_params['attn_hidden_channels'],
                    num_heads = x3_decoder_encoder_params['num_heads'], 
                    attn_alpha_channels = x3_decoder_encoder_params['attn_alpha_channels'],
                    attn_value_channels = x3_decoder_encoder_params['attn_value_channels'],
                    ffn_hidden_channels = x3_decoder_encoder_params['ffn_hidden_channels'],
                    
                    norm_type='layer_norm_sh',
                    
                    lmax_list = x3_decoder_encoder_params['lmax_list'],
                    mmax_list = x3_decoder_encoder_params['mmax_list'],
                    grid_resolution = x3_decoder_encoder_params['grid_resolution'],
                    cutoff = x3_decoder_encoder_params['cutoff'],
                
                    num_sphere_samples=x3_decoder_encoder_params['num_sphere_samples'],
                    edge_channels=x3_decoder_encoder_params['edge_channels'],
                
                    use_atom_edge_embedding=True,
                    share_atom_edge_embedding=False,
                    use_m_share_rad=False,
                    distance_function="gaussian",
                    num_distance_basis=600, # not used; hard-coded by Equiformer-V2 to 600
                
                    attn_activation='silu',
                    use_s2_act_attn=False, 
                    use_attn_renorm=True,
                    ffn_activation='silu',
                    use_gate_act=False,
                    use_grid_mlp=True, 
                    use_sep_s2_act=True,
                    alpha_drop=0.0,
                    drop_path_rate=0.0, 
                    proj_drop=0.0, 
                    weight_init='normal',
            )
            
            
            assert params['x3']['decoder']['node_channels'] == params['x3']['decoder']['encoder']['sphere_channels']
            self.x3_decoder_denoiser_MLP = MultiLayerPerceptron(
                input_dim = params['x3']['decoder']['node_channels'], # see above assertion
                hidden_dim = params['x3']['decoder']['denoiser']['MLP_hidden_dim'], 
                output_dim = params['x3']['decoder']['denoiser']['output_node_channels'], # should be 1 for a scalar potential
                num_hidden_layers = params['x3']['decoder']['denoiser']['num_MLP_hidden_layers'], 
                activation=torch.nn.LeakyReLU(0.2),
                include_final_activation = False,
            )
            
                
            if params['x3']['decoder']['denoiser']['use_e3nn']:
                #self.SO3_grid = self.x3_decoder_encoder.SO3_grid
                lmax_list = params['x3']['decoder']['denoiser']['e3nn']['lmax_list']
                grid_resolution = params['x3']['decoder']['denoiser']['e3nn']['grid_resolution']
                self.x3_denoiser_SO3_grid = ModuleListInfo('({}, {})'.format(max(lmax_list), max(lmax_list)))
                for l in range(max(lmax_list) + 1):
                    SO3_m_grid = nn.ModuleList()
                    for m in range(max(lmax_list) + 1):
                        SO3_m_grid.append(
                            SO3_Grid(
                                l, 
                                m, 
                                resolution=grid_resolution, 
                                normalization='component'
                            )
                        )
                    self.x3_denoiser_SO3_grid.append(SO3_m_grid)
                
                self.x3_decoder_denoiser_E3NN = FeedForwardNetwork(
                    sphere_channels = params['x3']['decoder']['node_channels'],
                    hidden_channels = params['x3']['decoder']['denoiser']['e3nn']['ffn_hidden_channels'],
                    output_channels = 1,
                    lmax_list = params['x3']['decoder']['denoiser']['e3nn']['lmax_list'],
                    mmax_list = params['x3']['decoder']['denoiser']['e3nn']['mmax_list'],
                    SO3_grid = self.x3_denoiser_SO3_grid,  
                    activation = 'silu',
                    use_gate_act = False,
                    use_grid_mlp = True,
                    use_sep_s2_act = True,
                )
            
            if params['x3']['decoder']['denoiser']['use_egnn_positions_update']:
                self.x3_decoder_denoiser_EGNN = EGNN(
                    node_embedding_dim = params['x3']['decoder']['node_channels'], 
                    node_output_embedding_dim = params['x3']['decoder']['denoiser']['output_node_channels'], # ignored; x3_decoder_denoiser_MLP is used for feature denoising
                    edge_attr_dim = 0, 
                    distance_expansion_dim = params['x3']['decoder']['denoiser']['egnn']['distance_expansion_dim'], 
                    normalize_distance_vectors = params['x3']['decoder']['denoiser']['egnn']['normalize_egnn_vectors'], 
                    num_MLP_hidden_layers = params['x3']['decoder']['denoiser']['egnn']['num_MLP_hidden_layers'],
                    MLP_hidden_dim = params['x3']['decoder']['denoiser']['egnn']['MLP_hidden_dim'],
                )
    
    

    
        if 'x4' in self.explicit_diffusion_variables:
            self.x4_decoder_encoder_embedding = torch.nn.Linear(
                params['x4']['decoder']['input_node_channels'], # (noised) one hot pharmacophore type embedding
                params['x4']['decoder']['node_channels'], # linear embedding
            )
            
            # embedding l=1 directions, conditioned on pharmacophore type linear embedding.
            self.x4_decoder_encoder_embedding_l1 = FeedForwardNetwork(
                sphere_channels = params['x4']['decoder']['node_channels'],
                hidden_channels = params['ffn_hidden_channels'], 
                output_channels = params['x4']['decoder']['node_channels'],
                lmax_list = params['lmax_list'],
                mmax_list = params['mmax_list'],
                SO3_grid = self.joint_SO3_grid,  
                activation = 'silu',
                use_gate_act = False,
                use_grid_mlp = True,
                use_sep_s2_act = True,
            )
            
            self.x4_decoder_local_timestep_embedding = torch.nn.Linear(
                params['x4']['decoder']['time_embedding_size'],
                params['x4']['decoder']['node_channels'],
            )
        
            
            x4_decoder_encoder_params = params['x4']['decoder']['encoder']
            assert params['x4']['decoder']['node_channels'] == x4_decoder_encoder_params['input_sphere_channels']
            assert x4_decoder_encoder_params['input_sphere_channels'] == x4_decoder_encoder_params['sphere_channels']
            self.x4_decoder_encoder = EquiformerV2(
                    
                    final_block_channels = 0,
                    
                    num_layers = x4_decoder_encoder_params['num_layers'],
                    input_sphere_channels = x4_decoder_encoder_params['input_sphere_channels'],
                    sphere_channels = x4_decoder_encoder_params['sphere_channels'],
                    attn_hidden_channels = x4_decoder_encoder_params['attn_hidden_channels'],
                    num_heads = x4_decoder_encoder_params['num_heads'], 
                    attn_alpha_channels = x4_decoder_encoder_params['attn_alpha_channels'],
                    attn_value_channels = x4_decoder_encoder_params['attn_value_channels'],
                    ffn_hidden_channels = x4_decoder_encoder_params['ffn_hidden_channels'],
                    
                    norm_type='layer_norm_sh',
                    
                    lmax_list = x4_decoder_encoder_params['lmax_list'],
                    mmax_list = x4_decoder_encoder_params['mmax_list'],
                    grid_resolution = x4_decoder_encoder_params['grid_resolution'],
                    cutoff = x4_decoder_encoder_params['cutoff'],
                
                    num_sphere_samples=x4_decoder_encoder_params['num_sphere_samples'],
                    edge_channels=x4_decoder_encoder_params['edge_channels'],
                
                    use_atom_edge_embedding=True,
                    share_atom_edge_embedding=False,
                    use_m_share_rad=False,
                    distance_function="gaussian",
                    num_distance_basis=600, # not used; hard-coded by Equiformer-V2 to 600
                
                    attn_activation='silu',
                    use_s2_act_attn=False, 
                    use_attn_renorm=True,
                    ffn_activation='silu',
                    use_gate_act=False,
                    use_grid_mlp=True, 
                    use_sep_s2_act=True,
                    alpha_drop=0.0,
                    drop_path_rate=0.0, 
                    proj_drop=0.0, 
                    weight_init='normal',
            )
            
            assert params['x4']['decoder']['node_channels'] == params['x4']['decoder']['encoder']['sphere_channels']
            
            self.x4_decoder_denoiser_MLP = MultiLayerPerceptron(
                input_dim = params['x4']['decoder']['node_channels'],
                hidden_dim = params['x4']['decoder']['denoiser']['MLP_hidden_dim'], 
                output_dim = params['x4']['decoder']['denoiser']['output_node_channels'],
                num_hidden_layers = params['x4']['decoder']['denoiser']['num_MLP_hidden_layers'], 
                activation=torch.nn.LeakyReLU(0.2),
                include_final_activation = False,
            )
            
            if params['x4']['decoder']['denoiser']['use_e3nn']:
                #self.SO3_grid = self.x1_decoder_encoder.SO3_grid
                lmax_list = params['x4']['decoder']['denoiser']['e3nn']['lmax_list']
                grid_resolution = params['x4']['decoder']['denoiser']['e3nn']['grid_resolution']
                self.x4_denoiser_SO3_grid = ModuleListInfo('({}, {})'.format(max(lmax_list), max(lmax_list)))
                for l in range(max(lmax_list) + 1):
                    SO3_m_grid = nn.ModuleList()
                    for m in range(max(lmax_list) + 1):
                        SO3_m_grid.append(
                            SO3_Grid(
                                l, 
                                m, 
                                resolution=grid_resolution, 
                                normalization='component'
                            )
                        )
                    self.x4_denoiser_SO3_grid.append(SO3_m_grid)
                
                self.x4_decoder_denoiser_E3NN = FeedForwardNetwork(
                    sphere_channels = params['x4']['decoder']['node_channels'],
                    hidden_channels = params['x4']['decoder']['denoiser']['e3nn']['ffn_hidden_channels'], 
                    output_channels = 1,
                    lmax_list = params['x4']['decoder']['denoiser']['e3nn']['lmax_list'],
                    mmax_list = params['x4']['decoder']['denoiser']['e3nn']['mmax_list'],
                    SO3_grid = self.x4_denoiser_SO3_grid,  
                    activation = 'silu',
                    use_gate_act = False,
                    use_grid_mlp = True,
                    use_sep_s2_act = True,
                )
            
            if params['x4']['decoder']['denoiser']['use_egnn_positions_update'] == True:
                self.x4_decoder_denoiser_EGNN = EGNN(
                    node_embedding_dim = params['x4']['decoder']['node_channels'], 
                    node_output_embedding_dim = params['x4']['decoder']['denoiser']['output_node_channels'], # ignored
                    edge_attr_dim = 0, 
                    distance_expansion_dim = params['x4']['decoder']['denoiser']['egnn']['distance_expansion_dim'], 
                    normalize_distance_vectors = params['x4']['decoder']['denoiser']['egnn']['normalize_egnn_vectors'], 
                    num_MLP_hidden_layers = params['x4']['decoder']['denoiser']['egnn']['num_MLP_hidden_layers'],
                    MLP_hidden_dim = params['x4']['decoder']['denoiser']['egnn']['MLP_hidden_dim'],
                )
            
            self.x4_decoder_denoiser_E3NN_direction = FeedForwardNetwork(
                sphere_channels = params['x4']['decoder']['node_channels'],
                hidden_channels = params['x4']['decoder']['denoiser']['e3nn']['ffn_hidden_channels'], 
                output_channels = 1,
                lmax_list = params['x4']['decoder']['denoiser']['e3nn']['lmax_list'],
                mmax_list = params['x4']['decoder']['denoiser']['e3nn']['mmax_list'],
                SO3_grid = self.x4_denoiser_SO3_grid,  
                activation = 'silu',
                use_gate_act = False,
                use_grid_mlp = True,
                use_sep_s2_act = True,
            )
    
    

    ######## Forward Pass ########
    
    
    def forward_x1_decoder_encoder(self, input_dict, output_dict):
        
        # initial node embeddings for the graph (for discrete or continuous atom features)
        
        x = SO3_Embedding(
            input_dict['x1']['decoder']['pos'].shape[0],
            self.params['x1']['decoder']['encoder']['lmax_list'],
            self.params['x1']['decoder']['encoder']['input_sphere_channels'],
            self.device,
            self.dtype,
        )
        x.embedding[:, 0, :] = self.x1_decoder_encoder_embedding(input_dict['x1']['decoder']['x'])
        
        
        # Adding time step encoding to l=0 node features 
            # we could also concatenate these as extra channels, but then we'd need to expand the l=1 channels as well. 
        
        x1_timestep = input_dict['x1']['decoder']['timestep'] # size (B,) where B is the number of molecules in the batch        
        x1_timestep_embedding = positional_encoding(x1_timestep, dim=self.params['x1']['decoder']['time_embedding_size'], device=self.device)
        x1_timestep_embedding = self.x1_decoder_local_timestep_embedding(x1_timestep_embedding)
        x1_timestep_embedding_pernode = x1_timestep_embedding[input_dict['x1']['decoder']['batch']]
        x.embedding[:, 0, :] = x.embedding[:, 0, :] + x1_timestep_embedding_pernode
        
        
        # 3D graph convolution (with EquiformerV2)
        
        edge_index = radius_graph(
            input_dict['x1']['decoder']['pos'],
            r = 1000000 if self.params['x1']['decoder']['encoder']['fully_connected'] else self.params['x1']['decoder']['encoder']['cutoff'],
            batch = input_dict['x1']['decoder']['batch'],
            max_num_neighbors = self.params['x1']['decoder']['encoder']['max_neighbors'] if 'max_neighbors' in self.params['x1']['decoder']['encoder'] else 1000000,
        )
        
        # True if VN, False otherwise
        virtual_node_mask = input_dict['x1']['decoder']['virtual_node_mask']
        
        if virtual_node_mask is not None:
            force_edges_to_virtual_nodes = self.params['x1']['decoder']['force_edges_to_virtual_nodes']
            if force_edges_to_virtual_nodes and (virtual_node_mask.any()):
                # if a graph instance has multiple VNs, this will introduce edges between those VNs
                    # this will remove self-loops on individual VNs
                edge_index = add_virtual_edges_to_edge_index(edge_index, virtual_node_mask, input_dict['x1']['decoder']['batch'])
        
        
        j, i = edge_index
        edge_distance_vec = input_dict['x1']['decoder']['pos'][j] - input_dict['x1']['decoder']['pos'][i]
        edge_distance = edge_distance_vec.norm(dim=-1)
        
        
        # embedding bond types into edge_attr
        edge_attr = None
        if self.x1_bond_diffusion:
            # fully connected, with both directed edges. Edges to virtual node will have all-zero features.
            undirected_bond_edge_index, undirected_bond_edge_x = torch_geometric.utils.to_undirected(
                input_dict['x1']['decoder']['bond_edge_index'], 
                input_dict['x1']['decoder']['bond_edge_x'],
                num_nodes = input_dict['x1']['decoder']['batch'].shape[0],
                reduce = 'mean',
            )
            dense_bond_edge_attr = torch_geometric.utils.to_dense_adj(
                undirected_bond_edge_index, 
                edge_attr = undirected_bond_edge_x,
                max_num_nodes = input_dict['x1']['decoder']['batch'].shape[0],
            )[0] # (N,N,channels)
            
            edge_attr = dense_bond_edge_attr[edge_index[0], edge_index[1]] # (N_edges, channels)
            edge_attr = self.x1_decoder_encoder_bond_edge_embedding(edge_attr)
        
        
        x1_decoder_encoder_nodes, _ = self.x1_decoder_encoder(
            x, 
            input_dict['x1']['decoder']['pos'], 
            edge_index, 
            edge_distance, 
            edge_distance_vec,
            input_dict['x1']['decoder']['batch'],
            edge_attr = edge_attr,
        )
        
        x1_decoder_encoder_global = SO3_Embedding(
            max(input_dict['x1']['decoder']['batch']) + 1, # number of molecules in batch
            self.params['x1']['decoder']['encoder']['lmax_list'],
            x1_decoder_encoder_nodes.embedding.shape[-1],
            self.device,
            self.dtype,
        )
        x1_decoder_encoder_global.embedding = torch_scatter.scatter_sum(
            x1_decoder_encoder_nodes.embedding,
            input_dict['x1']['decoder']['batch'],
            dim = 0,
        )
        
        # store results in output_dict
        output_dict['x1']['decoder']['encoder']['edge_index'] = edge_index
        output_dict['x1']['decoder']['encoder']['node_embedding'] = x1_decoder_encoder_nodes
        output_dict['x1']['decoder']['encoder']['global_embedding'] = x1_decoder_encoder_global
        
        return output_dict
    
    
    
    
    def forward_x2_decoder_encoder(self, input_dict, output_dict):
        
        # initial node embeddings for the surface cloud
        
        x = SO3_Embedding(
            input_dict['x2']['decoder']['pos'].shape[0],
            self.params['x2']['decoder']['encoder']['lmax_list'],
            self.params['x2']['decoder']['encoder']['input_sphere_channels'],
            self.device,
            self.dtype,
        )
        x2_embedding = self.x2_decoder_encoder_embedding(input_dict['x2']['decoder']['x']) # this embeds one-hot representations of virtual vs real nodes
        
        x.embedding[:, 0, :] = x2_embedding
            
        
        # Adding time step encoding to l=0 node features 
            # we could also concatenate these as extra channels, but then we'd need to expand the l=1 channels as well. 
        
        x2_timestep = input_dict['x2']['decoder']['timestep'] # size (B,) where B is the number of molecules in the batch        
        x2_timestep_embedding = positional_encoding(x2_timestep, dim=self.params['x2']['decoder']['time_embedding_size'], device=self.device)
        x2_timestep_embedding = self.x2_decoder_local_timestep_embedding(x2_timestep_embedding)
        x2_timestep_embedding_pernode = x2_timestep_embedding[input_dict['x2']['decoder']['batch']]
        x.embedding[:, 0, :] = x.embedding[:, 0, :] + x2_timestep_embedding_pernode
        
        
        
        # 3D surface cloud convolution  (with EquiformerV2)
        
        edge_index = radius_graph(
            input_dict['x2']['decoder']['pos'],
            r = self.params['x2']['decoder']['encoder']['cutoff'],
            batch = input_dict['x2']['decoder']['batch'],
            max_num_neighbors = self.params['x2']['decoder']['encoder']['max_neighbors'] if 'max_neighbors' in self.params['x2']['decoder']['encoder'] else 1000000,
        )
        
        # True if VN, False otherwise
        virtual_node_mask = input_dict['x2']['decoder']['virtual_node_mask'] 
        
        if virtual_node_mask is not None:
            force_edges_to_virtual_nodes = self.params['x2']['decoder']['force_edges_to_virtual_nodes']
            if force_edges_to_virtual_nodes and (virtual_node_mask.any()):
                # if a graph instance has multiple VNs, this will introduce edges between those VNs
                    # this will remove self-loops on individual VNs
                edge_index = add_virtual_edges_to_edge_index(edge_index, virtual_node_mask, input_dict['x2']['decoder']['batch'])
        
        
        j, i = edge_index
        edge_distance_vec = input_dict['x2']['decoder']['pos'][j] - input_dict['x2']['decoder']['pos'][i]
        edge_distance = edge_distance_vec.norm(dim=-1)
        
        _, x2_decoder_encoder_nodes = self.x2_decoder_encoder(
            x, 
            input_dict['x2']['decoder']['pos'], 
            edge_index, 
            edge_distance, 
            edge_distance_vec,
            input_dict['x2']['decoder']['batch'],
        )
        
        x2_decoder_encoder_global = SO3_Embedding(
            max(input_dict['x2']['decoder']['batch']) + 1, # number of molecules in batch
            self.params['x2']['decoder']['encoder']['lmax_list'],
            x2_decoder_encoder_nodes.embedding.shape[-1],
            self.device,
            self.dtype,
        )
        x2_decoder_encoder_global.embedding = torch_scatter.scatter_sum(
            x2_decoder_encoder_nodes.embedding,
            input_dict['x2']['decoder']['batch'],
            dim = 0,
        )
        
            
        # store results in output_dict
        output_dict['x2']['decoder']['encoder']['edge_index'] = edge_index
        output_dict['x2']['decoder']['encoder']['node_embedding'] = x2_decoder_encoder_nodes
        output_dict['x2']['decoder']['encoder']['global_embedding'] = x2_decoder_encoder_global

        return output_dict
    
    
    
    
    def forward_x3_decoder_encoder(self, input_dict, output_dict):
        
        # initial node embeddings for the surface cloud
        
        x = SO3_Embedding(
            input_dict['x3']['decoder']['pos'].shape[0],
            self.params['x3']['decoder']['encoder']['lmax_list'],
            self.params['x3']['decoder']['encoder']['input_sphere_channels'],
            self.device,
            self.dtype,
        )
        x3_embedding = self.x3_decoder_scalar_expansion(input_dict['x3']['decoder']['x'])
        x3_embedding = self.x3_decoder_encoder_embedding(x3_embedding)
        virtual_node_mask = input_dict['x3']['decoder']['virtual_node_mask'] 
        if virtual_node_mask is not None:
            # zeroing-out x3_embedding for virtual nodes (which have no electrostatic potential)
            mask = torch.ones(x3_embedding.shape[0], device = self.device)
            mask[virtual_node_mask] = 0.0
            x3_embedding = x3_embedding * mask[:, None]
        x.embedding[:, 0, :] = x3_embedding
        
        
        # Adding time step encoding to l=0 node features 
            # we could also concatenate these as extra channels, but then we'd need to expand the l=1 channels as well. 
        
        x3_timestep = input_dict['x3']['decoder']['timestep'] # size (B,) where B is the number of molecules in the batch        
        x3_timestep_embedding = positional_encoding(x3_timestep, dim=self.params['x3']['decoder']['time_embedding_size'], device=self.device)
        x3_timestep_embedding = self.x3_decoder_local_timestep_embedding(x3_timestep_embedding)
        x3_timestep_embedding_pernode = x3_timestep_embedding[input_dict['x3']['decoder']['batch']]
        x.embedding[:, 0, :] = x.embedding[:, 0, :] + x3_timestep_embedding_pernode
        
        
        # 3D surface cloud convolution (with EquiformerV2)
        
        edge_index = radius_graph(
            input_dict['x3']['decoder']['pos'],
            r = self.params['x3']['decoder']['encoder']['cutoff'],
            batch = input_dict['x3']['decoder']['batch'],
            max_num_neighbors = self.params['x3']['decoder']['encoder']['max_neighbors'] if 'max_neighbors' in self.params['x3']['decoder']['encoder'] else 1000000,
        )
        
        # True if VN, False otherwise
        virtual_node_mask = input_dict['x3']['decoder']['virtual_node_mask']
        
        if virtual_node_mask is not None:
            force_edges_to_virtual_nodes = self.params['x3']['decoder']['force_edges_to_virtual_nodes']
            if force_edges_to_virtual_nodes and (virtual_node_mask.any()):
                # if a graph instance has multiple VNs, this will introduce edges between those VNs
                    # this will remove self-loops on individual VNs
                edge_index = add_virtual_edges_to_edge_index(edge_index, virtual_node_mask, input_dict['x3']['decoder']['batch'])
        
        
        j, i = edge_index
        edge_distance_vec = input_dict['x3']['decoder']['pos'][j] - input_dict['x3']['decoder']['pos'][i]
        edge_distance = edge_distance_vec.norm(dim=-1)
        
        x3_decoder_encoder_nodes, _ = self.x3_decoder_encoder(
            x, 
            input_dict['x3']['decoder']['pos'], 
            edge_index, 
            edge_distance, 
            edge_distance_vec,
            input_dict['x3']['decoder']['batch'],
        )
        
        x3_decoder_encoder_global = SO3_Embedding(
            max(input_dict['x3']['decoder']['batch']) + 1, # number of molecules in batch
            self.params['x3']['decoder']['encoder']['lmax_list'],
            x3_decoder_encoder_nodes.embedding.shape[-1],
            self.device,
            self.dtype,
        )
        x3_decoder_encoder_global.embedding = torch_scatter.scatter_sum(
            x3_decoder_encoder_nodes.embedding,
            input_dict['x3']['decoder']['batch'],
            dim = 0,
        )
        
        
        # store results in output_dict
        output_dict['x3']['decoder']['encoder']['edge_index'] = edge_index
        output_dict['x3']['decoder']['encoder']['node_embedding'] = x3_decoder_encoder_nodes
        output_dict['x3']['decoder']['encoder']['global_embedding'] = x3_decoder_encoder_global
        
        
        return output_dict
    

    
    def forward_x4_decoder_encoder(self, input_dict, output_dict):
        # initial node embeddings for the graph
        
        x = SO3_Embedding(
            input_dict['x4']['decoder']['pos'].shape[0],
            self.params['x4']['decoder']['encoder']['lmax_list'],
            self.params['x4']['decoder']['encoder']['input_sphere_channels'],
            self.device,
            self.dtype,
        )
        x.embedding[:, 0, :] = self.x4_decoder_encoder_embedding(input_dict['x4']['decoder']['x'])
        
        # insert vector directions as l=1 features
        x.embedding[:, 1:4, :] = input_dict['x4']['decoder']['direction'][..., None]
        
        # further embedding of l=0, l=1 input features
        x = self.x4_decoder_encoder_embedding_l1(x) # FeedForward 
        
        
        # Adding time step encoding to l=0 node features 
            # we could also concatenate these as extra channels, but then we'd need to expand the l=1 channels as well. 
        
        x4_timestep = input_dict['x4']['decoder']['timestep'] # size (B,) where B is the number of molecules in the batch        
        x4_timestep_embedding = positional_encoding(x4_timestep, dim=self.params['x4']['decoder']['time_embedding_size'], device=self.device)
        x4_timestep_embedding = self.x4_decoder_local_timestep_embedding(x4_timestep_embedding)
        x4_timestep_embedding_pernode = x4_timestep_embedding[input_dict['x4']['decoder']['batch']]
        x.embedding[:, 0, :] = x.embedding[:, 0, :] + x4_timestep_embedding_pernode

        
        # 3D graph convolution (with EquiformerV2)
        
        edge_index = radius_graph(
            input_dict['x4']['decoder']['pos'],
            r = self.params['x4']['decoder']['encoder']['cutoff'],
            batch = input_dict['x4']['decoder']['batch'],
            max_num_neighbors = self.params['x4']['decoder']['encoder']['max_neighbors'] if 'max_neighbors' in self.params['x4']['decoder']['encoder'] else 1000000,
        )
        
        # True if VN, False otherwise
        virtual_node_mask = input_dict['x4']['decoder']['virtual_node_mask'] 
        
        if virtual_node_mask is not None:
            force_edges_to_virtual_nodes = self.params['x4']['decoder']['force_edges_to_virtual_nodes']
            if force_edges_to_virtual_nodes and (virtual_node_mask.any()):
                # if a graph instance has multiple VNs, this will introduce edges between those VNs
                    # this will remove self-loops on individual VNs
                edge_index = add_virtual_edges_to_edge_index(edge_index, virtual_node_mask, input_dict['x4']['decoder']['batch'])
        
        
        j, i = edge_index
        edge_distance_vec = input_dict['x4']['decoder']['pos'][j] - input_dict['x4']['decoder']['pos'][i]
        edge_distance = edge_distance_vec.norm(dim=-1)
        
        x4_decoder_encoder_nodes, _ = self.x4_decoder_encoder(
            x, 
            input_dict['x4']['decoder']['pos'], 
            edge_index, 
            edge_distance, 
            edge_distance_vec,
            input_dict['x4']['decoder']['batch'],
        )
        
        x4_decoder_encoder_global = SO3_Embedding(
            max(input_dict['x4']['decoder']['batch']) + 1, # number of molecules in batch
            self.params['x4']['decoder']['encoder']['lmax_list'],
            x4_decoder_encoder_nodes.embedding.shape[-1],
            self.device,
            self.dtype,
        )
        x4_decoder_encoder_global.embedding = torch_scatter.scatter_sum(
            x4_decoder_encoder_nodes.embedding,
            input_dict['x4']['decoder']['batch'],
            dim = 0,
        )
        
        # store results in output_dict
        output_dict['x4']['decoder']['encoder']['edge_index'] = edge_index
        output_dict['x4']['decoder']['encoder']['node_embedding'] = x4_decoder_encoder_nodes
        output_dict['x4']['decoder']['encoder']['global_embedding'] = x4_decoder_encoder_global
        
        return output_dict
    
    
    
    def forward_decoder_joint_heterogeneous_graph_encoder(self, input_dict, output_dict):
        
        heterogeneous_variables = deepcopy([x_ for x_ in self.explicit_diffusion_variables if x_ not in self.exclude_variables_from_decoder_heterogeneous_graph])
        
        hetero_pos = torch.cat(
            [input_dict[x_]['decoder']['pos'] for x_ in heterogeneous_variables]
        , dim = 0)
        hetero_virtual_node_mask = torch.cat(
            [input_dict[x_]['decoder']['virtual_node_mask'] for x_ in heterogeneous_variables],
            dim = 0)
        hetero_batch = torch.cat(
            [input_dict[x_]['decoder']['batch'] for x_ in heterogeneous_variables]
            , dim = 0)
        hetero_x_identifier = torch.cat(
            [torch.ones_like(input_dict[x_]['decoder']['batch']) * i for i, x_ in enumerate(heterogeneous_variables)]
            , dim = 0)
        
        
        hetero_node_embeddings = SO3_Embedding(
                    hetero_batch.shape[0],
                    self.params['lmax_list'],
                    self.params['decoder_heterogeneous_graph_encoder']['sphere_channels'],
                    self.device,
                    self.dtype,
        )
        hetero_node_embeddings.embedding = torch.cat(
            [output_dict[x_]['decoder']['encoder']['node_embedding'].embedding for x_ in heterogeneous_variables]
        , dim = 0)
        
        
        # creating new edge index for heteregeneous graph, adding new edges for heterogeneous nodes within cut-off radius
        argsorted_batch = torch.argsort(hetero_batch)
        hetero_edge_index = radius_graph(
            hetero_pos[argsorted_batch],
            r = self.params['decoder_heterogeneous_graph_encoder']['cutoff'],
            batch = hetero_batch[argsorted_batch],
            max_num_neighbors = 1000000,
        )
        hetero_edge_index = remap_values(
            (torch.arange(len(hetero_batch), device = argsorted_batch.device), argsorted_batch), 
            hetero_edge_index,
        )
        hetero_edge_index = torch_geometric.utils.sort_edge_index(hetero_edge_index)

        # removing intra-x edges (except for intra-x1 edges), mainly to increase speed
        assert 'x1' in heterogeneous_variables # we'll have to change this code if we don't want to explicitly diffuse over x1
        edge_index_mask = hetero_x_identifier[hetero_edge_index]
        x1_edges = hetero_x_identifier[hetero_edge_index] == heterogeneous_variables.index('x1')
        edge_index_mask = (edge_index_mask[0] != edge_index_mask[1]) | (x1_edges[0] == x1_edges[1])
        hetero_edge_index = hetero_edge_index[:, edge_index_mask]
        
        # removing any edges to or from a virtual node
        edge_index_mask = hetero_virtual_node_mask[hetero_edge_index]
        edge_index_mask = edge_index_mask[0] | edge_index_mask[1]
        hetero_edge_index = hetero_edge_index[:, ~edge_index_mask]
        
        j, i = hetero_edge_index
        hetero_edge_distance_vec = hetero_pos[j] - hetero_pos[i]
        hetero_edge_distance = hetero_edge_distance_vec.norm(dim=-1) + 1e-6

        
        hetero_node_embeddings, _ = self.decoder_joint_heterogeneous_graph_encoder(
            hetero_node_embeddings, 
            hetero_pos, 
            hetero_edge_index, 
            hetero_edge_distance, 
            hetero_edge_distance_vec,
            hetero_batch,
        )
        
        
        for i, x_ in enumerate(heterogeneous_variables):
            # residual connection to heterogeneous node embeddings
            
            x_node_embedding = output_dict[x_]['decoder']['encoder']['node_embedding']
            x_node_embedding.embedding = x_node_embedding.embedding + hetero_node_embeddings.embedding[hetero_x_identifier == i, ...]
            
            output_dict[x_]['decoder']['encoder']['node_embedding'] = x_node_embedding
            
            # also updating the global embeddings
            output_dict[x_]['decoder']['encoder']['global_embedding'].embedding = torch_scatter.scatter_sum(
                x_node_embedding.embedding,
                input_dict[x_]['decoder']['batch'],
                dim = 0,
            )
        
        return output_dict
    
    
    
    
    def forward_decoder_joint_processing(self, x_str, input_dict, output_dict):

        assert x_str in self.explicit_diffusion_variables
        
        x = output_dict[x_str]['decoder']['encoder']['node_embedding']
        batch_size = output_dict[x_str]['decoder']['encoder']['global_embedding'].embedding.shape[0]
        
            
        # Obtaining joint l=1 global embeddings from all explicit decoders
        
        joint_l1_embedding = SO3_Embedding(
            batch_size,
            self.params['lmax_list'],
            sum([output_dict[x_]['decoder']['encoder']['global_embedding'].embedding.shape[-1] for x_ in self.explicit_diffusion_variables]),
            self.device,
            self.dtype,
        )
        joint_l1_embeddings_ = []
        for x_ in self.explicit_diffusion_variables:
            embedding = output_dict[x_]['decoder']['encoder']['global_embedding'].embedding[:, 1:4, :]
            joint_l1_embeddings_.append(embedding)
        joint_l1_embedding.embedding[:,1:4,:] = torch.cat(joint_l1_embeddings_, dim = -1) # (B, 3, channels)
        
        
        # l=0 features are all zero --> FeedForwardNetwork doesn't mix between l orders, so l=0 features will be zero still.
        if x_str == 'x1':
            joint_l1_embedding = self.x1_decoder_global_l1_embedding(joint_l1_embedding)
        if x_str == 'x2':
            joint_l1_embedding = self.x2_decoder_global_l1_embedding(joint_l1_embedding)
        if x_str == 'x3':
            joint_l1_embedding = self.x3_decoder_global_l1_embedding(joint_l1_embedding)
        if x_str == 'x4':
            joint_l1_embedding = self.x4_decoder_global_l1_embedding(joint_l1_embedding)
        
        
        
        # Obtaining l=0 time-step embeddings

        concat_timestep_embedding = torch.cat([
            positional_encoding(
                    input_dict[x_]['decoder']['timestep'], 
                    dim = self.params[x_]['decoder']['time_embedding_size'], 
                    device = self.device,
                )
            for x_ in self.explicit_diffusion_variables], dim = -1)
        
        if x_str == 'x1':
            global_timestep_embedding = self.x1_decoder_global_timestep_embedding(concat_timestep_embedding)
        if x_str == 'x2':
            global_timestep_embedding = self.x2_decoder_global_timestep_embedding(concat_timestep_embedding)
        if x_str == 'x3':
            global_timestep_embedding = self.x3_decoder_global_timestep_embedding(concat_timestep_embedding)
        if x_str == 'x4':
            global_timestep_embedding = self.x4_decoder_global_timestep_embedding(concat_timestep_embedding)
        
        
        # Aggregating all global features, mixing their l-channels, and applying them at once in a residual update to the node embeddings
        
        # aggregating updates
        joint_embedding_update = SO3_Embedding(
            batch_size,
            self.params['lmax_list'],
            output_dict[x_str]['decoder']['encoder']['global_embedding'].embedding.shape[-1],
            self.device,
            self.dtype,
        )
        joint_embedding_update.embedding[:, 1:4, :] = joint_l1_embedding.embedding[:, 1:4, :]
        joint_embedding_update.embedding[:, 0, :] = global_timestep_embedding
        
        
        # (learnable) mixing of l=0 and l=1 channels of joint_embedding_update with tensor products (operation from equiformer, not equiformerv2)
        joint_embedding_update_e3nn = convert_equiformerv2_to_e3nn(joint_embedding_update.embedding[:, 0:4, :], lmax=1)
        if x_str == 'x1':
            joint_embedding_update_e3nn = self.x1_decoder_equiformer_tensor_product(joint_embedding_update_e3nn, joint_embedding_update_e3nn)
        if x_str == 'x2':
            joint_embedding_update_e3nn = self.x2_decoder_equiformer_tensor_product(joint_embedding_update_e3nn, joint_embedding_update_e3nn)
        if x_str == 'x3':
            joint_embedding_update_e3nn = self.x3_decoder_equiformer_tensor_product(joint_embedding_update_e3nn, joint_embedding_update_e3nn)
        if x_str == 'x4':
            joint_embedding_update_e3nn = self.x4_decoder_equiformer_tensor_product(joint_embedding_update_e3nn, joint_embedding_update_e3nn)
        
        joint_embedding_update.embedding[:, 0:4, :] = convert_e3nn_to_equiformerv2(
            joint_embedding_update_e3nn, 
            lmax = 1, 
            num_channels = joint_embedding_update.embedding.shape[-1],
        )
        
        # residually updating node embeddings with mixed global joint embeddings
        x.embedding[:, 0:4, :] = x.embedding[:, 0:4, :] + joint_embedding_update.embedding[input_dict[x_str]['decoder']['batch'], 0:4, :]
        
        
        output_dict[x_str]['decoder'][f'node_joint_embedding'] = x
        
        return output_dict
    
    
    
    
    def forward_x1_decoder_denoiser(self, input_dict, output_dict):
        
        x1 = output_dict['x1']['decoder']['node_joint_embedding']        
        x1_positions = input_dict['x1']['decoder']['pos']
        
        virtual_node_mask = input_dict['x1']['decoder']['virtual_node_mask'] 

        # atom type update
        x1_features = x1.embedding[:,0,:]  # only l=0 features
        x1_features_update = self.x1_decoder_denoiser_MLP(x1_features)
        
        
        # bond type update
        x1_bond_features_update = None
        if self.x1_bond_diffusion:
            x1_features = x1.embedding[:,0,:]  # only l=0 features
            # this bond_edge_index includes only 1 directed edge per bond
            bond_edge_index = input_dict['x1']['decoder']['bond_edge_index']
            x1_bond_features = input_dict['x1']['decoder']['bond_edge_x']
            
            # get distance expansion of pairwise distances between nodes
            x1_bond_distance_expansion = x1_positions[bond_edge_index[0]] - x1_positions[bond_edge_index[1]]
            x1_bond_distance_expansion = x1_bond_distance_expansion.norm(dim = -1, keepdim = True)
            x1_bond_distance_expansion = self.x1_decoder_denoiser_bond_distance_scalar_expansion(x1_bond_distance_expansion)
            
            x1_bond_features_update_01 = self.x1_decoder_denoiser_bond_MLP(
                torch.cat([x1_bond_features, x1_bond_distance_expansion, x1_features[bond_edge_index[0]], x1_features[bond_edge_index[1]]], dim = 1)
            )
            x1_bond_features_update_10 = self.x1_decoder_denoiser_bond_MLP(
                torch.cat([x1_bond_features, x1_bond_distance_expansion, x1_features[bond_edge_index[1]], x1_features[bond_edge_index[0]]], dim = 1)
            )
            x1_bond_features_update = (x1_bond_features_update_01 + x1_bond_features_update_10) / 2.0 # symmetrical update
            
        
   
        # denoising steps for node coordinates
        
        # re-using edge_index from radius graph of the structure encoder
            # already forces edges between every node and the virtual nodes
        edge_index = output_dict['x1']['decoder']['encoder']['edge_index'] 

        
        x1_positions_update = torch.zeros_like(x1_positions)
        if self.params['x1']['decoder']['denoiser']['use_e3nn']:
            x1_e3nn_update = self.x1_decoder_denoiser_E3NN(x1)
            x1_positions_update_e3nn = x1_e3nn_update.embedding[:, 1:4, :].squeeze(dim=2) # (B,3) l=1 outputs
            
            # need to apply VN mask here
            if virtual_node_mask is not None:
                x1_positions_update_e3nn[virtual_node_mask] = 0.0
                
            x1_positions = x1_positions + x1_positions_update_e3nn
            x1_positions_update = x1_positions_update + x1_positions_update_e3nn
  
        if self.params['x1']['decoder']['denoiser']['use_egnn_positions_update']:
            _, x1_positions_update_egnn = self.x1_decoder_denoiser_EGNN(
                x = x1_features,
                pos = x1_positions, 
                edge_index = edge_index,
                batch = input_dict['x1']['decoder']['batch'], 
                edge_attr = None, 
                pos_update_mask = None, # mask applied separately below
                residual_pos_update = False,
                residual_x_update = False,
            )
            
            # need to apply VN mask here
            if virtual_node_mask is not None:
                x1_positions_update_egnn[virtual_node_mask] = 0.0
                
            x1_positions = x1_positions + x1_positions_update_egnn
            x1_positions_update = x1_positions_update + x1_positions_update_egnn
            
        # can we use an bond distance/angle validity loss on top of x1_positions?
            
        # store results in output_dict
        output_dict['x1']['decoder']['denoiser']['x_out'] = x1_features_update # continuous for now
        output_dict['x1']['decoder']['denoiser']['pos_out'] = x1_positions_update # these are "delta" positions (e.g., predicted noise, not a predicted structure)
        output_dict['x1']['decoder']['denoiser']['bond_edge_x_out'] = x1_bond_features_update # continuous for now
                
        return output_dict
    
    
    
    def forward_x2_decoder_denoiser(self, input_dict, output_dict):
        
        x2 = output_dict['x2']['decoder']['node_joint_embedding']    
        x2_positions = input_dict['x2']['decoder']['pos']
        
        virtual_node_mask = input_dict['x2']['decoder']['virtual_node_mask'] 
        
        
        # denoising steps for coordinates (e.g., with E3NN/EGNN)
        
        # re-using edge_index from radius graph of the structure encoder
            # already forces edges between every node and the virtual nodes
        edge_index = output_dict['x2']['decoder']['encoder']['edge_index'] 

        x2_positions_update = torch.zeros_like(x2_positions)
        if self.params['x2']['decoder']['denoiser']['use_e3nn']:
            x2_e3nn_update = self.x2_decoder_denoiser_E3NN(x2)
            x2_positions_update_e3nn = x2_e3nn_update.embedding[:, 1:4, :].squeeze(dim=2) # (B,3) l=1 outputs
            
            # need to apply VN mask here
            if virtual_node_mask is not None:
                x2_positions_update_e3nn[virtual_node_mask] = 0.0
                
            x2_positions = x2_positions + x2_positions_update_e3nn
            x2_positions_update = x2_positions_update + x2_positions_update_e3nn
        
        if self.params['x2']['decoder']['denoiser']['use_egnn_positions_update']:
            x2_features = x2.embedding[:,0,:]
            
            _, x2_positions_update_egnn = self.x2_decoder_denoiser_EGNN(
                x = x2_features,  # only l=0 features
                pos = x2_positions, 
                edge_index = edge_index,
                batch = input_dict['x2']['decoder']['batch'], 
                edge_attr = None, 
                pos_update_mask = None, # mask applied separately below
                residual_pos_update = False,
                residual_x_update = False,
            )
            
            # need to apply VN mask here
            if virtual_node_mask is not None:
                x2_positions_update_egnn[virtual_node_mask] = 0.0
                
            x2_positions = x2_positions + x2_positions_update_egnn
            x2_positions_update = x2_positions_update + x2_positions_update_egnn
                    
        # store results in output_dict
        output_dict['x2']['decoder']['denoiser']['pos_out'] = x2_positions_update # these are "delta" positions (e.g., predicted noise, not a predicted structure)
        
        return output_dict
    
    
    
    def forward_x3_decoder_denoiser(self, input_dict, output_dict):
        
        x3 = output_dict['x3']['decoder']['node_joint_embedding']
        virtual_node_mask = input_dict['x3']['decoder']['virtual_node_mask']
            
        x3_features_update = self.x3_decoder_denoiser_MLP(x3.embedding[:,0,:])
        if virtual_node_mask is not None:
            x3_features_update[virtual_node_mask] = 0.0
        output_dict['x3']['decoder']['denoiser']['x_out'] = x3_features_update

        
        x3_positions = input_dict['x3']['decoder']['pos']
        
        # denoising steps for coordinates (e.g., with E3NN/EGNN)
        
        # re-using edge_index from radius graph of the structure encoder
            # already forces edges between every node and the virtual nodes
        edge_index = output_dict['x3']['decoder']['encoder']['edge_index'] 
        
        x3_positions_update = torch.zeros_like(x3_positions)
        if self.params['x3']['decoder']['denoiser']['use_e3nn']:
            x3_e3nn_update = self.x3_decoder_denoiser_E3NN(x3)
            x3_positions_update_e3nn = x3_e3nn_update.embedding[:, 1:4, :].squeeze(dim=2) # (B,3) l=1 outputs
            
            # need to apply VN mask here
            if virtual_node_mask is not None:
                x3_positions_update_e3nn[virtual_node_mask] = 0.0
                
            x3_positions = x3_positions + x3_positions_update_e3nn
            x3_positions_update = x3_positions_update + x3_positions_update_e3nn
        
        if self.params['x3']['decoder']['denoiser']['use_egnn_positions_update']:
            x3_features = x3.embedding[:,0,:]
        
            _, x3_positions_update_egnn = self.x3_decoder_denoiser_EGNN(
                x = x3_features,  # only l=0 features
                pos = x3_positions, 
                edge_index = edge_index,
                batch = input_dict['x3']['decoder']['batch'], 
                edge_attr = None, 
                pos_update_mask = None, # mask applied separately below
                residual_pos_update = False,
                residual_x_update = False,
            )
        
            # need to apply VN mask here
            if virtual_node_mask is not None:
                x3_positions_update_egnn[virtual_node_mask] = 0.0
                
            x3_positions = x3_positions + x3_positions_update_egnn
            x3_positions_update = x3_positions_update + x3_positions_update_egnn
                    
        # store results in output_dict
        output_dict['x3']['decoder']['denoiser']['pos_out'] = x3_positions_update # these are "delta" positions (e.g., predicted noise, not a predicted structure)
        
        return output_dict
    
    
    
    def forward_x4_decoder_denoiser(self, input_dict, output_dict):
        
        x4 = output_dict['x4']['decoder']['node_joint_embedding']        
        x4_positions = input_dict['x4']['decoder']['pos']
        x4_directions = input_dict['x4']['decoder']['direction']
        
        virtual_node_mask = input_dict['x4']['decoder']['virtual_node_mask']

        
        x4_features = x4.embedding[:,0,:]  # only l=0 features
        x4_features_update = self.x4_decoder_denoiser_MLP(x4_features)        

        
        # denoising steps for node directions and coordinates
        
        x4_e3nn_direction_update = self.x4_decoder_denoiser_E3NN_direction(x4).embedding[:, 1:4, :].squeeze(dim=2) # (B,3) l=1 outputs
        if virtual_node_mask is not None:
            x4_e3nn_direction_update[virtual_node_mask] = 0.0
        
        
        # re-using edge_index from radius graph of the structure encoder
            # already forces edges between every node and the virtual nodes
        edge_index = output_dict['x4']['decoder']['encoder']['edge_index'] 
        
        x4_positions_update = torch.zeros_like(x4_positions)
        if self.params['x4']['decoder']['denoiser']['use_e3nn']:
            x4_e3nn_update = self.x4_decoder_denoiser_E3NN(x4)
            x4_positions_update_e3nn = x4_e3nn_update.embedding[:, 1:4, :].squeeze(dim=2) # (B,3) l=1 outputs
            
            # need to apply VN mask here
            if virtual_node_mask is not None:
                x4_positions_update_e3nn[virtual_node_mask] = 0.0
                
            x4_positions = x4_positions + x4_positions_update_e3nn
            x4_positions_update = x4_positions_update + x4_positions_update_e3nn
  
        if self.params['x4']['decoder']['denoiser']['use_egnn_positions_update']:
            _, x4_positions_update_egnn = self.x4_decoder_denoiser_EGNN(
                x = x4_features,
                pos = x4_positions, 
                edge_index = edge_index,
                batch = input_dict['x4']['decoder']['batch'], 
                edge_attr = None, 
                pos_update_mask = None, # mask applied separately below
                residual_pos_update = False,
                residual_x_update = False,
            )
            
            # need to apply VN mask here
            if virtual_node_mask is not None:
                x4_positions_update_egnn[virtual_node_mask] = 0.0
                
            x4_positions = x4_positions + x4_positions_update_egnn
            x4_positions_update = x4_positions_update + x4_positions_update_egnn
        

        # store results in output_dict
        output_dict['x4']['decoder']['denoiser']['x_out'] = x4_features_update # continuous for now
        output_dict['x4']['decoder']['denoiser']['pos_out'] = x4_positions_update # these are "delta" positions (e.g., predicted noise, not a predicted structure)
        output_dict['x4']['decoder']['denoiser']['direction_out'] = x4_e3nn_direction_update # these are "delta" directions (e.g., predicted noise, not a predicted structure)
                
        return output_dict
    
    
    

    
    # forward function for training
        # this training function could also be split into separate diffusion processes
            # nothing REQUIRES us to train on all diffusion branches in the same batch ...
    def forward(self, input_dict):
        
        self.device = input_dict['device']
        self.dtype = input_dict['dtype']
        
        # placeholder to define the organization of this dictionary
        output_dict = {
            'x1': {

                'decoder': {
                    
                    'encoder': {
                        'node_embedding': None,
                        'global_embedding': None,
                        'edge_index': None,
                    },
                    
                    'node_joint_embedding': None,
                    
                    'denoiser': {
                        'x_out': None,
                        'pos_out': None,
                    },
                
                },
            },
            
            
            
            'x2': {
                
                'decoder': {
                    'encoder': {
                        'node_embedding': None,
                        'global_embedding': None,
                        'edge_index': None,
                    },
                    
                    'node_joint_embedding': None,
                    
                    'denoiser': {
                        'pos_out': None,
                    },
                    
                },
            },
            
            
            
            'x3': {

                'decoder': {
                    'encoder': {
                        'node_embedding': None,
                        'global_embedding': None,
                        'edge_index': None,
                    },
                    
                    'node_joint_embedding': None,
                    
                    'denoiser': {
                        'x_out': None,
                        'pos_out': None, 
                    },
                    
                },
            },
        
            
            'x4': {

                'decoder': {
                    'encoder': {
                        'node_embedding': None,
                        'global_embedding': None,
                        'edge_index': None,
                    },
                    
                    'node_joint_embedding': None,
                    
                    'denoiser': {
                        'x_out': None,
                        'pos_out': None,
                        'direction_out': None,
                    },
                    
                },
            },
            
        }
        
        # Embedding Modules
        if 'x1' in self.explicit_diffusion_variables:
            output_dict = self.forward_x1_decoder_encoder(input_dict, output_dict)
        if 'x2' in self.explicit_diffusion_variables:
            output_dict = self.forward_x2_decoder_encoder(input_dict, output_dict)
        if 'x3' in self.explicit_diffusion_variables:
            output_dict = self.forward_x3_decoder_encoder(input_dict, output_dict)
        if 'x4' in self.explicit_diffusion_variables:
            output_dict = self.forward_x4_decoder_encoder(input_dict, output_dict)
        
        # Joint Module
            # - pass local messages within the heterogeneous graph of the explicit diffusion (decoder) variables
            # - jointly process global codes
        if self.decoder_joint_heterogeneous_graph_encoder is not None:
            output_dict = self.forward_decoder_joint_heterogeneous_graph_encoder(input_dict, output_dict)
        
        if 'x1' in self.explicit_diffusion_variables:
            output_dict = self.forward_decoder_joint_processing('x1', input_dict, output_dict)
        if 'x2' in self.explicit_diffusion_variables:
            output_dict = self.forward_decoder_joint_processing('x2', input_dict, output_dict)
        if 'x3' in self.explicit_diffusion_variables:
            output_dict = self.forward_decoder_joint_processing('x3', input_dict, output_dict)
        if 'x4' in self.explicit_diffusion_variables:
            output_dict = self.forward_decoder_joint_processing('x4', input_dict, output_dict)
        
        # Denoising Modules
        if 'x1' in self.explicit_diffusion_variables:
            output_dict = self.forward_x1_decoder_denoiser(input_dict, output_dict)
        if 'x2' in self.explicit_diffusion_variables:
            output_dict = self.forward_x2_decoder_denoiser(input_dict, output_dict)
        if 'x3' in self.explicit_diffusion_variables:
            output_dict = self.forward_x3_decoder_denoiser(input_dict, output_dict)
        if 'x4' in self.explicit_diffusion_variables:
            output_dict = self.forward_x4_decoder_denoiser(input_dict, output_dict)
        
        return input_dict, output_dict
    