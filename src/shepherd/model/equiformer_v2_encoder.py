import logging
import time
import math
import numpy as np
import torch
import torch.nn as nn
from pyexpat.model import XML_CQUANT_OPT
import copy

from shepherd.model.equiformer_v2.ocpmodels.common.registry import registry
from shepherd.model.equiformer_v2.ocpmodels.common.utils import conditional_grad
from shepherd.model.equiformer_v2.ocpmodels.models.base import BaseModel
from shepherd.model.equiformer_v2.ocpmodels.models.scn.sampling import CalcSpherePoints
from shepherd.model.equiformer_v2.ocpmodels.models.scn.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)

try:
    from e3nn import o3
except ImportError:
    pass

from shepherd.model.equiformer_v2.nets.equiformer_v2.gaussian_rbf import GaussianRadialBasisLayer
from torch.nn import Linear
from shepherd.model.equiformer_v2.nets.equiformer_v2.edge_rot_mat import init_edge_rot_mat
from shepherd.model.equiformer_v2.nets.equiformer_v2.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2
)
from shepherd.model.equiformer_v2.nets.equiformer_v2.module_list import ModuleListInfo
from shepherd.model.equiformer_v2.nets.equiformer_v2.so2_ops import SO2_Convolution
from shepherd.model.equiformer_v2.nets.equiformer_v2.radial_function import RadialFunction
from shepherd.model.equiformer_v2.nets.equiformer_v2.layer_norm import (
    EquivariantLayerNormArray, 
    EquivariantLayerNormArraySphericalHarmonics, 
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer
)
from shepherd.model.equiformer_v2.nets.equiformer_v2.transformer_block import (
    SO2EquivariantGraphAttention,
    FeedForwardNetwork,
    TransBlockV2, 
)

# in addition to the custom classes defined in this file, I also made direct changes to equiformer_v2/nets/equiformer_v2/transformer_block.py to alter the node embedding strategy (to use linear layers instead of embedding layers)

# I've copied EdgeDegreeEmbedding into this file for enable easy changes to the node embedding strategy 
    # originally from equiformer_v2.nets.equiformer_v2.input_block import EdgeDegreeEmbedding 
    
# other than those changes, the original EquiformerV2 codebase hasn't been altered in any signficant way

class EdgeDegreeEmbedding(torch.nn.Module):
    """

    Args:
        input_sphere_channels (int): Number of input spherical channels (for nodes)
        sphere_channels (int):      Number of spherical channels
        
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        
        SO3_rotation (list:SO3_Rotation): Class to calculate Wigner-D matrices and rotate embeddings
        mappingReduced (CoefficientMappingModule): Class to convert l and m indices once node embedding is rotated
        
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                        The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features

        rescale_factor (float):     Rescale the sum aggregation
    """

    def __init__(
        self,
        input_sphere_channels,
        sphere_channels,
        
        lmax_list,
        mmax_list,
        
        SO3_rotation,
        mappingReduced,

        edge_channels_list,
        use_atom_edge_embedding,
        
        rescale_factor
    ):
        super(EdgeDegreeEmbedding, self).__init__()
        self.input_sphere_channels = input_sphere_channels
        self.sphere_channels = sphere_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(self.lmax_list)
        self.SO3_rotation = SO3_rotation
        self.mappingReduced = mappingReduced
        
        self.m_0_num_coefficients = self.mappingReduced.m_size[0] 
        self.m_all_num_coefficents = len(self.mappingReduced.l_harmonic)

        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        #self.max_num_elements = max_num_elements
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.use_atom_edge_embedding = use_atom_edge_embedding
        
        """
        if self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None
        """
        if self.use_atom_edge_embedding:
            self.source_embedding = nn.Linear(self.input_sphere_channels, self.edge_channels_list[-1])
            self.target_embedding = nn.Linear(self.input_sphere_channels, self.edge_channels_list[-1])
            nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None
        
        # Embedding function of distance
        self.edge_channels_list.append(self.m_0_num_coefficients * self.sphere_channels)
        self.rad_func = RadialFunction(self.edge_channels_list)

        self.rescale_factor = rescale_factor
    
    def forward(
        self,
        x_input,
        edge_distance, # these are edge FEATURES (not restricted to be a dim=1 float)
        edge_index,
    ):    
        
        """
        if self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            x_edge = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)
        else:
            x_edge = edge_distance
        """
        if self.use_atom_edge_embedding:
            source_embedding = x_input.embedding[edge_index[0], 0, :] # l=0 embeddings only
            target_embedding = x_input.embedding[edge_index[1], 0, :] # l=0 embeddings only
            source_embedding = self.source_embedding(source_embedding)
            target_embedding = self.target_embedding(target_embedding)
            x_edge = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)
        else:
            x_edge = edge_distance
            
        x_edge_m_0 = self.rad_func(x_edge)
        x_edge_m_0 = x_edge_m_0.reshape(-1, self.m_0_num_coefficients, self.sphere_channels)
        x_edge_m_pad = torch.zeros((
            x_edge_m_0.shape[0], 
            (self.m_all_num_coefficents - self.m_0_num_coefficients), 
            self.sphere_channels), 
            device=x_edge_m_0.device)
        x_edge_m_all = torch.cat((x_edge_m_0, x_edge_m_pad), dim=1)

        x_edge_embedding = SO3_Embedding(
            0, 
            self.lmax_list.copy(), 
            self.sphere_channels, 
            device=x_edge_m_all.device, 
            dtype=x_edge_m_all.dtype
        )
        x_edge_embedding.set_embedding(x_edge_m_all)
        x_edge_embedding.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # Reshape the spherical harmonics based on l (degree)
        x_edge_embedding._l_primary(self.mappingReduced)

        # Rotate back the irreps
        x_edge_embedding._rotate_inv(self.SO3_rotation, self.mappingReduced)

        # Compute the sum of the incoming neighboring messages for each target node
        #x_edge_embedding._reduce_edge(edge_index[1], atomic_numbers.shape[0])
        x_edge_embedding._reduce_edge(edge_index[1], x_input.embedding.shape[0])
        x_edge_embedding.embedding = x_edge_embedding.embedding / self.rescale_factor

        return x_edge_embedding



# Statistics of IS2RE 100K 
#_AVG_NUM_NODES  = 77.81317
#_AVG_DEGREE     = 23.395238876342773    # IS2RE: 100k, max_radius = 5, max_neighbors = 100


@registry.register_model("equiformer_v2")
class EquiformerV2(BaseModel):
    """
    Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon S2 activation

    Args:
        final_block_channels (int): Number of spherical channels in final x embedding; if 0, there is no final block.
        
        num_layers (int):             Number of layers in the GNN
        input_sphere_channels (int): Number of spherical channels in input x embedding, used for edge embedding
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])

        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        grid_resolution (int):        Resolution of SO3_Grid
        
        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks
        
        edge_channels (int):                Number of channels for the edge invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel share radial function weights
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances
        
        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs. 
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks

        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions
    """
    def __init__(
        self,
    
        final_block_channels = 0,
        
        num_layers=12,
        input_sphere_channels = 128,
        sphere_channels=128,
        attn_hidden_channels=128,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=512,
        
        norm_type='rms_norm_sh',
        
        lmax_list=[6],
        mmax_list=[2],
        grid_resolution=None, 
        cutoff = 5.0,
        num_sphere_samples=128,
        
        edge_attr_input_channels = 0, 
        
        edge_channels=128,
        use_atom_edge_embedding=True, 
        share_atom_edge_embedding=False,
        use_m_share_rad=False,
        distance_function="gaussian",
        num_distance_basis=512, 

        attn_activation='scaled_silu',
        use_s2_act_attn=False, 
        use_attn_renorm=True,
        ffn_activation='scaled_silu',
        use_gate_act=False,
        use_grid_mlp=False, 
        use_sep_s2_act=True,

        alpha_drop=0.0,
        drop_path_rate=0.0, 
        proj_drop=0.0, 

        weight_init='normal',
        
    ):
        super().__init__()
        
        # set to dataset parameters
        self._AVG_NUM_NODES = 18.03065905448718
        self._AVG_DEGREE = 15.57930850982666

        self.final_block_channels = final_block_channels
        
        self.num_layers = num_layers
        
        self.input_sphere_channels = input_sphere_channels
        self.sphere_channels = sphere_channels
        # for now, we must enforce input_sphere_channels == sphere_channels. This could be made more flexible by including a feed-forward SO3 network as a node-embedding step
        assert self.input_sphere_channels == self.sphere_channels
        
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type
        
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution
        self.cutoff = cutoff # radial cutoff
        
        self.num_sphere_samples = num_sphere_samples

        self.edge_attr_input_channels = edge_attr_input_channels
        
        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding 
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act
        
        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        self.weight_init = weight_init
        assert self.weight_init in ['normal', 'uniform']

        self.device = 'cpu' #torch.cuda.current_device()

        self.grad_forces = False
        self.num_resolutions = len(self.lmax_list)
        assert self.num_resolutions == 1
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        
        # Weights for message initialization
        #self.sphere_embedding = nn.Embedding(self.max_num_elements, self.sphere_channels_all) # replace
        
        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            'gaussian',
        ]
        if self.distance_function == 'gaussian':
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                100, # == self.distance_expansion.num_output
                2.0,
            )
            #self.distance_expansion = GaussianRadialBasisLayer(num_basis=self.num_distance_basis, cutoff=self.max_radius)
        else:
            raise ValueError
        
        self.edge_attr_embedding = None
        if self.edge_attr_input_channels > 0:
            self.edge_attr_embedding = torch.nn.Linear(self.edge_attr_input_channels, int(self.distance_expansion.num_output))
        
        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [self.edge_channels] * 2

        # Initialize atom edge embedding
        """
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None
        """
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Linear(self.input_sphere_channels, self.edge_channels_list[-1])
            self.target_embedding = nn.Linear(self.input_sphere_channels, self.edge_channels_list[-1])
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None
        
        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo('({}, {})'.format(max(self.lmax_list), max(self.lmax_list)))
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, 
                        m, 
                        resolution=self.grid_resolution, 
                        normalization='component'
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.input_sphere_channels,
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=self._AVG_DEGREE
        )

        # Initialize the blocks for each layer of EquiformerV2
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV2(
                self.input_sphere_channels,
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels, 
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                #self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop, 
                self.drop_path_rate,
                self.proj_drop
            )
            self.blocks.append(block)
        
        self.norm = get_normalization_layer(self.norm_type, lmax=max(self.lmax_list), num_channels=self.sphere_channels)
        
        if self.final_block_channels > 0:
            self.final_block = FeedForwardNetwork(
                self.sphere_channels,
                self.ffn_hidden_channels, 
                self.final_block_channels,
                self.lmax_list,
                self.mmax_list,
                self.SO3_grid,  
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act
            )
        else:
            self.final_block = None
        
        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)


    #@conditional_grad(torch.enable_grad()) # relevant if EquiformerV2 is used to regress forces (which we're not)
    def forward(self, x_input, pos, edge_index, edge_distance, edge_distance_vec, batch, edge_attr = None):
        
        self.dtype = pos.dtype
        self.device = pos.device
        
        #atomic_numbers = data.atomic_numbers.long()        
        
        ###############################################################
        # Initialize data structures
        ###############################################################
        
        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)
        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

            
        ###############################################################
        # Initialize node embeddings
        ###############################################################
        
        # interpret x_input as a minimal embedding of one-hot atomic numbers
        # do we want to include an extra FeedForwardNetwork mapping x (input_sphere_channels) to (sphere_channels) ?
        x = x_input
        
        
        ###############################################################
        # Initialize edge embeddings
        ###############################################################
        
        # Edge encoding (distance and atom edge)
        edge_distance = self.distance_expansion(edge_distance)
        if edge_attr is not None:
            assert self.edge_attr_embedding is not None
            edge_distance = edge_distance + self.edge_attr_embedding(edge_attr)
        
        
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_embedding = x_input.embedding[edge_index[0], 0, :] # l=0 embeddings only
            target_embedding = x_input.embedding[edge_index[1], 0, :] # l=0 embeddings only
            source_embedding = self.source_embedding(source_embedding) # replace
            target_embedding = self.target_embedding(target_embedding) # replace
            edge_distance = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)
        
        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            x_input,
            edge_distance,
            edge_index)
        
        # x.embedding only has non-zero l=0 features; all others are zero (until added to edge_degree.embedding)
        x.embedding = x.embedding + edge_degree.embedding
        
        # x.embedding has shape (num_nodes, (l_max+1)^2, num_channels)
        
        
        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            x = self.blocks[i](
                x,                  # SO3_Embedding
                x_input,
                edge_distance, 
                edge_index, 
                batch=batch    # for GraphDropPath
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)
        
        
        if self.final_block is not None:
            x_final = self.final_block(x)
        else:
            x_final = x
        
        """
        node_out = self.output_block(x)
        node_scalar_out = node_out.embedding.narrow(1, 0, 1)
        node_vector_out = node_out.embedding.narrow(1, 1, 3)
        node_vector_out = node_vector_out.view(-1, 3)
        """
        """
        print('node_out')
        print(f'    {node_scalar_out.shape}')
        print(f'    {node_vector_out.shape}')
        """
        
        return x, x_final
    

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


    def _init_weights(self, m):
        if (isinstance(m, torch.nn.Linear)
            or isinstance(m, SO3_LinearV2)
        ):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == 'normal':
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    
    def _uniform_init_rad_func_linear_weights(self, m):
        if (isinstance(m, RadialFunction)):
            m.apply(self._uniform_init_linear_weights)


    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, SO3_LinearV2)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormArray)
                or isinstance(module, EquivariantLayerNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonicsV2)
                or isinstance(module, GaussianRadialBasisLayer)):
                for parameter_name, _ in module.named_parameters():
                    if (isinstance(module, torch.nn.Linear)
                        or isinstance(module, SO3_LinearV2)
                    ):
                        if 'weight' in parameter_name:
                            continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)
