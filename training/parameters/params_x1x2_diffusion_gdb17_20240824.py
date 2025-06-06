# x3 implicitly includes x2

import numpy as np

diffuse_formal_charges = True # model will diffuse the formal charge on the atoms in addition to their element type
charge_types = [0, 1, 2, -1, -2] # dataset must be limited to these formal charges
num_charge_types = int(diffuse_formal_charges) * len(charge_types)

diffuse_bonds = True
bond_types = [None, 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
num_bond_types = len(bond_types)

atom_types = [None, 'H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P', 'Si']
num_atom_types = len(atom_types)

num_pharmacophore_types = 10 # ('Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'Halogen', 'Cation', 'Anion', 'ZnBinder') plus buffers

num_channels = 64


params = {
    'data': 'GDB17',
    
    # major architecture decisions
    
    'use_ema': False,
    'x1_bond_diffusion': diffuse_bonds,
    'x1_formal_charge_diffusion': diffuse_formal_charges,

    'explicit_diffusion_variables': ['x1', 'x2'],
        
    'exclude_variables_from_decoder_heterogeneous_graph': [], # if any variables (besides x1) get recentered in the decoder/denoiser, exclude them from any heterogeneous graph (which requires a common reference frame).
    
    'training': {
        
        'train_x1_denoising': True,
        'train_x2_denoising': True,
        'train_x3_denoising': False,
        'train_x4_denoising': False,
        
        # effective batch size of 48
        'batch_size': 8,
        'accumulate_grad_batches': 3,
        'num_gpus': 2,
        
        'lr': 0.0003,
        'min_lr': 0.0003,
        'lr_steps': 1,
        
        'gradient_clip_val': 5.0,
        
        'num_workers': 10,
        
        'output_dir': 'x1x2_diffusion_gdb17_20240824/',
        
        'log_every_n_steps': 1000,
        
        'multiprocessing_spawn': True,
    },
    
    
    'dataset': {
    
        'explicit_hydrogens': True,
        'use_MMFF94_charges': False,
        'probe_radius': 0.6, # for x2 and x3
    
        'compute_x1': True,
        'compute_x2': True,
        'compute_x3': False,
        'compute_x4': False,
        
        'x1': {
            'recenter': True, 
            'add_virtual_node': True,
            'remove_noise_COM': True,
            'atom_types': atom_types,
            'charge_types': charge_types,
            'bond_types': bond_types,
            'scale_atom_features': 0.25,
            'scale_bond_features': 1.0,
        },
        
        
        'x2': {
            'recenter': False,
            'add_virtual_node': True,
            'remove_noise_COM': False,
            'num_points': 75,
            'independent_timesteps': False,
        },
        
        
        'x3': {
            'independent_timesteps': False, # coupled to x1 timesteps
            
            'recenter': False, 
            'add_virtual_node': True, 
            'remove_noise_COM': False,
            'num_points': 75,
                        
            'scale_node_features': 2.0, # scaling electrostatic potential
        }, 
        
        
        'x4': {
            'independent_timesteps': False, # coupled to x1 timesteps
            'recenter': False, 
            'add_virtual_node': True, 
            'remove_noise_COM': False,
            'max_node_types': num_pharmacophore_types,
            'scale_node_features': 2.0,
            'scale_vector_features': 2.0,
                        
            'multivectors': False,
            'check_accessibility': False,
        }, 

    },

    
    
    # Model Hyperparameters
    
    # for joint/global l1 embeddings (must be the same for each x1, x2, ...)
    'lmax_list': [1],
    'mmax_list': [1],
    'ffn_hidden_channels': 32,
    'grid_resolution': 16,
    
    
    'decoder_heterogeneous_graph_encoder': {
        'use': True,
        
        'num_layers': 2,
        'input_sphere_channels': num_channels,
        'sphere_channels': num_channels,
        
        'attn_hidden_channels': 24,
        'num_heads': 2,
        'attn_alpha_channels': 24,
        'attn_value_channels': 24,
        'ffn_hidden_channels': 32,
        
        'lmax_list': [1],
        'mmax_list': [1],
        'grid_resolution': 16,
        'cutoff': 5.0,
        'max_neighbors': 1000000, # essentially infinite
        
        'num_sphere_samples': 128,
        'edge_channels': 128,

    },
    
    
    
    
    'x1': {
        'decoder': {
            'input_node_channels': num_atom_types + num_charge_types,
            'node_channels': num_channels,
            'time_embedding_size': 32,
            
            'force_edges_to_virtual_nodes': True, # for both encoder and denoiser
                        
            'encoder': {
                
                'fully_connected': True, # whether to force the 3D graph to be fully connected

                'num_layers': 4,
                'input_sphere_channels': num_channels,
                'sphere_channels': num_channels,
                
                'input_bond_channels': num_bond_types,
                'edge_attr_channels': num_channels,
                
                'attn_hidden_channels': 32,
                'num_heads': 4,
                'attn_alpha_channels': 32,
                'attn_value_channels': 32,
                'ffn_hidden_channels': 64,
                
                'lmax_list': [1],
                'mmax_list': [1],
                'grid_resolution': 16,
                'cutoff': 5.0, # if fully_connected, this is still used for the Gaussian distance expansion
                'max_neighbors': 1000000, # essentially infinite
                
                'num_sphere_samples': 128,
                'edge_channels': 128,
            },
            
            'denoiser': {
                
                'output_node_channels': num_atom_types + num_charge_types, # must equal params['x1']['decoder']['input_node_channels']
                'output_bond_channels': num_bond_types, # must equal params['x1']['decoder']['input_bond_channels']
                
                # this is for the feature update
                'MLP_hidden_dim': 64,
                'num_MLP_hidden_layers': 2,
                
                # this is for the positional update
                'use_e3nn': True,
                'e3nn': {
                    'lmax_list': [1],
                    'mmax_list': [1],
                    'ffn_hidden_channels': 32,
                    'grid_resolution': 16,
                },
                
                'use_egnn_positions_update': True,
                'egnn': {
                    'normalize_egnn_vectors': True,
                    'distance_expansion_dim': 32,
                    'num_MLP_hidden_layers': 2,
                    'MLP_hidden_dim': 64,
                },
            
            },
            
        },
    },
     
    
    'x2': {
        'decoder': {
            'input_node_channels': 2, # real or virtual node
            'node_channels': num_channels,
            'time_embedding_size': 32,
            
            'force_edges_to_virtual_nodes': True, # for both encoder and denoiser
            
            'encoder': {
                'num_layers': 2,
                'input_sphere_channels': num_channels,
                'sphere_channels': num_channels,
                
                'attn_hidden_channels': 24,
                'num_heads': 2,
                'attn_alpha_channels': 24,
                'attn_value_channels': 24,
                'ffn_hidden_channels': 32,
                
                'lmax_list': [1],
                'mmax_list': [1],
                'grid_resolution': 16,
                'cutoff': 5.0,
                'max_neighbors': 1000000, # essentially infinite
                
                'num_sphere_samples': 128,
                'edge_channels': 128,
            },
            
            'denoiser': {
                
                'output_node_channels': num_channels, # ignored
                
                'use_e3nn': True,
                'e3nn': {
                    'lmax_list': [1],
                    'mmax_list': [1],
                    'ffn_hidden_channels': 32,
                    'grid_resolution': 16,
                },
                
                'use_egnn_positions_update': False,
                'egnn': {
                    'normalize_egnn_vectors': True,
                    'distance_expansion_dim': 32,
                    'num_MLP_hidden_layers': 2,
                    'MLP_hidden_dim': 64,
                },
            
            },
            
        },
        
    },
        

    # ignored
    'x3': {
        'decoder': {
        
            'scalar_expansion_min': -10.0,
            'scalar_expansion_max': 10.0,
            'input_node_channels': num_channels,
            'node_channels': num_channels,
            'time_embedding_size': 32,
            
            'force_edges_to_virtual_nodes': True, # for both encoder and denoiser
            
            
            'encoder': {
                'num_layers': 2,
                'input_sphere_channels': num_channels,
                'sphere_channels': num_channels,
                
                'attn_hidden_channels': 24,
                'num_heads': 2,
                'attn_alpha_channels': 24,
                'attn_value_channels': 24,
                'ffn_hidden_channels': 32,
                
                'lmax_list': [1],
                'mmax_list': [1],
                'grid_resolution': 16,
                'cutoff': 5.0,
                'max_neighbors': 1000000, # essentially infinite
                
                'num_sphere_samples': 128,
                'edge_channels': 128,
            }, 
            
            
            'denoiser': {
            
                'output_node_channels': 1, # denoised coulombic potential / partial charge
                
                'MLP_hidden_dim': 64,
                'num_MLP_hidden_layers': 2,
                
                'use_e3nn': True,
                'e3nn': {
                    'lmax_list': [1],
                    'mmax_list': [1],
                    'ffn_hidden_channels': 32,
                    'grid_resolution': 16,
                },
                
                'use_egnn_positions_update': False,
                'egnn': {
                    'normalize_egnn_vectors': True,
                    'distance_expansion_dim': 32,
                    'num_MLP_hidden_layers': 2,
                    'MLP_hidden_dim': 64,
                },
            
            },
            
        },
        
    },
    
    # ignored
    'x4': {
        'decoder': {
        
            'input_node_channels': num_pharmacophore_types,
            'node_channels': num_channels,
            'time_embedding_size': 32,
            
            'force_edges_to_virtual_nodes': True, # for both encoder and denoiser
            
            'encoder': {
                'num_layers': 2,
                'input_sphere_channels': num_channels,
                'sphere_channels': num_channels,
                
                'attn_hidden_channels': 24,
                'num_heads': 2,
                'attn_alpha_channels': 24,
                'attn_value_channels': 24,
                'ffn_hidden_channels': 32,
                
                'lmax_list': [1],
                'mmax_list': [1],
                'grid_resolution': 16,
                'cutoff': 5.0,
                'max_neighbors': 1000000, # essentially infinite
                
                'num_sphere_samples': 128,
                'edge_channels': 128,
            }, 
            
            
            'denoiser': {
            
                'output_node_channels': num_pharmacophore_types, # must equal params['x4']['decoder']['input_node_channels']
                
                'MLP_hidden_dim': 64,
                'num_MLP_hidden_layers': 2,
                
                'use_e3nn': True, # ONLY RELEVANT FOR DENOISING POSITIONS; denoising directions use e3nn automatically
                'e3nn': {
                    'lmax_list': [1],
                    'mmax_list': [1],
                    'ffn_hidden_channels': 32,
                    'grid_resolution': 16,
                },
                
                'use_egnn_positions_update': False, # ONLY RELEVANT FOR DENOISING POSITIONS
                'egnn': {
                    'normalize_egnn_vectors': True,
                    'distance_expansion_dim': 32,
                    'num_MLP_hidden_layers': 2,
                    'MLP_hidden_dim': 64,
                },
            
            },
            
        },
        
    },
    
}



noise_schedule_dict = {}

T = 400
ts = np.arange(1, T + 1)

beta_min = 0.001 / (T//100)
beta_max = 0.35 / (T//100)
beta_ts_linear = beta_min + ts / T * (beta_max - beta_min) # variance schedule used by RFDiffusion for translations

# (slightly adjusted) cosine schedule, introduced by https://arxiv.org/pdf/2102.09672
ts_ = np.arange(0, T + 1)
s = 0.008
f_ts = np.cos(np.pi/2.1 * ((ts_/ (T+1)) + s)/(1. + s) )**2.0
f_ts = f_ts / f_ts[0]
f_ts = np.clip(f_ts, 0.0001, 0.9999)
beta_ts_cosine = (1 - f_ts[1:]/f_ts[0:-1])
beta_ts_cosine = np.clip(beta_ts_cosine, 0.0001, 0.9999)

beta_ts = 0.65*beta_ts_cosine + 0.35*beta_ts_linear

sigma_ts = beta_ts**0.5 # std deviation schedule
alpha_ts = (1. - sigma_ts**2.0)**0.5

alpha_dash_ts = np.cumprod(alpha_ts)
var_dash_ts = 1. - alpha_dash_ts**2.0
sigma_dash_ts = var_dash_ts**0.5


noise_schedule_dict['x1'] = {
    'T': T,
    'ts': ts,
    'alpha_ts': alpha_ts,
    'sigma_ts': sigma_ts,
    'alpha_dash_ts': alpha_dash_ts,
    'var_dash_ts': var_dash_ts,
    'sigma_dash_ts': sigma_dash_ts,
}

noise_schedule_dict['x2'] = {
    'T': T,
    'ts': ts,
    'alpha_ts': alpha_ts,
    'sigma_ts': sigma_ts,
    'alpha_dash_ts': alpha_dash_ts,
    'var_dash_ts': var_dash_ts,
    'sigma_dash_ts': sigma_dash_ts,
}

noise_schedule_dict['x3'] = {
    'T': T,
    'ts': ts,
    'alpha_ts': alpha_ts,
    'sigma_ts': sigma_ts,
    'alpha_dash_ts': alpha_dash_ts,
    'var_dash_ts': var_dash_ts,
    'sigma_dash_ts': sigma_dash_ts,
}

noise_schedule_dict['x4'] = {
    'T': T,
    'ts': ts,
    'alpha_ts': alpha_ts,
    'sigma_ts': sigma_ts,
    'alpha_dash_ts': alpha_dash_ts,
    'var_dash_ts': var_dash_ts,
    'sigma_dash_ts': sigma_dash_ts,
}

params['noise_schedules'] = noise_schedule_dict
