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
import numpy as np
import torch
import torch_geometric

from copy import deepcopy

# in most cases, this function won't be used, as we use xTB charges rather than MMFF charges.
def get_atomic_partial_charges(mol: rdkit.Chem.Mol) -> np.ndarray:
    """
    Gets partial charges for a given molecule.
    Assumes the input "mol" already has an optimized conformer. Gets partial charges from
    MMFF or Gasteiger.

    Parameters
    ----------
    mol : rdkit.Chem.Mol object
        RDKit molecule object with an optimized geometry in conformers.
    
    Returns
    -------
    np.ndarray (N)
        Partial charges for each atom in the molecule.
    """
    
    try:
        mol.GetConformer()
    except ValueError as e:
        raise ValueError(f"Provided rdkit.Chem.Mol object did not have conformer embedded.", e)
    
    molec_props = rdkit.Chem.AllChem.MMFFGetMoleculeProperties(mol)
    if molec_props:
        # electron units
        charges = np.array([molec_props.GetMMFFPartialCharge(i) for i, _ in enumerate(mol.GetAtoms())])
    else:
        print("MMFF charges not available for the input molecule, defaulting to Gasteiger charges.")
        rdkit.Chem.AllChem.ComputeGasteigerCharges(mol)
        charges=np.array([a.GetDoubleProp('_GasteigerCharge') for a in mol.GetAtoms()])
    
    return charges


class HeteroDataset(torch_geometric.data.Dataset):
    def __init__(self, 
                 
            molblocks_and_charges, 
            
            noise_schedule_dict,
            
            explicit_hydrogens = True,
            use_MMFF94_charges = False,
            formal_charge_diffusion = False,
            
            x1 = True,
            x2 = True,
            x3 = True,
            x4 = True,
                
            recenter_x1 = True, 
            add_virtual_node_x1 = True, 
            remove_noise_COM_x1 = True,
            atom_types_x1 = [None, 'H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P', 'Si'],
            charge_types_x1 = [0,1,2,-1,-2],
            bond_types_x1 = [None, 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
            scale_atom_features_x1 = 1.0,
            scale_bond_features_x1 = 1.0,

            
            
            independent_timesteps_x2 = False,
            recenter_x2 = False, # we want the center of x2 to be the virtual node (whose position is the center of x1)
            add_virtual_node_x2 = True,
            remove_noise_COM_x2 = False,
            num_points_x2 = 75,

            
            independent_timesteps_x3 = False,
            recenter_x3 = False,
            add_virtual_node_x3 = True,
            remove_noise_COM_x3 = False,
            num_points_x3 = 75,
            scale_node_features_x3 = 1.0,
            
                 
            independent_timesteps_x4 = False,
            recenter_x4 = False, 
            add_virtual_node_x4 = True, # must be true, for edge-case where molecule doesn't have any pharamcophores
            remove_noise_COM_x4 = False,
            max_node_types_x4 = 16, # number of pharmacophore types (can be set larger than represented in dataset)
            scale_node_features_x4 = 1.0,
            scale_vector_features_x4 = 1.0,
            multivectors = False,
            check_accessibility = False,
            
            probe_radius = 0.6,                 
        ):
        
        self.molblocks_and_charges = molblocks_and_charges
        self.length = len(molblocks_and_charges)
        self.use_MMFF94_charges = use_MMFF94_charges
        
        self.noise_schedule_dict = noise_schedule_dict
        
        self.explicit_hydrogens = explicit_hydrogens
        assert self.explicit_hydrogens == True
        
        self.formal_charge_diffusion = formal_charge_diffusion
        
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        
        self.recenter_x1 = recenter_x1
        self.add_virtual_node_x1 = add_virtual_node_x1
        self.remove_noise_COM_x1 = remove_noise_COM_x1 # True
        self.atom_types_x1 = atom_types_x1
        self.charge_types_x1 = charge_types_x1
        self.bond_types_x1 = bond_types_x1
        self.scale_atom_features_x1 = scale_atom_features_x1
        self.scale_bond_features_x1 = scale_bond_features_x1
        
        self.recenter_x2 = recenter_x2 
        self.add_virtual_node_x2 = add_virtual_node_x2
        self.remove_noise_COM_x2 = remove_noise_COM_x2
        self.num_points_x2 = num_points_x2
        self.independent_timesteps_x2 = independent_timesteps_x2
        
        self.recenter_x3 = recenter_x3
        self.add_virtual_node_x3 = add_virtual_node_x3
        self.remove_noise_COM_x3 = remove_noise_COM_x3
        self.num_points_x3 = num_points_x3
        self.independent_timesteps_x3 = independent_timesteps_x3
        self.scale_node_features_x3 = scale_node_features_x3

        self.independent_timesteps_x4 = independent_timesteps_x4
        self.recenter_x4 = recenter_x4
        self.add_virtual_node_x4 = add_virtual_node_x4 
        self.remove_noise_COM_x4 = remove_noise_COM_x4
        self.max_node_types_x4 = max_node_types_x4
        self.scale_node_features_x4 = scale_node_features_x4
        self.scale_vector_features_x4 = scale_vector_features_x4
        self.multivectors = multivectors
        self.check_accessibility = check_accessibility
        
        self.probe_radius = probe_radius
        self.scale_electrostatics = self.scale_node_features_x3  # alias
    
    
    
    def get_x1_data(self, mol, t, alpha_dash_t, sigma_dash_t):
        # this uses the same noise schedule for both positions and atom types/features
        
        data = {}
        data['timestep'] = torch.as_tensor(np.array([t]))

        atom_types = [self.atom_types_x1.index(a.GetSymbol()) for a in mol.GetAtoms()]
        if self.formal_charge_diffusion:
            formal_charges = [int(a.GetFormalCharge()) for a in mol.GetAtoms()]
            formal_charge_map = {c:self.charge_types_x1.index(c) for c in self.charge_types_x1}
            formal_charges_mapped = [formal_charge_map[f] for f in formal_charges]
            
        pos = np.array(mol.GetConformer().GetPositions())
        num_atoms = len(pos)
        
        bond_adj = 1-np.diag(np.ones(num_atoms, dtype = int))
        bond_adj = np.triu(bond_adj) # directed graph, to only include 1 edge per bond
        bond_edge_index = np.stack(bond_adj.nonzero(), axis = 0) # this doesn't include any edges to the virtual node
        bond_types_dict = {b:self.bond_types_x1.index(b) for b in self.bond_types_x1}
        max_bond_types_x1 = len(bond_types_dict)
        bond_types = []
        for b in range(bond_edge_index.shape[1]):
            idx_1 = int(bond_edge_index[0, b])
            idx_2 = int(bond_edge_index[1, b])
            bond = mol.GetBondBetweenAtoms(idx_1, idx_2)
            if bond is None:
                bond_types.append(bond_types_dict[None]) # non-bonded edge type; == 0
            else:
                bond_type = bond_types_dict[str(bond.GetBondType())]
                bond_types.append(bond_type)
        data['bond_edge_mask'] = torch.as_tensor(np.array(bond_types) != 0, dtype = torch.bool) # True indicates a real bond
        
        
        COM_before_centering = pos.mean(0)[None, ...]
        data['com_before_centering'] = torch.as_tensor(COM_before_centering, dtype = torch.float)
        pos_recentered = pos - pos.mean(0)
        if self.recenter_x1:
            pos = pos_recentered
        COM = pos.mean(0)[None, ...]
        data['com'] = torch.as_tensor(COM, dtype = torch.float)

        virtual_node_mask = np.zeros(pos.shape[0] + int(self.add_virtual_node_x1))
        if self.add_virtual_node_x1: # should change according to desired behavior
            assert self.atom_types_x1[0] == None
            atom_types.insert(0, 0)
            bond_edge_index = bond_edge_index + 1 # accounting for virtual node
            virtual_node_pos = COM
            pos = np.concatenate([virtual_node_pos, pos], axis = 0) # setting virtual node position to (non-zero) COM
            pos_recentered = np.concatenate([virtual_node_pos * 0.0, pos_recentered], axis = 0) # setting virtual node position to zero
            virtual_node_mask[0] = 1
        virtual_node_mask = virtual_node_mask == 1
        num_nodes = num_atoms + int(self.add_virtual_node_x1)
        
        data['bond_edge_index'] = torch.as_tensor(bond_edge_index, dtype = torch.long)
        data['pos'] = torch.as_tensor(pos, dtype = torch.float)
        data['pos_recentered'] = torch.as_tensor(pos_recentered, dtype = torch.float)
        data['virtual_node_mask'] = torch.as_tensor(virtual_node_mask)
        
        
        # (scaled) one-hot embedding of atom types and formal charges for non-noised structure
        x = np.zeros((num_nodes, len(self.atom_types_x1))) #torch.as_tensor(atomic_numbers, dtype = torch.long)
        x[np.arange(num_nodes), atom_types] = 1
        x = x * self.scale_atom_features_x1
        if self.formal_charge_diffusion:
            x_formal_charges = np.zeros((len(formal_charges_mapped), len(self.charge_types_x1)))
            x_formal_charges[np.arange(len(formal_charges_mapped)), formal_charges_mapped] = 1
            x_formal_charges = x_formal_charges * self.scale_atom_features_x1
            if self.add_virtual_node_x1:
                # virtual node has all zeros for the formal charge one-hot features
                x_formal_charges = np.concatenate((np.zeros(len(self.charge_types_x1), dtype = x_formal_charges.dtype)[None, ...], x_formal_charges), axis = 0)
            x = np.concatenate((x, x_formal_charges), axis = 1)
        data['x'] = torch.as_tensor(x, dtype = torch.float)
        
        
        # (scaled) one-hot embedding of bond types for non-noised structure
            # this doesn't include any edges to the virtual node
        bond_edge_x = np.zeros((bond_edge_index.shape[1], max_bond_types_x1))
        bond_edge_x[np.arange(len(bond_types)), bond_types] = 1
        bond_edge_x = bond_edge_x * self.scale_bond_features_x1
        data['bond_edge_x'] = torch.as_tensor(bond_edge_x, dtype = torch.float)
        
        
        # forward noising non-virtual-nodes
        
        pos_noise = np.random.randn(*pos.shape)
        pos_noise[virtual_node_mask] = 0.0
        if self.remove_noise_COM_x1: # removing COM from added noise
            pos_noise[~virtual_node_mask] = pos_noise[~virtual_node_mask] - np.mean(pos_noise[~virtual_node_mask], axis = 0) 
        data['pos_noise'] = torch.as_tensor(pos_noise, dtype = torch.float)
        
        x_noise = np.random.randn(*x.shape)
        x_noise[virtual_node_mask] = 0.0
        data['x_noise'] = torch.as_tensor(x_noise, dtype = torch.float)
        
        # this doesn't include any edges to the virtual node
        bond_edge_x_noise = np.random.randn(*bond_edge_x.shape)
        data['bond_edge_x_noise'] = torch.as_tensor(bond_edge_x_noise, dtype = torch.float)
        
        
        pos_forward_noised = alpha_dash_t * pos  +  sigma_dash_t * pos_noise 
        pos_forward_noised[virtual_node_mask] = pos[virtual_node_mask]
        data['pos_forward_noised'] = torch.as_tensor(pos_forward_noised, dtype = torch.float)
        
        x_forward_noised = alpha_dash_t * x  +  sigma_dash_t * x_noise 
        x_forward_noised[virtual_node_mask] = x[virtual_node_mask]
        data['x_forward_noised'] = torch.as_tensor(x_forward_noised, dtype = torch.float)
        
        bond_edge_x_forward_noised = alpha_dash_t * bond_edge_x  +  sigma_dash_t * bond_edge_x_noise 
        data['bond_edge_x_forward_noised'] = torch.as_tensor(bond_edge_x_forward_noised, dtype = torch.float)

        return data, pos, virtual_node_mask
    
    
    
    def get_x2_data(self, radii, atom_centers, num_points, recenter, add_virtual_node, remove_noise_COM, t, alpha_dash_t, sigma_dash_t, virtual_node_pos = None):
        
        data = {}
        data['timestep'] = torch.as_tensor(np.array([t]))
        
        pos = get_molecular_surface(
            atom_centers,
            radii,
            num_points=num_points,
            probe_radius = self.probe_radius,
            num_samples_per_atom = 20,
        )
        
        COM_before_centering = pos.mean(0)[None, :]
        data['com_before_centering'] = torch.as_tensor(COM_before_centering, dtype = torch.float)
        pos_recentered = pos - pos.mean(0)
        if recenter:
            pos = pos_recentered
        COM = pos.mean(0)[None, :]
        data['com'] = torch.as_tensor(COM, dtype = torch.float)
        
        virtual_node_mask = np.zeros(pos.shape[0] + int(add_virtual_node))
        if add_virtual_node: # should change according to desired behavior
            if (virtual_node_pos is None) or (recenter == True):
                virtual_node_pos = COM
            pos = np.concatenate([virtual_node_pos, pos], axis = 0)
            pos_recentered = np.concatenate([virtual_node_pos * 0.0, pos_recentered], axis = 0)
            virtual_node_mask[0] = 1
        virtual_node_mask = virtual_node_mask == 1
        
        data['pos'] = torch.as_tensor(pos, dtype = torch.float)
        data['pos_recentered'] = torch.as_tensor(pos_recentered, dtype = torch.float)
        data['virtual_node_mask'] = torch.as_tensor(virtual_node_mask)
        
        
        # one-hot embedding indicating real vs virtual nodes
        x = np.zeros((pos.shape[0], 2))
        x[~virtual_node_mask,0] = 1
        x[virtual_node_mask,1] = 1
        data['x'] = torch.as_tensor(x, dtype = torch.float)
        data['x_forward_noised'] = data['x'] # there are no features to be noised in x2
        
        # forward noising non-virtual-nodes
        pos_noise = np.random.randn(*pos.shape)
        pos_noise[virtual_node_mask] = 0.0
        if remove_noise_COM:
            pos_noise[~virtual_node_mask] = pos_noise[~virtual_node_mask] - np.mean(pos_noise[~virtual_node_mask], axis = 0) # removing COM from added noise
        data['pos_noise'] = torch.as_tensor(pos_noise, dtype = torch.float)
        
        pos_forward_noised = alpha_dash_t * pos  +  sigma_dash_t * pos_noise 
        pos_forward_noised[virtual_node_mask] = pos[virtual_node_mask]
        data['pos_forward_noised'] = torch.as_tensor(pos_forward_noised, dtype = torch.float)
        
        return data, pos, virtual_node_mask
    
    
    
    def get_x3_data_electrostatics_only(self, charges, charge_centers, data, pos, virtual_node_mask, t, alpha_dash_t, sigma_dash_t):
        
        x = get_electrostatics_given_point_charges(charges, charge_centers, pos) # compute ESP at each point in pos
        x[virtual_node_mask] = 0.0
        x = x * self.scale_node_features_x3
        
        data['x'] = torch.as_tensor(x, dtype = torch.float)
        
        x_noise = np.random.randn(*x.shape)
        x_noise[virtual_node_mask] = 0.0
        data['x_noise'] = torch.as_tensor(x_noise, dtype = torch.float)
        
        x_forward_noised = alpha_dash_t * x  +  sigma_dash_t * x_noise 
        x_forward_noised[virtual_node_mask] = x[virtual_node_mask]
        data['x_forward_noised'] = torch.as_tensor(x_forward_noised, dtype = torch.float)
        
        return data
    
    
    def get_x4_data(self, mol, recenter, add_virtual_node, remove_noise_COM, t, alpha_dash_t, sigma_dash_t, virtual_node_pos = None):
        
        # it is  important to include a virtual node in case there are NO pharmacophores in the molecule
        assert add_virtual_node
        
        data = {}
        data['timestep'] = torch.as_tensor(np.array([t]))
        
        pharm_types, pos, direction = get_pharmacophores(
            mol, 
            multi_vector = self.multivectors, 
            check_access=self.check_accessibility,
        )
        pharm_types = pharm_types + 1 # need to accomodate potential virtual node as 0th index
        
        # add a small amount of noise to positions of pharmacophores to avoid identically overlapping points
        pos = pos + np.random.randn(*pos.shape) * 0.05
        
        # no pharmacophores --> only virtual node remains
        if pharm_types.shape[0] == 0:
            
            pharm_types = np.array([0])
            x = np.zeros((pharm_types.size, self.max_node_types_x4))
            x[np.arange(pharm_types.size), pharm_types] = 1
            x = x * self.scale_node_features_x4
            data['x'] = torch.as_tensor(x, dtype = torch.float)
            
            if (virtual_node_pos is None) or (recenter == True):
                virtual_node_pos = np.zeros(3)[None, ...]
            data['com_before_centering'] = torch.as_tensor(virtual_node_pos, dtype = torch.float)
            data['com'] = torch.as_tensor(virtual_node_pos, dtype = torch.float)
            
            virtual_node_mask = np.array([1])
            virtual_node_mask = virtual_node_mask == 1
            
            pos = virtual_node_pos
            direction = np.zeros(3)[None, ...]
            
            direction = direction * self.scale_vector_features_x4
            
            data['pos'] = torch.as_tensor(pos, dtype = torch.float)
            data['pos_recentered'] = torch.as_tensor(pos * 0.0, dtype = torch.float)
            data['direction'] = torch.as_tensor(direction, dtype = torch.float)
            data['virtual_node_mask'] = torch.as_tensor(virtual_node_mask)
            
            # virtual node remains unnoised
            x_noise = np.zeros(x.shape)
            data['x_noise'] = torch.as_tensor(x_noise, dtype = torch.float)
            x_forward_noised = x
            data['x_forward_noised'] = torch.as_tensor(x_forward_noised, dtype = torch.float)
            
            pos_noise = np.zeros(pos.shape)
            data['pos_noise'] = torch.as_tensor(pos_noise, dtype = torch.float)
            pos_forward_noised = pos
            data['pos_forward_noised'] = torch.as_tensor(pos_forward_noised, dtype = torch.float)
            
            direction_noise = np.zeros(direction.shape)
            data['direction_noise'] = torch.as_tensor(direction_noise, dtype = torch.float)
            direction_forward_noised = direction
            data['direction_forward_noised'] = torch.as_tensor(direction_forward_noised, dtype = torch.float)
            
            return data
        
        
        COM_before_centering = pos.mean(0)[None, :]
        data['com_before_centering'] = torch.as_tensor(COM_before_centering, dtype = torch.float)
        pos_recentered = pos - pos.mean(0)
        if recenter:
            pos = pos_recentered
        COM = pos.mean(0)[None, :]
        data['com'] = torch.as_tensor(COM, dtype = torch.float)
        
        
        virtual_node_mask = np.zeros(pos.shape[0] + int(add_virtual_node))
        if add_virtual_node: # should change according to desired behavior
            if (virtual_node_pos is None) or (recenter == True):
                virtual_node_pos = COM
            
            pharm_types = np.concatenate([np.array([0]), pharm_types], axis = 0)
            pos = np.concatenate([virtual_node_pos, pos], axis = 0)
            pos_recentered = np.concatenate([virtual_node_pos * 0.0, pos_recentered], axis = 0)
            direction = np.concatenate([np.zeros(3)[None, ...], direction], axis = 0)
            
            virtual_node_mask[0] = 1
        virtual_node_mask = virtual_node_mask == 1
        
        
        x = np.zeros((pharm_types.size, self.max_node_types_x4)) #torch.as_tensor(atomic_numbers, dtype = torch.long)
        x[np.arange(pharm_types.size), pharm_types] = 1
        x = x * self.scale_node_features_x4
        data['x'] = torch.as_tensor(x, dtype = torch.float)
        
        data['pos'] = torch.as_tensor(pos , dtype = torch.float)
        data['pos_recentered'] = torch.as_tensor(pos_recentered , dtype = torch.float)
        
        direction = direction * self.scale_vector_features_x4
        data['direction'] = torch.as_tensor(direction, dtype = torch.float)
        data['virtual_node_mask'] = torch.as_tensor(virtual_node_mask)
        
        
        # forward noising non-virtual-nodes
            
        x_noise = np.random.randn(*x.shape)
        x_noise[virtual_node_mask] = 0.0 # x_noise[virtual_node_mask] * 0.0
        data['x_noise'] = torch.as_tensor(x_noise, dtype = torch.float)
        
        x_forward_noised = alpha_dash_t * x  +  sigma_dash_t * x_noise 
        x_forward_noised[virtual_node_mask] = x[virtual_node_mask]
        data['x_forward_noised'] = torch.as_tensor(x_forward_noised, dtype = torch.float)
        
        
        pos_noise = np.random.randn(*pos.shape)
        pos_noise[virtual_node_mask] = 0.0
        if remove_noise_COM: # removing COM from added noise
            pos_noise[~virtual_node_mask] = pos_noise[~virtual_node_mask] - np.mean(pos_noise[~virtual_node_mask], axis = 0) 
        data['pos_noise'] = torch.as_tensor(pos_noise, dtype = torch.float)
        
        pos_forward_noised = alpha_dash_t * pos  +  sigma_dash_t * pos_noise 
        pos_forward_noised[virtual_node_mask] = pos[virtual_node_mask]
        data['pos_forward_noised'] = torch.as_tensor(pos_forward_noised, dtype = torch.float)
        
        
        direction_noise = np.random.randn(*direction.shape)
        direction_noise[virtual_node_mask] = 0.0
        data['direction_noise'] = torch.as_tensor(direction_noise, dtype = torch.float)
        
        direction_forward_noised = alpha_dash_t * direction  +  sigma_dash_t * direction_noise 
        direction_forward_noised[virtual_node_mask] = direction[virtual_node_mask]
        data['direction_forward_noised'] = torch.as_tensor(direction_forward_noised, dtype = torch.float)
        
        return data
    
    
    def __getitem__(self, k):
        
        mol_block = self.molblocks_and_charges[k][0]
        charges = np.array(self.molblocks_and_charges[k][1]) # precomputed charges (e.g., from xTB)
        
        mol = rdkit.Chem.MolFromMolBlock(mol_block, removeHs = False)
        atomic_numbers = np.array([int(a.GetAtomicNum()) for a in mol.GetAtoms()])
        
        assert self.explicit_hydrogens # if we want to treat hydrogens implicitly, then we need to adjust how x2,x3,x4 are computed
        
        # centering molecule coordinates
        mol_coordinates = np.array(mol.GetConformer().GetPositions())
        mol_coordinates = mol_coordinates - np.mean(mol_coordinates, axis = 0)
        #mol = update_mol_coordinates(mol, mol_coordinates, copy = False)
        mol = update_mol_coordinates(mol, mol_coordinates)

        radii = get_atomic_vdw_radii(mol)
        if self.use_MMFF94_charges:
            charges = get_atomic_partial_charges(mol) #MMFF94 charges
        
        data_dict = {
            'molecule_id': torch.as_tensor(np.array([k]), dtype = torch.long),
            'x1': {},
            'x2': {},
            'x3': {},
            'x4': {},
        }
        
        
        if self.x1:
            ts = self.noise_schedule_dict['x1']['ts']
            
            #t = np.random.choice(ts)  # random time step sampled uniformly from time sequence
            T = ts.shape[0]
            ts_end = ts[0:int(T*0.125)] # 0 to 50 for T=400
            ts_middle = ts[int(T*0.125):int(T*0.625)] # 50 to 250 for T=400
            ts_start = ts[int(T*0.625):] # 250 to 400 for T=400
            ts_prob = np.random.uniform(0,1)
            if ts_prob < 0.075:
                t = np.random.choice(ts_end) # 7.5% chance to sample from last time steps
            elif ts_prob < (0.075 + 0.75):
                t = np.random.choice(ts_middle) # 75% chance to sample from middle time steps
            else:
                t = np.random.choice(ts_start) # 17.5% chance to sample from starting time steps
            
            ts_x1 = ts
            t_x1 = t
            t_idx = np.where(ts == t)[0][0]
            alpha_t = self.noise_schedule_dict['x1']['alpha_ts'][t_idx]
            sigma_t = self.noise_schedule_dict['x1']['sigma_ts'][t_idx]
            alpha_dash_t = self.noise_schedule_dict['x1']['alpha_dash_ts'][t_idx]
            var_dash_t = self.noise_schedule_dict['x1']['var_dash_ts'][t_idx]
            sigma_dash_t = self.noise_schedule_dict['x1']['sigma_dash_ts'][t_idx]
            
            x1_data, x1_pos, x1_virtual_node_mask = self.get_x1_data(mol, t, alpha_dash_t, sigma_dash_t)
            
            x1_data['alpha_t'] = torch.as_tensor(np.array([alpha_t]), dtype = torch.float)
            x1_data['sigma_t'] = torch.as_tensor(np.array([sigma_t]), dtype = torch.float)
            x1_data['alpha_dash_t'] = torch.as_tensor(np.array([alpha_dash_t]), dtype = torch.float)
            x1_data['sigma_dash_t'] = torch.as_tensor(np.array([sigma_dash_t]), dtype = torch.float)
                        
            data_dict['x1'] = x1_data
        
        
        if self.x2:
            
            if self.independent_timesteps_x2:
                ts = self.noise_schedule_dict['x2']['ts']
                
                #t = np.random.choice(ts)  # random time step sampled uniformly from time sequence
                T = ts.shape[0]
                ts_end = ts[0:int(T*0.125)] # 0 to 50 for T=400
                ts_middle = ts[int(T*0.125):int(T*0.625)] # 50 to 250 for T=400
                ts_start = ts[int(T*0.625):] # 250 to 400 for T=400
                ts_prob = np.random.uniform(0,1)
                if ts_prob < 0.075:
                    t = np.random.choice(ts_end) # 7.5% chance to sample from last time steps
                elif ts_prob < (0.075 + 0.75):
                    t = np.random.choice(ts_middle) # 75% chance to sample from middle time steps
                else:
                    t = np.random.choice(ts_start) # 17.5% chance to sample from starting time steps
                
            else:
                assert self.x1 == True
                # use same time sequence as x1
                assert (self.noise_schedule_dict['x2']['ts'] == self.noise_schedule_dict['x1']['ts']).all()
                ts = ts_x1
                t = t_x1
            ts_x2 = ts
            t_x2 = t
            t_idx = np.where(ts == t)[0][0]
            alpha_t = self.noise_schedule_dict['x2']['alpha_ts'][t_idx]
            sigma_t = self.noise_schedule_dict['x2']['sigma_ts'][t_idx]
            alpha_dash_t = self.noise_schedule_dict['x2']['alpha_dash_ts'][t_idx]
            var_dash_t = self.noise_schedule_dict['x2']['var_dash_ts'][t_idx]
            sigma_dash_t = self.noise_schedule_dict['x2']['sigma_dash_ts'][t_idx]
            
            if self.x1:
                atom_centers = x1_pos[~x1_virtual_node_mask,:]
                virtual_node_pos = atom_centers.mean(0)[None, ...] if ((self.add_virtual_node_x2) and (self.recenter_x2 == False)) else None
            else:
                atom_centers = mol_coordinates
                virtual_node_pos = None # this will get re-set to be the COM of x2 (NOT mol_coordinates) in get_x2_data
            
            x2_data, x2_pos, x2_virtual_node_mask = self.get_x2_data(
                radii,
                atom_centers, 
                self.num_points_x2,
                self.recenter_x2,
                self.add_virtual_node_x2,
                self.remove_noise_COM_x2,
                t, alpha_dash_t, sigma_dash_t, 
                virtual_node_pos = virtual_node_pos,
            )
            
            x2_data['alpha_t'] = torch.as_tensor(np.array([alpha_t]), dtype = torch.float)
            x2_data['sigma_t'] = torch.as_tensor(np.array([sigma_t]), dtype = torch.float)
            x2_data['alpha_dash_t'] = torch.as_tensor(np.array([alpha_dash_t]), dtype = torch.float)
            x2_data['sigma_dash_t'] = torch.as_tensor(np.array([sigma_dash_t]), dtype = torch.float)
                        
            data_dict['x2'] = x2_data
        
        
        
        if self.x3:
            if self.independent_timesteps_x3:
                ts = self.noise_schedule_dict['x3']['ts']
                
                #t = np.random.choice(ts)  # random time step sampled uniformly from time sequence
                T = ts.shape[0]
                ts_end = ts[0:int(T*0.125)] # 0 to 50 for T=400
                ts_middle = ts[int(T*0.125):int(T*0.625)] # 50 to 250 for T=400
                ts_start = ts[int(T*0.625):] # 250 to 400 for T=400
                ts_prob = np.random.uniform(0,1)
                if ts_prob < 0.075:
                    t = np.random.choice(ts_end) # 7.5% chance to sample from last time steps
                elif ts_prob < (0.075 + 0.75):
                    t = np.random.choice(ts_middle) # 75% chance to sample from middle time steps
                else:
                    t = np.random.choice(ts_start) # 17.5% chance to sample from starting time steps
                
            else:
                assert self.x1 == True
                
                # use same time sequence as x1 
                assert (self.noise_schedule_dict['x3']['ts'] == self.noise_schedule_dict['x1']['ts']).all()
                ts = ts_x1
                t = t_x1
            
            ts_x3 = ts
            t_x3 = t
            t_idx = np.where(ts == t)[0][0]
            alpha_t = self.noise_schedule_dict['x3']['alpha_ts'][t_idx]
            sigma_t = self.noise_schedule_dict['x3']['sigma_ts'][t_idx]
            alpha_dash_t = self.noise_schedule_dict['x3']['alpha_dash_ts'][t_idx]
            var_dash_t = self.noise_schedule_dict['x3']['var_dash_ts'][t_idx]
            sigma_dash_t = self.noise_schedule_dict['x3']['sigma_dash_ts'][t_idx]
            
            if self.x1:
                atom_centers = x1_pos[~x1_virtual_node_mask,:]
                virtual_node_pos = atom_centers.mean(0)[None, ...] if ((self.add_virtual_node_x3) and (self.recenter_x3 == False)) else None  
            else:
                atom_centers = mol_coordinates # this might need to be centered before we assign it to charge_centers
                virtual_node_pos = None # this will get re-set to be the COM of x3 (NOT mol_coordinates) in get_x3_data
            
            # we use the same surface cloud formulation as x2 for the points in x3
            x3_data, x3_pos, x3_virtual_node_mask, x3_duplicate_points = self.get_x2_data(
                radii, 
                atom_centers, 
                self.num_points_x3, 
                self.recenter_x3, 
                self.add_virtual_node_x3, 
                self.remove_noise_COM_x3, 
                t, alpha_dash_t, sigma_dash_t, 
                virtual_node_pos = virtual_node_pos,
            )
            
            # the x3 point cloud, if re-centered, is displaced from the atom centers used to generate it. 
                # Before computing electrostatics for x3, we have to displace the charge centers to account for this.
            x3_COM_displacement = x3_data['com'].numpy() - x3_data['com_before_centering'].numpy()
            charge_centers = atom_centers + x3_COM_displacement
            
            # same noise is applied to both coordinates and features
            x3_data = self.get_x3_data_electrostatics_only(
                charges, 
                charge_centers, 
                x3_data, 
                x3_pos, 
                x3_virtual_node_mask, 
                t, alpha_dash_t, sigma_dash_t,
            )
            
            x3_data['alpha_t'] = torch.as_tensor(np.array([alpha_t]), dtype = torch.float)
            x3_data['sigma_t'] = torch.as_tensor(np.array([sigma_t]), dtype = torch.float)
            x3_data['alpha_dash_t'] = torch.as_tensor(np.array([alpha_dash_t]), dtype = torch.float)
            x3_data['sigma_dash_t'] = torch.as_tensor(np.array([sigma_dash_t]), dtype = torch.float)
                        
            data_dict['x3'] = x3_data

        
        if self.x4:
            
            if self.independent_timesteps_x4:
                ts = self.noise_schedule_dict['x4']['ts']
                
                #t = np.random.choice(ts)  # random time step sampled uniformly from time sequence
                T = ts.shape[0]
                ts_end = ts[0:int(T*0.125)] # 0 to 50 for T=400
                ts_middle = ts[int(T*0.125):int(T*0.625)] # 50 to 250 for T=400
                ts_start = ts[int(T*0.625):] # 250 to 400 for T=400
                ts_prob = np.random.uniform(0,1)
                if ts_prob < 0.075:
                    t = np.random.choice(ts_end) # 7.5% chance to sample from last time steps
                elif ts_prob < (0.075 + 0.75):
                    t = np.random.choice(ts_middle) # 75% chance to sample from middle time steps
                else:
                    t = np.random.choice(ts_start) # 17.5% chance to sample from starting time steps
                
            else:
                assert self.x1 == True
                # use same time sequence as x1
                assert (self.noise_schedule_dict['x4']['ts'] == self.noise_schedule_dict['x1']['ts']).all()
                ts = ts_x1
                t = t_x1
            ts_x4 = ts
            t_x4 = t
            t_idx = np.where(ts == t)[0][0]
            alpha_t = self.noise_schedule_dict['x4']['alpha_ts'][t_idx]
            sigma_t = self.noise_schedule_dict['x4']['sigma_ts'][t_idx]
            alpha_dash_t = self.noise_schedule_dict['x4']['alpha_dash_ts'][t_idx]
            var_dash_t = self.noise_schedule_dict['x4']['var_dash_ts'][t_idx]
            sigma_dash_t = self.noise_schedule_dict['x4']['sigma_dash_ts'][t_idx]
            
            if self.x1:
                atom_centers = x1_pos[~x1_virtual_node_mask,:]
                virtual_node_pos = atom_centers.mean(0)[None, ...] if ((self.add_virtual_node_x4) and (self.recenter_x4 == False)) else None
            else:
                atom_centers = mol_coordinates
                virtual_node_pos = None # this will get re-set to be the COM of x4 (NOT mol_coordinates) in get_x4_data
            
            x4_data = self.get_x4_data(
                mol, 
                self.recenter_x4, 
                self.add_virtual_node_x4, 
                self.remove_noise_COM_x4, 
                t, 
                alpha_dash_t,
                sigma_dash_t,
                virtual_node_pos,
            )
            
            x4_data['alpha_t'] = torch.as_tensor(np.array([alpha_t]), dtype = torch.float)
            x4_data['sigma_t'] = torch.as_tensor(np.array([sigma_t]), dtype = torch.float)
            x4_data['alpha_dash_t'] = torch.as_tensor(np.array([alpha_dash_t]), dtype = torch.float)
            x4_data['sigma_dash_t'] = torch.as_tensor(np.array([sigma_dash_t]), dtype = torch.float)
                        
            data_dict['x4'] = x4_data
        
        
        data = torch_geometric.data.HeteroData(
            molecule_id = data_dict['molecule_id'],
            x1 = data_dict['x1'],
            x2 = data_dict['x2'],
            x3 = data_dict['x3'],
            x4 = data_dict['x4'],
            
            # this exploits some really weird PyG behavior: see https://github.com/pyg-team/pytorch_geometric/issues/7138
                # any '*_edge_index' that we want automatically incremented must be specified below (will NOT be incremented in data_dict['x1'], ...)
            x1__x1 = {'bond_edge_index': data_dict['x1']['bond_edge_index'], 'num_nodes': data_dict['x1']['pos'].shape[0]}, 
        )
        return data
    
    
    
    def __len__(self): return self.length
    def len(self): return self.__len__()
    def getitem(self, k): return self.__getitem__(k)
