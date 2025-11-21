import streamlit as st
import open3d
import os
import numpy as np
import torch
from tqdm import tqdm
from stmol import showmol
from rdkit import Chem
from rdkit.Chem import Draw
import py3Dmol
from typing import Literal
import pandas as pd
from copy import deepcopy
from io import StringIO

from shepherd_score.container import Molecule
from shepherd_score.conformer_generation import optimize_conformer_with_xtb, charges_from_single_point_conformer_with_xtb
from shepherd_score.conformer_generation import update_mol_coordinates
from shepherd_score.evaluations.evaluate import ConditionalEval
from shepherd.inference.sampler import generate, generate_from_intermediate_time
from shepherd_score.pharm_utils.pharmacophore import get_pharmacophores
from shepherd_score.conformer_generation import update_mol_coordinates

tmp_dir = os.environ.get('TMPDIR', './')


def get_conditioning_information(mol, optimize: bool = False,
                                 probe_radius: float = 0.6) -> Molecule:
    """
    Load a molecule from an SDF file and return a Molecule object.
    Centers the molecule's center of mass before creating the Molecule object.
    """
    # Center the molecule's center of mass
    mol_coordinates = np.array(mol.GetConformer().GetPositions())
    mol_coordinates = mol_coordinates - np.mean(mol_coordinates, axis=0)
    mol = update_mol_coordinates(mol, mol_coordinates)

    opt_mol, orig_partial_charges, opt_partial_charges = conformer_charges(mol, optimize=optimize)

    molecule = Molecule(opt_mol if optimize else mol,
                        num_surf_points=75,
                        probe_radius=probe_radius,
                        partial_charges = orig_partial_charges if not optimize else opt_partial_charges,
                        pharm_multi_vector=False)
    return molecule

def conformer_charges(mol: Chem.Mol, optimize: bool = False) -> tuple:
    """
    Generate charges for a molecule.
    """
    orig_partial_charges = charges_from_single_point_conformer_with_xtb(mol, solvent='water', charge=Chem.GetFormalCharge(mol), temp_dir=tmp_dir)
    if not optimize:
        return None, orig_partial_charges, None
    
    opt_mol, _, opt_partial_charges = optimize_conformer_with_xtb(mol, solvent='water', charge=Chem.GetFormalCharge(mol), temp_dir=tmp_dir)

    return opt_mol, orig_partial_charges, opt_partial_charges


def evaluate_conditional_samples(conditional_samples, molec: Molecule):
    """
    Sequential function that runs our evaluation suite on all of the samples with progress tracking.
    Use `shepherd_score`'s `ConditionalEval` class to evaluate each sample.
    It evaluates 1) the validity of the sampled molecule, 2) evaluates the
    quality of the conformer 3) measures the interaction profile similarity
    to the target molecule.

    Args
    ----
    conditional_samples : outputted samples
    molec : target molecule

    Returns
    -------
    results_df : pandas DataFrame with evaluation results
    cond_evals : list of `shepherd_score` `ConditionalEval` objects
    """
    # Prepare reference molecule for evaluation
    ref_molec = Molecule(molec.mol,
                         num_surf_points=400,
                         probe_radius=1.2,
                         partial_charges=molec.partial_charges,
                         pharm_multi_vector=False)

    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Sequential evaluation with progress tracking (no multiprocessing)
    evaluation_data = []
    cond_evals = []

    for i, sample in enumerate(conditional_samples):
        # Check for interruption during evaluation too
        if st.session_state.get('stop_generation', False):
            status_text.text("Evaluation stopped by user")
            progress_bar.empty()
            status_text.empty()
            break
            
        cond_eval = None
        sample_data = {}
        try:
            # Run the evaluation
            cond_eval = ConditionalEval(atoms=sample['x1']['atoms'],
                                        positions=sample['x1']['positions'],
                                        condition='all', solvent='water',
                                        ref_molec=ref_molec,
                                        num_surf_points=400,
                                        pharm_multi_vector=False)

            # Collect evaluation data
            sample_data = {
                'Valid': cond_eval.is_valid,
                'Valid (post-opt)': cond_eval.is_valid_post_opt,
                'SMILES (post-opt)': Chem.MolToSmiles(Chem.RemoveHs(cond_eval.mol_post_opt)) if cond_eval.is_valid_post_opt else None,
                'Shape Similarity': cond_eval.sim_surf_target_relax_optimal if cond_eval.is_valid_post_opt else None,
                'ESP Similarity': cond_eval.sim_esp_target_relax_optimal if cond_eval.is_valid_post_opt else None,
                'Pharm. Similarity': cond_eval.sim_pharm_target_relax_optimal if cond_eval.is_valid_post_opt else None,
                'SA Score': cond_eval.SA_score_post_opt if cond_eval.SA_score_post_opt else None,
                'QED': cond_eval.QED_post_opt if cond_eval.QED_post_opt else None,
                'LogP': cond_eval.logP_post_opt if cond_eval.logP_post_opt else None,
                'fsp3': cond_eval.fsp3_post_opt if cond_eval.fsp3_post_opt else None,
            }

        except Exception as e:
            cond_eval = None
            # Return error entry with concise error info for expander display
            error_msg = f"Sample {i}: {str(e)}"
            sample_data = {
                'Valid': False,
                'Valid (post-opt)': False,
                'SMILES (post-opt)': None,
                'Shape Similarity': None,
                'ESP Similarity': None,
                'Pharm. Similarity': None,
                'SA Score': None,
                'QED': None,
                'LogP': None,
                'fsp3': None,
                'Error': error_msg  # Concise error message for expander
            }

        evaluation_data.append(sample_data)
        cond_evals.append(cond_eval)

        # Update progress
        progress = min(1.0, (i+1) / len(conditional_samples))
        progress_bar.progress(progress)
        status_text.text(f"Finished evaluating {i+1}/{len(conditional_samples)} samples")

    progress_bar.empty()
    status_text.empty()


    # Create DataFrame
    results_df = pd.DataFrame(evaluation_data, index=range(len(conditional_samples)))

    return cond_evals, results_df


def generate_conditional_samples(
    model,
    molec: Molecule,
    batch_size: int = 5,
    n_atoms: int = 35,
    condition_types: list | None = None,
    num_pharmacophores: int = 10,
    inpaint_atom_indices: list | None = None,
    stop_inpainting_at_time_x1_pos: float | None = None,
    stop_inpainting_at_time_x1_x: float | None = None,
    exit_vector_indices: list | None = None,
    intermediate_time: bool = False,
    start_time: float = 0.0,
    new_atom_placement_region: np.ndarray | None = None,
    new_atom_placement_radius: float = 1.5,
    num_sampling_steps: int = 400,
):
    """
    Generate conditional samples using shepherd.sampler.generate function.
    This function prepares the conditioning information and calls the generate function.

    Args
    ----
    model : PyTorch Lightning model
    molec : target molecule (Molecule object)
    batch_size : number of samples to generate
    n_atoms : number of atoms in generated molecules
    condition_types : list of conditioning types ['shape', 'electrostatics', 'pharmacophores']
    num_pharmacophores : number of pharmacophores to condition on
    inpaint_atom_indices : list of atom indices to inpaint
    stop_inpainting_at_time_x1 : time to stop inpainting at for x1
    intermediate_time : whether to use intermediate time sampling
    start_time : time to start sampling from
    new_atom_placement_region : cluster center for new atoms (3,)
    new_atom_placement_radius : radius of cluster for new atoms (float)

    Returns
    -------
    generated_samples : list of generated samples
    """

    if condition_types is None:
        condition_types = ['shape', 'electrostatics', 'pharmacophores']

    # Prepare conditioning parameters
    conditioning_params = {}

    # Shape conditioning (x3)
    if (('shape' in condition_types and 'electrostatics' in condition_types) or ('electrostatics' in condition_types)):
        conditioning_params.update({
            'inpaint_x3_pos': True,
            'inpaint_x3_x': True,
            'stop_inpainting_at_time_x3': 0.0,
            'add_noise_to_inpainted_x3_pos': 0.0,
            'add_noise_to_inpainted_x3_x': 0.0,
        })
    
    elif 'shape' in condition_types and 'electrostatics' not in condition_types:
        conditioning_params.update({
            'inpaint_x2_pos': True,
            'inpaint_x2_direction': True,
            'stop_inpainting_at_time_x2': 0.0,
            'add_noise_to_inpainted_x2_pos': 0.0,
            'add_noise_to_inpainted_x2_direction': 0.0,
        })

    # Pharmacophores conditioning (x4)
    if 'pharmacophores' in condition_types:
        conditioning_params.update({
            'inpaint_x4_pos': True,
            'inpaint_x4_direction': True,
            'inpaint_x4_type': True,
            'stop_inpainting_at_time_x4': 0.0,
            'add_noise_to_inpainted_x4_pos': 0.0,
            'add_noise_to_inpainted_x4_direction': 0.0,
            'add_noise_to_inpainted_x4_type': 0.0,
        })

    if stop_inpainting_at_time_x1_pos is None and stop_inpainting_at_time_x1_x is None:
        stop_inpainting_at_time_x1_bonds = 0.0
    else:
        if stop_inpainting_at_time_x1_pos is not None:
            stop_inpainting_at_time_x1_bonds = stop_inpainting_at_time_x1_pos
        if stop_inpainting_at_time_x1_x is not None:
            if stop_inpainting_at_time_x1_pos is not None:
                stop_inpainting_at_time_x1_bonds = max(stop_inpainting_at_time_x1_pos, stop_inpainting_at_time_x1_x)
            else:
                stop_inpainting_at_time_x1_bonds = stop_inpainting_at_time_x1_x

    if inpaint_atom_indices is not None:
        atom_conditioning_params = {
            'inpaint_x1_bonds': True,
            'mol' : molec.mol,
            'exit_vector_atom_inds' : exit_vector_indices,
            'atom_inds_to_inpaint' : inpaint_atom_indices,
            'stop_inpainting_at_time_x1_pos': 0.0 if stop_inpainting_at_time_x1_pos is None else stop_inpainting_at_time_x1_pos,
            'stop_inpainting_at_time_x1_x': 0.0 if stop_inpainting_at_time_x1_x is None else stop_inpainting_at_time_x1_x,
            'stop_inpainting_at_time_x1_bonds': stop_inpainting_at_time_x1_bonds,
        }
    else:
        atom_conditioning_params = {}

    # Set number of pharmacophores
    if len(molec.pharm_types) > num_pharmacophores:
        st.warning(f"Number of pharmacophores is greater than the number of pharmacophores in the target molecule. Using {len(molec.pharm_types)} pharmacophores.")
        num_pharmacophores = len(molec.pharm_types)

    if not intermediate_time:
        # Only set inpainting flags if we're actually doing atom inpainting
        if inpaint_atom_indices is not None:
            atom_conditioning_params.update({
                'inpaint_x1_pos': True,
                'inpaint_x1_x': True,
            })
        generated_samples = generate(
            model_pl=model,
            batch_size=batch_size,
            N_x1=n_atoms,
            N_x4=num_pharmacophores,
            unconditional=False,
            prior_noise_scale=1.0,
            denoising_noise_scale=1.0,
            # Noise injection parameters
            inject_noise_at_ts=[],
            inject_noise_scales=[],
            # Harmonization parameters
            harmonize=False,
            harmonize_ts=[],
            harmonize_jumps=[],
            # Conditioning parameters
            **conditioning_params,
            **atom_conditioning_params,
            # Conditioning targets
            center_of_mass=np.zeros(3),
            surface=molec.surf_pos,
            electrostatics=molec.surf_esp,
            pharm_types=molec.pharm_types,
            pharm_pos=molec.pharm_ancs,
            pharm_direction=molec.pharm_vecs,
            verbose=True,
            num_steps=num_sampling_steps,
            # Add interruption callback
            interruption_callback=lambda: st.session_state.get('stop_generation', False),
        )

    else:
        generated_samples = generate_from_intermediate_time(
            model_pl=model,
            batch_size=batch_size,
            N_x1=n_atoms,
            N_x4=num_pharmacophores,
            start_time=start_time,
            new_atom_placement_region=new_atom_placement_region,
            new_atom_placement_radius=new_atom_placement_radius,
            **atom_conditioning_params,
            # Conditioning targets
            center_of_mass=np.zeros(3),
            surface=molec.surf_pos,
            electrostatics=molec.surf_esp,
            pharm_types=molec.pharm_types,
            pharm_pos=molec.pharm_ancs,
            pharm_direction=molec.pharm_vecs,
            verbose=True,
            # Add interruption callback
            interruption_callback=lambda: st.session_state.get('stop_generation', False),
        )

    return generated_samples


def create_xyz_content(cond_evals: ConditionalEval):
    """
    Create XYZ content from a list of molecules.

    Args:
        cond_evals: List of ConditionalEval objects

    Returns:
        XYZ content as string
    """
    xyz_content = ''
    for cond_eval in cond_evals:
        if cond_eval is not None and cond_eval.xyz_block is not None:
            xyz_content += f"{cond_eval.xyz_block}\n"
    return xyz_content


def create_sdf_content(molecules, properties_list=None):
    """
    Create SDF content from a list of molecules.

    Args:
        molecules: List of RDKit molecules
        properties_list: List of dictionaries containing properties for each molecule

    Returns:
        SDF content as string
    """
    if properties_list is None:
        properties_list = [{}] * len(molecules)

    sdf_content = StringIO()
    with Chem.SDWriter(sdf_content) as w:
        for i, (mol, props) in enumerate(zip(molecules, properties_list)):
            if mol is None:
                continue
            try:
                mol_copy = Chem.Mol(mol)

                for key, value in props.items():
                    if key == 'SMILES':
                        mol_copy.SetProp('_Name', value)
                    else:
                        mol_copy.SetProp(key, str(f'{value:.4f}'))
                w.write(mol_copy)

            except Exception as e:
                st.warning(f"Failed to write molecule {i} to SDF: {str(e)}")
                continue

    return sdf_content.getvalue()


def extract_valid_post_opt_molecules(cond_evals, results_df=None):
    """
    Extract valid molecules from conditional evaluations.

    Args:
        cond_evals: List of ConditionalEval objects
        results_df: Results dataframe with evaluation data

    Returns:
        Tuple of (valid_molecules, valid_properties)
    """
    valid_molecules = []
    valid_properties = []

    for i, cond_eval in enumerate(cond_evals):
        if cond_eval is not None and cond_eval.mol_post_opt is not None:
            # Extract properties from results_df if available
            props = {}
            if results_df is not None and i < len(results_df):
                row = results_df.iloc[i]
                props = {
                    'SMILES': row.get('SMILES (post-opt)', ''),
                    'Shape_Similarity': row.get('Shape Similarity', ''),
                    'ESP_Similarity': row.get('ESP Similarity', ''),
                    'Pharmacophore_Similarity': row.get('Pharm. Similarity', ''),
                    'Original_Index': i
                }

            valid_molecules.append(cond_eval.mol_post_opt)
            valid_properties.append(props)

    return valid_molecules, valid_properties


def mol_with_atom_index(mol, label: Literal['atomLabel', 'molAtomMapNumber', 'atomNote']='atomLabel'):
    mol_label = deepcopy(mol)
    for atom in mol_label.GetAtoms():
        atom.SetProp(label, str(atom.GetIdx()))
    return mol_label
