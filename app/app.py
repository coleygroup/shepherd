import streamlit as st
import open3d
import sys
from typing import Literal
import os
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import pickle
import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

from stmol import showmol

from shepherd.lightning_module import LightningModule
from shepherd.checkpoint_manager import get_checkpoint_path
from shepherd.extract import remove_overlaps

from shepherd_score.visualize import draw_molecule, draw_sample, draw, draw_2d_highlight, draw_2d_valid
from shepherd_score.container import Molecule
from shepherd_score.conformer_generation import embed_conformer_from_smiles
from shepherd_score.evaluations.utils.convert_data import extract_mol_from_xyz_block

# Constants and configuration
TMP_DIR = os.environ.get('TMPDIR', './')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, '../data')

# Add current directory to path for local imports
sys.path.append(THIS_DIR)
from utils import (
    get_conditioning_information,
    generate_conditional_samples,
    evaluate_conditional_samples,
    create_sdf_content,
    extract_valid_post_opt_molecules,
    create_xyz_content,
    mol_with_atom_index
)

# Matplotlib styling constants
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18


def configure_matplotlib():
    """Configure matplotlib styling for the app."""
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)


def configure_streamlit():
    """Configure Streamlit app settings."""
    st.set_page_config(
        page_title="ShEPhERD for Bioisosteric Design",
        page_icon="⌬",
        layout="wide"
    )

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    session_defaults = {
        'molec': None,
        'mol': None,
        'smiles': None,
        'generated_samples': None,
        'cond_evals': None,
        'results_df': None,
        'evaluation_results': [],
        'inpaint_atoms': None,
        'do_inpaint_atoms': False,
        'stop_inpainting_at_time_x1_pos': None,
        'stop_inpainting_at_time_x1_x': None,
        'exit_vector_indices': None,
        'inpaint_from_intermediate_time': False,
        'intermediate_start_time': 0.5,
        'new_atom_placement_region': None,
        'new_atom_placement_radius': 1.5,
        'stop_generation': False,
        'generation_in_progress': False,
        'num_sampling_steps': 400,
    }
    
    for key, default_value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Load model function (cached)
@st.cache_resource
def load_model(model_type: Literal['mosesaq', 'gdb_x2', 'gdb_x3', 'gdb_x4'] = 'mosesaq'):
    """Load the correct ShEPhERD model with automatic checkpoint downloading"""
    try:
        # Get checkpoint path - will download from HuggingFace if not found locally
        model_path = get_checkpoint_path(
            model_type=model_type,
            local_data_dir=os.path.join(DATA_DIR, 'shepherd_chkpts'),  # Check local directory first
            cache_dir=None, # fallback to HuggingFace cache
            force_download=False
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_pl = LightningModule.load_from_checkpoint(
            model_path,
            weights_only=True,
            map_location=device
        )
        model_pl.eval()
        model_pl.model.device = device
        return model_pl
        
    except Exception as e:
        st.error(f"Failed to load model {model_type}: {str(e)}")
        st.info("If this is your first time running the app, the model will be downloaded from Hugging Face.")
        raise


@st.cache_data
def load_moses_molblock_charges():
    """Load MOSES molecule block charges data."""
    moses_molblock_charges_path = os.path.join(DATA_DIR, 'conformers/moses_aq/example_molblock_charges.pkl')
    with open(moses_molblock_charges_path, 'rb') as f:
        moses_molblock_charges = pickle.load(f)
    return moses_molblock_charges


@st.cache_resource
def load_atom_pharm_counts():
    """Load atom pharmacophore counts data."""
    atom_pharm_counts_path = os.path.join(DATA_DIR, 'conformers/distributions/atom_pharm_count.npz')
    atom_pharm_counts = np.load(atom_pharm_counts_path)
    return atom_pharm_counts


def clear_session_state():
    """Clear all session state variables."""
    session_keys = [
        'molec', 'mol', 'smiles', 'generated_samples', 'cond_evals', 
        'results_df', 'evaluation_results', 'inpaint_atoms', 
        'stop_inpainting_at_time_x1_pos', 'stop_inpainting_at_time_x1_x', 
        'do_inpaint_atoms', 'exit_vector_indices', 'inpaint_from_intermediate_time',
        'intermediate_start_time', 'new_atom_placement_region', 'new_atom_placement_radius',
        'stop_generation', 'generation_in_progress', 'num_sampling_steps'
    ]
    
    for key in session_keys:
        if key in st.session_state:
            st.session_state[key] = None if key != 'evaluation_results' else []
    
    # Reset specific values
    st.session_state.do_inpaint_atoms = False
    st.session_state.inpaint_from_intermediate_time = False
    st.session_state.intermediate_start_time = 0.3
    st.session_state.new_atom_placement_radius = 1.5
    st.session_state.stop_generation = False
    st.session_state.generation_in_progress = False


def main():
    """Main Streamlit application function."""
    # Configure the app
    configure_streamlit()
    configure_matplotlib()
    initialize_session_state()
    
    # App title and description
    st.title("ShEPhERD: Diffusing shape, electrostatics, and pharmacophores for bioisosteric drug design")
    st.markdown("##### Generate molecules with similar shape, electrostatics, and pharmacophores as a reference molecule")

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # Add a button to clear cache if needed
    if st.sidebar.button("Clear Model Cache"):
        st.cache_resource.clear()
        st.sidebar.success("Cache cleared! The app will reload the model on next generation.")

    if st.sidebar.button("Clear Session State"):
        clear_session_state()
        st.sidebar.success("Session state cleared!")
        st.rerun()

    # with st.sidebar:
    #     st.header("Session State")
    #     if st.session_state: # Check if session state is not empty
    #         for key, value in st.session_state.items():
    #             st.write(f"**{key}**: {value}")
    #     else:
    #         st.write("Session state is empty.")

    # Load example molecule data
    moses_molblock_charges = load_moses_molblock_charges()
    atom_pharm_counts = load_atom_pharm_counts()

    moses_mol = Chem.MolFromMolBlock(moses_molblock_charges[0][0], removeHs=False)
    moses_charges = np.array(moses_molblock_charges[0][1])


    ########################################################
    # Main app interface
    ########################################################
    col1, col2 = st.columns([2, 3])


    ########################################################
    # Target molecule input
    ########################################################
    with col1:
        st.header("Target Molecule Input")

        input_type = st.radio("Input Type", ["SMILES", "MolBlock", "XYZ", "MOSES"], index=1,
                            captions=['', 'Preferred', '', 'test set'],
                            horizontal=True)

        optimize_me = st.checkbox("Optimize Geometry ", value=False,
                                help='Local optimization of provided structure with xTB')

        # SMILES input
        if input_type == "SMILES":
            smiles = st.text_input("Enter SMILES", f"{Chem.MolToSmiles(Chem.RemoveHs(moses_mol))}", help="Enter a valid SMILES string")
            mol = embed_conformer_from_smiles(smiles, MMFF_optimize=True)

        elif input_type == "MolBlock":
            molblock = st.text_area("Enter MolBlock", Chem.MolToMolBlock(moses_mol), help="Enter a valid MolBlock string that contains 3D geometry")
            mol = Chem.MolFromMolBlock(molblock, removeHs=False)
            smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        elif input_type == "XYZ":
            xyz = st.text_area("Enter XYZ", Chem.MolToXYZBlock(moses_mol), help="Enter a valid XYZ string")
            template_smiles = st.text_input("Enter Template SMILES for bond order assignment (particularly if no hydrogens are in xyz)",
                                            f"{Chem.MolToSmiles(Chem.RemoveHs(moses_mol))}", help="Enter a valid SMILES string")
            charge = st.number_input("Enter Charge", value=0, min_value=-2, max_value=2, step=1)
            mol = extract_mol_from_xyz_block(xyz, charge=charge)

            if template_smiles:
                template = Chem.MolFromSmiles(template_smiles)
                mol = Chem.AddHs(AllChem.AssignBondOrdersFromTemplate(template, mol), addCoords=True)

            if mol is not None:
                smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
            else:
                st.error("Invalid XYZ string")
        # elif input_type == "PDB":
        #     st.error("PDB input not supported yet")
        #     # smiles = st.text_input("Enter Template SMILES", f"{Chem.MolToSmiles(Chem.RemoveHs(moses_mol))}", help="Enter a valid SMILES string")
        #     # pdb_code = st.text_input("Enter PDB Code", 'PDB_CODE', help="Enter a valid PDB code")
        elif input_type == "MOSES":
            ind = st.number_input("Enter index", min_value=0, max_value=len(moses_molblock_charges) - 1, value=0, step=1)
            _moses_mol = Chem.MolFromMolBlock(moses_molblock_charges[ind][0], removeHs=False)
            _moses_charges = np.array(moses_molblock_charges[ind][1])
            mol = _moses_mol
            smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        st.session_state.mol = mol
        st.session_state.smiles = smiles
        
        probe_radius = st.slider("Probe Radius (ShEPhERD is trained with 0.6)", min_value=0.4, max_value=1.2, value=0.6, step=0.1,
                                help="Surface probe radius. 0.6 is default since ShEPhERD was trained with this radius.")

        xtb_button = st.button("Get Interaction Profile", type="primary", help="Runs an xTB single point for charges (and optimize if selected)")

        if xtb_button:
            with st.spinner(f"Running xTB{' (and optimizing)' if optimize_me else ''}..."):
                molec = get_conditioning_information(st.session_state.mol, optimize=optimize_me, probe_radius=probe_radius)
                st.session_state.molec = molec  # Store in session state
                # Update the mol with the potentially modified molecule from molec
                st.session_state.mol = molec.mol  # Store the updated mol (coordinates may have changed)
                # Also store the SMILES from the updated molecule
                st.session_state.smiles = Chem.MolToSmiles(Chem.RemoveHs(molec.mol))
                st.success(f"xTB run complete!{' (and optimized)' if optimize_me else ''}")


        ########################################################
        # Generation parameters
        ########################################################
        st.header("Generation parameters")

        col3, col4 = st.columns([1, 2])

        with col3:
            model_type = st.radio("Model", ["MOSES-aq", "GDB-x2", "GDB-x3", "GDB-x4"], index=0)

        with col4:
            if model_type == "MOSES-aq":
                condition_type = st.multiselect("Conditioning Type", ["Shape (x2)", "Electrostatics (x3)", "Pharmacophores (x4)"],
                                                default=["Shape (x2)", "Electrostatics (x3)", "Pharmacophores (x4)"])
            elif model_type == "GDB-x2":
                condition_type = st.multiselect("Conditioning Type", ["Shape (x2)"],
                                                default=["Shape (x2)"])
            elif model_type == "GDB-x3":
                condition_type = st.multiselect("Conditioning Type", ["Shape (x2)", "Electrostatics (x3)"],
                                                default=["Shape (x2)", "Electrostatics (x3)"])
            elif model_type == "GDB-x4":
                condition_type = st.multiselect("Conditioning Type", ["Pharmacophores (x4)"],
                                                default=["Pharmacophores (x4)"])
            else:
                st.error("Invalid model type")
                st.stop()

        # Number of conformers
        batch_size = st.slider("Batch size", min_value=1, max_value=20, value=10,
                            help="Number of molecules to generate in a batch. Max is dependent on your GPU memory and roughly the number of atoms.")

        # Number of atoms
        atom_help = "Number of atoms to use for the generation. Typically ±5 of the reference molecule is reasonable to fit within the specified shape."
        if 'molec' in st.session_state and st.session_state.molec is not None:
            n_atoms = st.slider("Number of atoms", min_value=10, max_value=90,
                                value=st.session_state.molec.mol.GetNumAtoms(),
                                help=atom_help)
        else:
            n_atoms = st.slider("Number of atoms", min_value=10, max_value=90,
                                value=st.session_state.mol.GetNumAtoms(), help=atom_help)

        # Number of pharmacophores
        pharm_help = "Number of pharmacophores to use for the generation. This is based on the number of pharmacophores in the reference molecule."
        if 'molec' in st.session_state and st.session_state.molec is not None:
            num_pharmacophores = st.slider("Number of pharmacophores",
                                        min_value=len(st.session_state.molec.pharm_types),
                                        max_value=25,
                                        value=len(st.session_state.molec.pharm_types),
                                        help=pharm_help)
        else:
            num_pharmacophores = st.slider("Number of pharmacophores", min_value=5,
                                        max_value=25, value=10, help=pharm_help)

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.session_state.do_inpaint_atoms:
                num_sampling_steps = st.number_input("Number of sampling steps (trained for 400)", min_value=100, max_value=400, value=400, step=50,
                    help="Number of total time steps to run the sampler for. ShEPhERD was trained with 400, but is can still yield decent results with 200 (2x faster). Fixed at 400 for atom inpainting.",
                    disabled=True)
            else:
                num_sampling_steps = st.number_input("Number of sampling steps (trained for 400)", min_value=100, max_value=400, value=400, step=50,
                    help="Number of total time steps to run the sampler for. ShEPhERD was trained with 400, but is can still yield decent results with 200 (2x faster).")
            st.session_state.num_sampling_steps = num_sampling_steps

        # Show different UI based on generation state
        if not st.session_state.generation_in_progress:
            # Normal state - show generate button
            generate_button = st.button("Generate", type="primary", help="Generates a batch of molecules with ShEPhERD")
        else:
            # Generation in progress - show disabled generate and stop button
            st.button("Generate", type="primary", help="Generation in progress...", disabled=True)
            col_stop1, col_stop2 = st.columns([1, 3])
            with col_stop1:
                if st.button("🛑 Stop", type="secondary", help="Stop the current generation process"):
                    st.session_state.stop_generation = True
                    st.rerun()
            with col_stop2:
                st.info("Generation in progress... Click stop to interrupt.")
            generate_button = False  # Ensure we don't trigger generation logic
        
        st.session_state.do_inpaint_atoms = st.checkbox("Inpaint Atoms (beta)", value=False, help="Inpaint the selected atoms (still in development)", disabled=st.session_state.inpaint_atoms is None)

    ########################################################
    # Visualization
    ########################################################
    with col2:
        st.header("Visualization")
        
        # center these ontop of each other
        row1 = st.container()
        with row1:
            # Use session state smiles if available, otherwise use the local smiles variable
            display_smiles = st.session_state.smiles if st.session_state.smiles else smiles
            if display_smiles:
                # Show 2D structure
                try:
                    mol_2d = Chem.MolFromSmiles(display_smiles)
                    if mol_2d:
                        img = Draw.MolToImage(mol_2d, size=(400, 250))
                        st.image(img, caption="2D Structure")
                    else:
                        st.error("Invalid SMILES string")
                except Exception as e:
                    st.error(f"Error drawing 2D structure: {str(e)}")

        row2 = st.container()
        with row2:
            if 'molec' in st.session_state and st.session_state.molec is not None:
                try:
                    view = draw_molecule(st.session_state.molec, height=400, width=600, custom_carbon_color='light steel blue')
                    showmol(view, height=400, width=600)
                    st.markdown(
                        "<p style='text-align: center; color: grey;'>3D Structure with Interaction Profile</p>",
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Error drawing molecule: {str(e)}")
            else:
                st.markdown("<div style='height: 400px;'></div>", unsafe_allow_html=True)

        row3 = st.container()
        with row3:
            if model_type == "MOSES-aq":
                values = atom_pharm_counts['moses_aq'][:90, :40]
            elif model_type.startswith("GDB"):
                values = atom_pharm_counts['gdb'][:90, :40]
            else:
                st.error("Invalid model type")
                st.stop()
            
            @st.cache_resource
            def plot_joint_distribution(current_n_atoms, current_n_pharmacophores, model_type, values,
                                        target_n_atoms, target_n_pharmacophores):
                df = pd.DataFrame({
                    'Atom Count': np.repeat(np.arange(values.shape[0]), values.shape[1]),
                    'Pharmacophore Count': np.tile(np.arange(values.shape[1]), values.shape[0]),
                    'Value': values.flatten()
                })
                df = df[df['Value'] > 0]
                joint_plot = sns.jointplot( data=df, x='Atom Count', y='Pharmacophore Count', cmap="Blues", kind='kde', fill=False
                )
                ax_joint = joint_plot.ax_joint

                scatter = sns.scatterplot(x=[current_n_atoms], y=[current_n_pharmacophores], marker="*",
                                        zorder=6, color='firebrick', s=300, label='Current settings', ax=ax_joint)
                if target_n_atoms is not None and target_n_pharmacophores is not None:
                    sns.scatterplot(x=[target_n_atoms], y=[target_n_pharmacophores],
                                                    zorder=5, color='olivedrab', s=300, label='Target settings', ax=ax_joint)

                handles, labels = scatter.get_legend_handles_labels()
                contour_handle = mlines.Line2D([], [], color='tab:blue', linestyle='-', label='KDE contours')
                handles.append(contour_handle)
                labels.append(f"{model_type.split('-')[0]} dataset distribution")

                ax_joint.legend(handles, labels, loc='center left', fontsize='small', frameon=True)
                sns.move_legend(ax_joint, "upper right", bbox_to_anchor=(2.15, 1))

                return joint_plot

            st.pyplot(plot_joint_distribution(n_atoms, num_pharmacophores, model_type, values,
                                            st.session_state.mol.GetNumAtoms(),
                                            len(st.session_state.molec.pharm_types) if st.session_state.molec else None))

    ########################################################
    # Inpainting settings
    ########################################################

    with st.expander("Atom inpainting settings"):
        st.write("Settings used for granular control over atom inpainting")

        col10, col11 = st.columns([3, 1])

        with col10:
            atom_indices_to_inpaint = st.text_input("Atom indices to inpaint (comma separated)", value="", help="Enter a list of atom indices to inpaint. Use commas to separate indices.")
            if atom_indices_to_inpaint:
                try:
                    atom_indices_to_inpaint = [int(i) for i in atom_indices_to_inpaint.split(',')]
                    st.write(f"Atom indices to inpaint: {atom_indices_to_inpaint}")
                    st.session_state.inpaint_atoms = atom_indices_to_inpaint
                except Exception as e:
                    st.error(f"Error parsing atom indices: {str(e)}")
            else:
                st.write("No atom indices to inpaint")
                st.session_state.inpaint_atoms = None

            st.markdown("---")
            exit_vector_indices = st.text_input("Atom indices to specify exit vector (comma separated) [changes bond-inpainting]",
            value="",
            help="Enter a list of atom indices as exit vectors. Use commas to separate indices.", disabled=st.session_state.inpaint_atoms is None)
            if exit_vector_indices:
                try:
                    exit_vector_indices = [int(i) for i in exit_vector_indices.split(',')]
                    st.write(f"Exit vector indices: {exit_vector_indices}")
                    if not all(i in atom_indices_to_inpaint for i in exit_vector_indices):
                        st.error("Exit vector indices must be a subset of the atom indices to inpaint")
                    st.session_state.exit_vector_indices = exit_vector_indices
                except Exception as e:
                    st.error(f"Error parsing exit vector indices: {str(e)}")
            else:
                st.write("No exit vector indices")
                st.session_state.exit_vector_indices = None

            # Intermediate time inpainting section
            st.markdown("---")
            inpaint_from_intermediate_time = st.checkbox(
                "Inpaint from intermediate time-point", 
                value=st.session_state.inpaint_from_intermediate_time,
                help="Start inpainting from an intermediate time point in the diffusion process",
                disabled=st.session_state.inpaint_atoms is None
            )
            st.session_state.inpaint_from_intermediate_time = inpaint_from_intermediate_time

            if st.session_state.inpaint_from_intermediate_time and st.session_state.inpaint_atoms is not None:
                if model_type != "MOSES-aq":
                    st.error("Intermediate time inpainting is only supported for MOSES-aq")
                    st.stop()

                # Start time input
                intermediate_start_time = st.slider(
                    "Start time for intermediate inpainting", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=st.session_state.intermediate_start_time, 
                    step=0.01,
                    help="Time point to start inpainting from (0.0 = end, 1.0 = beginning of diffusion process)"
                )
                st.session_state.intermediate_start_time = intermediate_start_time
                
                # Atom placement region (atom index)
                if st.session_state.mol is not None:
                    max_atom_idx = st.session_state.mol.GetNumAtoms() - 1
                    new_atom_placement_region_idx = st.number_input(
                        "Atom index for new atom placement region center", 
                        min_value=0, 
                        max_value=max_atom_idx, 
                        value=0, 
                        step=1,
                        help="Index of the atom to use as center for placing new atoms. We suggest using an atom that is not inpainted."
                    )
                    # Convert atom index to coordinates
                    if st.session_state.mol is not None:
                        atom_coords = st.session_state.mol.GetConformer().GetPositions()[new_atom_placement_region_idx]
                        st.session_state.new_atom_placement_region = atom_coords
                        st.write(f"Using atom {new_atom_placement_region_idx} coordinates: [{atom_coords[0]:.2f}, {atom_coords[1]:.2f}, {atom_coords[2]:.2f}]")
                
                # Placement radius
                new_atom_placement_radius = st.number_input(
                    "New atom placement radius (Angstroms)", 
                    min_value=0.1, 
                    max_value=10.0, 
                    value=st.session_state.new_atom_placement_radius, 
                    step=0.1,
                    help="Radius in Angstroms around the placement region center"
                )
                st.session_state.new_atom_placement_radius = new_atom_placement_radius
            else:
                # Reset values when not using intermediate time inpainting
                st.session_state.new_atom_placement_region = None

        with col11:
            stop_inpainting_at_time_x1_pos = st.number_input("Time to stop inpainting (atom positions)", min_value=0.0, max_value=1.0, value=0.0, step=0.001,
            help="Time to stop inpainting at for x1. 0.0 is the end of the trajectory. There are 400 steps in the trajectory.")
            if stop_inpainting_at_time_x1_pos:
                st.session_state.stop_inpainting_at_time_x1_pos = stop_inpainting_at_time_x1_pos
            else:
                st.session_state.stop_inpainting_at_time_x1_pos = None

            stop_inpainting_at_time_x1_x = st.number_input("Time to stop inpainting (atom types)", min_value=0.0, max_value=1.0, value=0.0, step=0.001,
            help="Time to stop inpainting at for x1. 0.0 is the end of the trajectory. There are 400 steps in the trajectory.")
            if stop_inpainting_at_time_x1_x:
                st.session_state.stop_inpainting_at_time_x1_x = stop_inpainting_at_time_x1_x
            else:
                st.session_state.stop_inpainting_at_time_x1_x = None

        if st.session_state.mol is not None:
            atom_sets = []
            if st.session_state.inpaint_atoms is not None:
                atom_sets.append(st.session_state.inpaint_atoms)
            if st.session_state.exit_vector_indices is not None:
                atom_sets.append(st.session_state.exit_vector_indices)
            if st.session_state.inpaint_from_intermediate_time and st.session_state.inpaint_atoms is not None:
                atom_sets.append([new_atom_placement_region_idx])

            st.image(draw_2d_highlight(
                st.session_state.mol,
                atom_sets=atom_sets,
                label='atomLabel',
                width=600,
                height=400,
                embed_display=False
            ))

        


    ########################################################
    # Conditional Generation
    ########################################################

    # Handle generation start
    if generate_button:
        if 'molec' not in st.session_state or st.session_state.molec is None:
            st.error("Please run xTB first to get the interaction profile")
            st.stop()

        # Set generation state and rerun to show stop button
        st.session_state.stop_generation = False
        st.session_state.generation_in_progress = True
        st.rerun()
    
    # Handle stop request
    if st.session_state.generation_in_progress and st.session_state.stop_generation:
        st.warning("Generation stopped by user")
        st.session_state.generation_in_progress = False
        st.session_state.stop_generation = False
        st.rerun()
    
    # Handle generation execution (when in progress and not stopped)
    if st.session_state.generation_in_progress and not st.session_state.stop_generation:
        try:
            with st.spinner("Loading ShEPhERD model..."):
                # Load the appropriate model
                if model_type == "MOSES-aq":
                    loaded_model = load_model('mosesaq')
                elif model_type == "GDB-x2":
                    loaded_model = load_model('gdb_x2')
                elif model_type == "GDB-x3":
                    loaded_model = load_model('gdb_x3')
                elif model_type == "GDB-x4":
                    loaded_model = load_model('gdb_x4')
                else:
                    st.error(f"Unknown model type: {model_type}")
                    st.stop()

            # Convert condition_type selections to the format expected by the generation function
            condition_types_map = {
                "Shape (x2)": "shape",
                "Electrostatics (x3)": "electrostatics",
                "Pharmacophores (x4)": "pharmacophores"
            }
            selected_condition_types = [condition_types_map[ct] for ct in condition_type]

            if st.session_state.inpaint_atoms is not None and st.session_state.do_inpaint_atoms:
                _selected_condition_types = selected_condition_types + ['atoms', 'bonds']
            else:
                _selected_condition_types = selected_condition_types

            with st.spinner(f"Generating {batch_size} samples by inpainting {', '.join(_selected_condition_types)}..."):
                start_time = time.time()
                # Generate conditional samples
                generated_samples = generate_conditional_samples(
                    model=loaded_model,
                    molec=st.session_state.molec,
                    batch_size=batch_size,
                    n_atoms=n_atoms,
                    condition_types=selected_condition_types,
                    num_pharmacophores=num_pharmacophores,
                    inpaint_atom_indices=st.session_state.inpaint_atoms if st.session_state.inpaint_atoms is not None and st.session_state.do_inpaint_atoms else None,
                    exit_vector_indices=st.session_state.exit_vector_indices if st.session_state.exit_vector_indices is not None else None,
                    stop_inpainting_at_time_x1_pos=st.session_state.stop_inpainting_at_time_x1_pos if st.session_state.stop_inpainting_at_time_x1_pos is not None else None,
                    stop_inpainting_at_time_x1_x=st.session_state.stop_inpainting_at_time_x1_x if st.session_state.stop_inpainting_at_time_x1_x is not None else None,
                    intermediate_time=st.session_state.inpaint_from_intermediate_time and st.session_state.inpaint_atoms is not None and st.session_state.do_inpaint_atoms,
                    start_time=st.session_state.intermediate_start_time if st.session_state.inpaint_from_intermediate_time else 0.0,
                    new_atom_placement_region=st.session_state.new_atom_placement_region if st.session_state.inpaint_from_intermediate_time else None,
                    new_atom_placement_radius=st.session_state.new_atom_placement_radius if st.session_state.inpaint_from_intermediate_time else 1.5,
                    num_sampling_steps=st.session_state.num_sampling_steps
                )
                end_time = time.time()
            
            # Check if generation was stopped
            if st.session_state.stop_generation:
                st.warning("Generation was stopped by user")
                st.session_state.generated_samples = None
            elif len(generated_samples) == 0:
                st.warning("No samples were generated (possibly stopped early)")
                st.session_state.generated_samples = None
            else:
                st.success(f"Generated {len(generated_samples)} samples in {end_time - start_time:.2f} seconds")
                st.session_state.generated_samples = generated_samples

            # Evaluate the generated samples (only if we have samples)
            if st.session_state.generated_samples is not None and len(st.session_state.generated_samples) > 0:
                with st.spinner("Evaluating generated samples..."):
                    if st.session_state.inpaint_atoms is not None and st.session_state.do_inpaint_atoms:
                        new_generated_samples = []
                        for generated_sample in st.session_state.generated_samples:
                            new_generated_samples.append(
                                remove_overlaps(generated_sample,
                                                st.session_state.molec.mol.GetConformer().GetPositions()[st.session_state.inpaint_atoms],
                                                cutoff=0.7)
                            )
                        st.session_state.generated_samples = new_generated_samples
                    cond_evals, results_df = evaluate_conditional_samples(st.session_state.generated_samples, st.session_state.molec)
                    st.session_state.cond_evals = cond_evals
                    st.session_state.results_df = results_df

                # Debug: Show error information if available
                if st.session_state.results_df is not None and 'Error' in st.session_state.results_df.columns and st.session_state.results_df['Error'].notna().any():
                    with st.expander("🔍 Debug: Show evaluation errors"):
                        st.write("Found errors during evaluation:")
                        error_rows = st.session_state.results_df[st.session_state.results_df['Error'].notna()]
                        for idx, row in error_rows.iterrows():
                            st.error(f"Sample {idx}: {str(row['Error'])[:200]}...")

        except Exception as e:
            st.error(f"Error during generation: {str(e)}")
            st.exception(e)
        finally:
            # Always reset generation state when done
            st.session_state.generation_in_progress = False
            st.session_state.stop_generation = False

    ########################################################
    # Results section - Conditional Generation
    ########################################################

    if st.session_state.generated_samples is not None:
        st.subheader("Generated Molecules")
        # Display evaluation results if available
        if st.session_state.results_df is not None:
            st.markdown("####  Evaluation Results")

            # Summary statistics
            valid_count = st.session_state.results_df['Valid'].sum()
            valid_post_opt_count = st.session_state.results_df['Valid (post-opt)'].sum()
            total_samples = len(st.session_state.results_df)

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("Total Samples", total_samples)
            with col2:
                st.metric("Valid Molecules", f"{valid_count} ({valid_count/total_samples*100:.1f}%)")
            with col3:
                st.metric("Valid (post-opt)", f"{valid_post_opt_count} ({valid_post_opt_count/total_samples*100:.1f}%)")
            with col4:
                st.metric("Best Shape Sim", f"{st.session_state.results_df['Shape Similarity'].max():.2f}")
            with col5:
                st.metric("Best ESP Sim", f"{st.session_state.results_df['ESP Similarity'].max():.2f}")
            with col6:
                st.metric("Best Pharm Sim", f"{st.session_state.results_df['Pharm. Similarity'].max():.2f}")

            # Display the results table
            st.dataframe(st.session_state.results_df, width='stretch')

        if st.session_state.cond_evals is not None:
            try:
                st.image(draw_2d_valid(
                    ref_mol=st.session_state.molec.mol,
                    mols=[cond_eval.mol_post_opt for cond_eval in st.session_state.cond_evals],
                    mols_per_row=5,
                    use_svg=True,
                    find_atomic_overlap=False,
                ))
            except Exception as e:
                st.error(f"Error drawing 2D valid molecules: {str(e)}")

        st.markdown("####  Sample Visualization")
        st.text("(Overlayed on greyed-out reference molecule)")
        use_sample = st.radio("Select a sample", options=["Sampled", "xTB-optimized"], index=0, horizontal=True,
                            captions=["All outputs directly from ShEPhERD", "Molecule post-relaxation\nPrior to alignment so similarity scores are not applicable."])
        sample_idx = st.number_input("Select a sample", min_value=0, max_value=len(st.session_state.generated_samples) - 1, value=0, step=1)
        if use_sample == "Sampled":
            if model_type == "MOSES-aq":
                model_type_ = "all"
            elif model_type == "GDB-x2":
                model_type_ = "x2"
            elif model_type == "GDB-x3":
                model_type_ = "x3"
            elif model_type == "GDB-x4":
                model_type_ = "x4"
            view = draw_sample(st.session_state.generated_samples[sample_idx],
                               height=400, width=600, custom_carbon_color='dark slate grey', model_type=model_type_)
            # Add reference molecule as overlay (model 1) with transparency
            if st.session_state.molec and st.session_state.molec.mol:
                try:
                    view = draw(st.session_state.molec.mol, height=400, width=600, custom_carbon_color='light steel blue', opacity=0.6, view=view)
                except Exception as e:
                    st.warning(f"Could not overlay reference molecule: {str(e)}")
            showmol(view, height=400, width=600)

        elif use_sample == "xTB-optimized":
            if st.session_state.cond_evals[sample_idx].mol_post_opt is None:
                st.error("No passing xTB-optimized molecule found")
            else:
                view = draw(st.session_state.cond_evals[sample_idx].mol_post_opt, height=400, width=600, custom_carbon_color='dark slate grey')
                # Add reference molecule as overlay (model 1) with transparency
                if st.session_state.molec and st.session_state.molec.mol:
                    try:
                        view = draw(st.session_state.molec.mol, height=400, width=600, custom_carbon_color='light steel blue', opacity=0.6, view=view)
                    except Exception as e:
                        st.warning(f"Could not overlay reference molecule: {str(e)}")
                showmol(view, height=400, width=600)

        # SDF Download section
        if st.session_state.cond_evals is not None:
            st.subheader("Download Molecules")

            col_download, col_download_valid = st.columns(2)

            with col_download:
                try:
                    st.download_button(
                        label=f"📥 Download All ShEPhERD Ouput XYZs",
                        data=create_xyz_content(st.session_state.cond_evals),
                        file_name='shepherd_samples.xyz',
                        mime="chemical/x-xyz",
                        help="Download all ShEPhERD output molecules as xyz files."
                    )
                except Exception as e:
                    st.error(f"Error creating XYZ content: {str(e)}")

            # Extract valid molecules
            valid_mols, valid_props = extract_valid_post_opt_molecules(
                st.session_state.cond_evals,
                st.session_state.results_df
            )
            with col_download_valid:
                if valid_mols:
                    # Download all valid molecules
                    sdf_content = create_sdf_content(valid_mols, valid_props)
                    filename = 'shepherd_valid_samples.sdf'

                    st.download_button(
                        label=f"📥 Download All Valid Molecules ({len(valid_mols)})",
                        data=sdf_content,
                        file_name=filename,
                        mime="chemical/x-mdl-sdfile",
                        help="Download all valid post-xTB optimization molecules as sdf files with evaluation data."
                    )
                else:
                    st.info("No valid post-optimization molecules found to download.")


    # Footer
    st.markdown("---")
    st.markdown("**ShEPhERD**: Diffusing shape, electrostatics, and pharmacophores for bioisosteric drug design")


if __name__ == '__main__':
    main()