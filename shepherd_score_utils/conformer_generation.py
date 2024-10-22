"""
Copied from: https://github.com/coleygroup/shepherd-score/blob/main/shepherd_score/conformer_generation.py

Handles anything related to generating conformers with xTB or MMFF94.

Requires:
- xtb installation 
    - command-line access
    - https://xtb-docs.readthedocs.io/en/latest/setup.html
        - conda config --add channels conda-forge
        - conda install xtb
"""
import os
from copy import deepcopy
import re
import shutil
import subprocess
import time
from pathlib import Path
from tqdm import tqdm
import uuid
from typing import Optional

import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdDistGeom
from rdkit.Chem import rdMolAlign
from rdkit.ML.Cluster import Butina


def update_mol_coordinates(mol: rdkit.Chem.Mol, coordinates) -> rdkit.Chem.Mol:
    """
    Updates the coordinates of a 3D RDKit mol object with a new set of coordinates
    
    Args:
        mol -- RDKit mol object with 3D coordinates to be replaced
        coordinates -- list/array of new [x,y,z] coordinates
    
    Returns:
        mol_new -- RDKit mol object with updated 3D coordinates
    """
    mol_new = deepcopy(mol)
    conf = mol_new.GetConformer()
    for i in range(mol_new.GetNumAtoms()):
        x,y,z = coordinates[i]
        conf.SetAtomPosition(i, Point3D(x,y,z))
    return mol_new

def read_multi_xyz_file(file_dir: str):
    """
    Reads an xyz file that potentially contains multiple structures
    
    Args:
        file_dir -- (str) path to .xyz file
    
    Returns:
        all_coordinates -- list of lists containing the coordinates of each structure in the xyz file
        all_elements -- list of lists containing the element types of each atom in each structure
    """
    atom_types = [rdkit.Chem.GetPeriodicTable().GetElementSymbol(i) for i in range(1, 119)]
    with open(file_dir, 'r') as file:
        lines = file.readlines()
        all_coordinates = []
        all_elements = []

        begin_coord = False
        coordinates = []
        elements = []
        for line in lines:
            stripped_line = (' '.join(line.split()))
            stripped_line_split = stripped_line.split(' ')
            numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", stripped_line)
            if len(stripped_line) == 0:
                continue

            if (len(numbers) == 3) & (stripped_line_split[0] in atom_types):
                begin_coord = True
                x,y,z = numbers
                coordinates.append([float(x),float(y),float(z)])
                elements.append(stripped_line_split[0])
            else:
                if begin_coord:
                    all_coordinates.append(coordinates)
                    all_elements.append(elements)
                    elements = []
                    coordinates = []
                begin_coord = False

    if begin_coord:
        all_coordinates.append(coordinates)
        all_elements.append(elements)

    return all_coordinates, all_elements


def embed_conformer(mol: rdkit.Chem.Mol, attempts: int=50, MMFF_optimize: bool=False):
    """
    Embeds a mol object into a 3D RDKit mol object with ETKDG (and optional MMFF94)

    Args:
        mol -- RDKit Mol object
        attempts -- (int) number of embedding attempts
        MMFF_optimize -- (bool) whether to optimize embedded conformer with MMFF94
        
    Returns:
        mol -- RDKit mol object with 3D coordinates
    """
    try:
        mol = rdkit.Chem.AddHs(mol)
        rdkit.Chem.AllChem.EmbedMolecule(mol, maxAttempts = attempts)
        if MMFF_optimize:
            rdkit.Chem.AllChem.MMFFOptimizeMolecule(mol)

        mol.GetConformer() # test whether conformer generation succeeded
    except Exception as e:
        return None

    return mol


def embed_conformer_from_smiles(smiles: str, attempts: int = 50, MMFF_optimize: bool = False):
    """
    Embeds a SMILES into a 3D RDKit mol object with ETKDG (and optionally MMFF94)
    
    Args:
        smiles -- SMILES string of molecule
        attempts -- (int) number of embedding attempts
        MMFF_optimize -- (bool) whether to optimize embedded conformer with MMFF94
        
    Returns:
        mol -- RDKit mol object with 3D coordinates
    """
    try:
        mol = rdkit.Chem.MolFromSmiles(smiles)
    except Exception as e:
        print('Error in SMILES string when embedding molecule:', e)
        return None
    mol = embed_conformer(mol, attempts, MMFF_optimize)
    return mol


def conf_to_mol(mol, conf_id):
    """
    Converts a conformer of a RDKit mol object into its own RDKit mol object
    
    Args:
        mol -- RDKit mol object with multiple conformers
        conf_id -- ID of conformer to be converted into its own mol object
    
    Returns:
        new_mol -- mol object with only 1 conformer (the selected conformer)
    """
    conf = mol.GetConformer(conf_id)
    new_mol = rdkit.Chem.Mol(mol)
    new_mol.RemoveAllConformers()
    new_mol.AddConformer(rdkit.Chem.Conformer(conf))
    return new_mol


def generate_conformer_ensemble(mol_3d: rdkit.Chem.Mol,
                                num_confs: int=100,
                                num_threads: int = 4,
                                threshold: float = 0.25,
                                num_opt_steps: int = 200):
    """
    Uses ETKDG algorithm to embed multiple (new) conformers from a given 3D conformer 'template'.
    Optionally optimizes each embedded conformer with MMFF94.
    
    Args:
        mol_3d -- RDKit mol object with 3D coordinates
        num_confs -- (int) maximum number of conformers to be embedded with ETKDG
        num_threads -- (int) number of processors to be used in parallel when embedding conformers
        threshold -- (float) RMSD threshold used to eliminate redundant conformers after ETKDG embedding
        num_opt_steps -- (int) number of MMFF94 optimization steps
    
    Returns:
        mols -- list of mol objects, each containing 1 (unique) conformer
    """
    mol_3d = deepcopy(mol_3d)

    cids = rdkit.Chem.AllChem.EmbedMultipleConfs(
        mol_3d,
        clearConfs=True,
        numConfs=num_confs,
        pruneRmsThresh = threshold,
        maxAttempts = 50,
        numThreads = num_threads,
    )

    if num_opt_steps > 0:
        for cid in cids:
            rdkit.Chem.AllChem.MMFFOptimizeMolecule(
                mol_3d,
                confId=cid,
                mmffVariant='MMFF94',
                maxIters = num_opt_steps,

            )
    mols = [conf_to_mol(mol_3d, c) for c in cids]

    return mols


def cluster_conformers_butina(conformers, threshold: float = 0.2, num_max_conformers: int = 100):
    """
    Clusters a list of conformers by their pairwise RMSD with Butina Clustering algorithm
    
    Args:
        conformers -- list of rdkit mol objects containing conformers of a common molecule to be clustered.
        threshold -- initial RMSD theshold for clustering
        num_max_conformers -- maximum number of conformers in the final clustered ensemble
    
    Returns: 
        select_confs -- a list (int) of the centroids of each cluster, to be indexed into conformers
    """
    dists = []
    for i, conformer_i in enumerate(conformers):
        for j in range(i):
            mol_i = rdkit.Chem.RemoveHs(deepcopy(conformer_i))
            mol_j = rdkit.Chem.RemoveHs(deepcopy(conformers[j]))
            dists.append(rdMolAlign.GetBestRMS(mol_i, mol_j))

    clusts = Butina.ClusterData(dists, len(conformers), threshold, isDistData=True, reordering=True)

    if isinstance(num_max_conformers, int):
        while len(clusts) > num_max_conformers:
            threshold += 0.1
            clusts = Butina.ClusterData(dists, len(conformers),
                                        threshold, isDistData=True, reordering=True)

    select_confs = [c[0] for c in clusts] # picking centroids of each cluster

    return select_confs


def optimize_conformer_with_xtb(conformer: rdkit.Chem.Mol,
                                solvent: Optional[str] = None,
                                num_cores: int = 1,
                                charge: int = 0,
                                temp_dir: str = ''):
    """
    Uses external calls to GFN2-XTB (command line) to optimize a conformer geometry
    
    Args:
        conformer -- RDKit mol object containing 3D coordinates
        solvent -- None or str indicating any implicit solvent to be used during optimization
            Solvent that is supported by XTB (https://xtb-docs.readthedocs.io/en/latest/gbsa.html)
        num_cores -- number of cpu cores to be used in the xtb geometry optimization
        charge -- int of the molecular charge
        temp_dir -- str temporary directory for I/O
    
    Returns: 
        (xtb_mol, energy, charges) -- tuple of optimized RDKit mol object, xtb energy (in Hartrees)
                                      and partial charges (in e-)
    """
    mol = deepcopy(conformer)

    # rand = str((os.getpid() * int(time.time())) % 123456789)
    rand = str(uuid.uuid4()) + ''.join(str(time.time()).split('.')[1])
    out_dir = Path(f'temp_xtb_opt_{rand}/')
    if temp_dir != '':
        out_dir = temp_dir / out_dir
    out_dir.mkdir(exist_ok=True)

    input_file = 'input_mol.xyz'
    rdkit.Chem.rdmolfiles.MolToXYZFile(mol, out_dir/input_file)

    if solvent is not None:
        subprocess.check_call(
            ['xtb', input_file, '--opt', '--alpb', solvent, '--parallel', str(num_cores), '--chrg', str(charge)],
            cwd = out_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    else:
        subprocess.check_call(
            ['xtb', input_file, '--opt', '--parallel', str(num_cores), '--chrg', str(charge)],
            cwd = out_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

    xtb_coords_list, xtb_elements_list = read_multi_xyz_file(out_dir/'xtbopt.xyz')
    xtb_coords = xtb_coords_list[0]
    xtb_mol = update_mol_coordinates(mol, xtb_coords)

    with open(out_dir/'xtbopt.xyz', 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'energy' in line:
                numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)
                energy = float(numbers[0])
                break
    
    with open(out_dir/'charges', 'r') as file:
        lines = file.readlines()
        charges = [0]*len(lines)
        for i, line in enumerate(lines):
            charges[i] = float(line.split()[0]) 
            xtb_mol.GetAtomWithIdx(i).SetProp('charge', str(charges[i]))

    xtb_mol.SetProp("energy", str(energy))

    shutil.rmtree(out_dir)

    return (xtb_mol, energy, charges)


def optimize_conformer_with_xtb_from_xyz_block(xyz_block: str,
                                               solvent: Optional[str] = None,
                                               num_cores: int = 1,
                                               charge: int = 0,
                                               temp_dir: str = ''):
    """
    Uses external calls to GFN2-XTB (command line) to optimize an set of coordinates provided
    as an xyz block.
    
    Args:
        xyz_block -- str of an xyz block
        solvent -- None or str indicating any implicit solvent to be used during optimization
            Solvent that is supported by XTB (https://xtb-docs.readthedocs.io/en/latest/gbsa.html)
        num_cores -- number of cpu cores to be used in the xtb geometry optimization
        charge -- int of the molecular charge
        temp_dir -- str temporary directory for I/O
    
    Returns: 
        (xtb_mol, energy, charges) -- tuple of optimized RDKit mol object, xtb energy (in Hartrees)
                                      and partial charges (in e-)
    """
    
    # rand = str((os.getpid() * int(time.time())) % 123456789)
    rand = str(uuid.uuid4()) + ''.join(str(time.time()).split('.')[1])
    out_dir = Path(f'temp_xtb_opt_{rand}/')
    if temp_dir != '':
        out_dir = temp_dir / out_dir
    out_dir.mkdir(exist_ok=True)

    input_file = 'input_mol.xyz'
    with open(out_dir/input_file, 'w') as f:
        f.write(xyz_block)

    if solvent is not None:
        subprocess.check_call(
            ['xtb', input_file, '--opt', '--alpb', solvent, '--parallel', str(num_cores), '--chrg', str(charge)],
            cwd = out_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    else:
        subprocess.check_call(
            ['xtb', input_file, '--opt', '--parallel', str(num_cores), '--chrg', str(charge)],
            cwd = out_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    
    opt_xyz_path = out_dir/'xtbopt.xyz'
    if opt_xyz_path.is_file():
        with open(out_dir/'xtbopt.xyz', 'r') as file:
            xtb_xyz_block = file.readlines()
            for line in xtb_xyz_block:
                if 'energy' in line:
                    numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)
                    energy = float(numbers[0])
                    break

        if 'energy' in xtb_xyz_block[1]:
            xtb_xyz_block[1] = '\n' # Replace energy line with blank
        xtb_xyz_block = ''.join(xtb_xyz_block)
    else:
        xtb_xyz_block = None
        energy = None

    with open(out_dir/'charges', 'r') as file:
        lines = file.readlines()
        charges = [0]*len(lines)
        for i, line in enumerate(lines):
            charges[i] = float(line.split()[0]) 

    shutil.rmtree(out_dir)

    return (xtb_xyz_block, energy, charges)


def charges_from_single_point_conformer_with_xtb(conformer: rdkit.Chem.Mol,
                                                 solvent: Optional[str] = None,
                                                 num_cores: int = 1,
                                                 charge: int = 0,
                                                 temp_dir: str = ''
                                                 ):
    """
    Uses external calls to GFN2-XTB (command line) to compute the atomic partial charges from
    a single point calculation of a provided conformer.
    
    Args:
        conformer -- RDKit mol object containing 3D coordinates
        solvent -- None or str indicating any implicit solvent to be used during optimization
            Solvent that is supported by XTB (https://xtb-docs.readthedocs.io/en/latest/gbsa.html)
        num_cores -- number of cpu cores to be used in the xtb geometry optimization
        charge -- int of the molecular charge
        temp_dir -- str temporary directory for I/O
    
    Returns: 
        charges -- list of partial charges for each atom (in e-)
    """

    mol = deepcopy(conformer)

    # rand = str((os.getpid() * int(time.time())) % 123456789)
    rand = str(uuid.uuid4()) + ''.join(str(time.time()).split('.')[1])
    out_dir = Path(f'temp_xtb_opt_{rand}/')
    if temp_dir != '':
        out_dir = temp_dir / out_dir
    out_dir.mkdir(exist_ok=True)

    input_file = 'input_mol.xyz'
    rdkit.Chem.rdmolfiles.MolToXYZFile(mol, out_dir/input_file)

    if solvent is not None:
        subprocess.check_call(
            ['xtb', input_file, '--scc', '--alpb', solvent, '--parallel', str(num_cores), '--chrg', str(charge)],
            cwd = out_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    else:
        subprocess.check_call(
            ['xtb', input_file, '--scc', '--parallel', str(num_cores), '--chrg', str(charge)],
            cwd = out_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

    with open(out_dir/'charges', 'r') as file:
        lines = file.readlines()
        charges = [0]*len(lines)
        for i, line in enumerate(lines):
            charges[i] = float(line.split()[0]) 

    shutil.rmtree(out_dir)

    return charges


def single_point_xtb_from_xyz(xyz_block: str,
                              solvent: Optional[str] = None,
                              num_cores: int = 1,
                              charge: int = 0,
                              temp_dir: str = ''):
    """
    Uses external calls to GFN2-XTB (command line) to compute the energy and atomic partial charges
    from a single point calculation of a provided conformer.
    
    Args:
        xyz_block -- str of xyz block representing a molecule
        solvent -- None or str indicating any implicit solvent to be used during optimization
            Solvent that is supported by XTB (https://xtb-docs.readthedocs.io/en/latest/gbsa.html)
        num_cores -- number of cpu cores to be used in the xtb geometry optimization
        charge -- int of the molecular charge
        temp_dir -- str temporary directory for I/O
    
    Returns: 
        energy -- float xtb energy in Hartrees
        charges -- list of partial charges for each atom (in e-)
    """

    # rand = str((os.getpid() * int(time.time())) % 123456789)
    rand = str(uuid.uuid4()) + ''.join(str(time.time()).split('.')[1])
    out_dir = Path(f'temp_xtb_opt_{rand}/')
    if temp_dir != '':
        out_dir = temp_dir / out_dir
    out_dir.mkdir(exist_ok=True)

    input_file = 'input_mol.xyz'
    with open(out_dir/input_file, 'w') as f:
        f.write(xyz_block)

    if solvent is not None:
        output = subprocess.check_output(
            ['xtb', input_file, '--scc', '--alpb', solvent, '--parallel', str(num_cores), '--chrg', str(charge)],
            cwd = out_dir,
            universal_newlines=True,
            stderr=subprocess.STDOUT,
        )
    else:
        output = subprocess.check_output(
            ['xtb', input_file, '--scc', '--parallel', str(num_cores), '--chrg', str(charge)],
            cwd = out_dir,
            universal_newlines=True,
            stderr=subprocess.STDOUT,
        )

    output = output.split('\n')
    for line in output[::-1]:
        if 'TOTAL ENERGY' in line:
            numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)
            energy = float(numbers[0])
            break

    with open(out_dir/'charges', 'r') as file:
        lines = file.readlines()
        charges = [0]*len(lines)
        for i, line in enumerate(lines):
            charges[i] = float(line.split()[0]) 

    shutil.rmtree(out_dir)

    return energy, charges


def optimize_conformer_ensemble_with_xtb(conformers,
                                         solvent: Optional[str] = None,
                                         num_processes: int = 8,
                                         charge: int = 0,
                                         temp_dir: str = '',
                                         verbose: bool = False):
    """
    GFN2-XTB geometry optimization of a list of conformers
    
    Args:
        conformers -- list of RDKit mol objects (with 3D coordinates) to be optimized
        solvent -- None or str indicating any implicit solvent to be used during optimization
            Solvent that is supported by XTB (https://xtb-docs.readthedocs.io/en/latest/gbsa.html)
        num_cores -- number of cpu cores to be used in the xtb geometry optimization
        charge -- int of the molecular charge
        temp_dir -- str temporary directory for I/O
        verbose -- toggle print tqdm
    
    Returns: 
        conformers_opt -- list of RDKit mol objects containing optimized geometries
        charges_opt -- list of lists containing partial charges corresponding to each atom of
                        the optimized geometries
    """

    if len(conformers) == 1:
        opt_conf, opt_energy, opt_charges = optimize_conformer_with_xtb(
            conformers[0],
            solvent = solvent,
            num_cores = num_processes,
            charge=charge,
            temp_dir=temp_dir
        )
        opt_conf.SetProp("energy", str(opt_energy))
        conformers_opt = [opt_conf]
        charges_opt = [opt_charges]
        energies_opt = [opt_energy]
    
    else:
        conformers_opt = []
        charges_opt = []
        energies_opt = []
        if verbose:
            pbar = tqdm(conformers, total=len(conformers), desc='XTB opt')
        else:
            pbar = conformers
        for conf in pbar:
            opt_conf, opt_energy, opt_charges = optimize_conformer_with_xtb(
                conformer=conf,
                solvent=solvent,
                num_cores=num_processes,
                charge=charge,
                temp_dir=temp_dir
            )
            opt_conf.SetProp("energy", str(opt_energy))
            for i, partial_charge in enumerate(opt_charges):
                opt_conf.GetAtomWithIdx(i).SetProp('charge', str(partial_charge))
            conformers_opt.append(opt_conf)
            charges_opt.append(opt_charges)
            energies_opt.append(opt_energy)

    return conformers_opt, energies_opt, charges_opt


def generate_opt_conformers_xtb(smiles: str,
                                charge: int = 0,
                                solvent: Optional[str]=None,
                                MMFF_optimize: bool = True,
                                num_processes: int = 1,
                                temp_dir: str = '',
                                verbose: bool = False,
                                num_confs: int = 1000):
    """
    Generate conformer ensemble with rdkit then relax with xTB.
    
    Args:
        smiles -- str
        charge -- int of the molecular charge
        MMFF_optimize -- bool optimize RDKit embedded molecules with MMFF94
        num_processes -- number of cpu cores to be used in the xtb geometry optimization
        solvent -- None or str indicating any implicit solvent to be used during optimization
            Solvent that is supported by XTB (https://xtb-docs.readthedocs.io/en/latest/gbsa.html)
        temp_dir -- str temporary directory for I/O
        verbose -- bool toggle tqdm
        num_confs -- int number of conformers to initially generate
    
    Returns: 
        clustered_conformers_xtb: list of rdkit conformers after xTB relaxation and clustering
        clustered_energies_xtb: list of energies for associated conformers
        clustered_charges_xtb: list of partial charges for associated conformers
    """
    mol_3d = embed_conformer_from_smiles(smiles, attempts = 50, MMFF_optimize = MMFF_optimize)

    if mol_3d is None:
        return None, None, None

    conformer_ensemble = generate_conformer_ensemble(
        mol_3d,
        num_confs=num_confs,
        num_threads = 4,
        threshold = 0.05,
        num_opt_steps = 50,
    )
    if verbose:
        print(f'Num conformers generated: \t{len(conformer_ensemble)}')

    clustered_conformers_index = cluster_conformers_butina(
        conformer_ensemble,
        threshold = 0.1,
        num_max_conformers = None,
    )
    clustered_conformers = [conformer_ensemble[c] for c in clustered_conformers_index]
    if verbose:
        print(f'Num conformers remaining: \t{len(clustered_conformers)}')

    # further optimizing each conformer with GFN2-XTB
    optimized_conformers_xtb, optimized_energies_xtb, optimized_charges_xtb = optimize_conformer_ensemble_with_xtb(
        clustered_conformers,
        solvent = solvent,
        num_processes = num_processes,
        charge=charge,
        temp_dir = temp_dir,
        verbose=verbose
    )
    if verbose:
        print(f'Num optimized confomers with xtb: \t{len(optimized_conformers_xtb)}')

    clustered_conformers_xtb_index = cluster_conformers_butina(
        optimized_conformers_xtb,
        threshold = 0.1,
        num_max_conformers = None,
    )
    clustered_conformers_xtb = [optimized_conformers_xtb[c] for c in clustered_conformers_xtb_index]
    clustered_charges_xtb = [optimized_charges_xtb[c] for c in clustered_conformers_xtb_index]
    clustered_energies_xtb = [optimized_energies_xtb[c] for c in clustered_conformers_xtb_index]

    if verbose:
        print(f'Num conformers remaining after xtb opt and cluster: \t{len(clustered_conformers_xtb)}')

    return clustered_conformers_xtb, clustered_energies_xtb, clustered_charges_xtb


def generate_opt_conformers(smiles: str,
                            MMFF_optimize: bool = True,
                            verbose: bool = False,
                            num_confs: int = 1000):
    """
    Generate optimal conformers with rdkit (MMFF94)
    
    Args:
        smiles -- str
        MMFF_optimize -- bool optimize RDKit embedded molecules with MMFF94
        verbose -- bool toggle tqdm
        num_confs -- int number of conformers to initially generate
    
    Returns: 
        clustered_conformers_xtb: list of clustered rdkit conformers after RDKit embedding and
            optional MMFF relaxation
    """
    mol_3d = embed_conformer_from_smiles(smiles, attempts = 50, MMFF_optimize = MMFF_optimize)

    if mol_3d is None:
        return None, None, None

    conformer_ensemble = generate_conformer_ensemble(
        mol_3d,
        num_confs=num_confs,
        num_threads = 4,
        threshold = 0.05,
        num_opt_steps = 50,
    )
    if verbose:
        print(f'Num conformers generated: \t{len(conformer_ensemble)}')

    clustered_conformers_index = cluster_conformers_butina(
        conformer_ensemble,
        threshold = 0.1,
        num_max_conformers = None,
    )
    clustered_conformers = [conformer_ensemble[c] for c in clustered_conformers_index]
    if verbose:
        print(f'Num conformers remaining: \t{len(clustered_conformers)}')

    return clustered_conformers