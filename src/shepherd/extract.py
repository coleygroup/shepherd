"""
Utility functions for ShEPhERD inference scaling.

This module provides helper functions for working with ShEPhERD's inference output.
"""

import logging
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


def get_xyz_content(sample):
    """
    Generate XYZ file content string from a ShEPhERD sample.
    
    Args:
        sample (dict): ShEPhERD output dictionary with x1 containing atoms and positions.
        
    Returns:
        str: XYZ file content or None if input is invalid.
    """
    if 'x1' not in sample or 'atoms' not in sample['x1'] or 'positions' not in sample['x1']:
        logging.warning("Invalid sample format for XYZ generation")
        return None

    try:
        atoms = sample['x1']['atoms']
        positions = sample['x1']['positions']

        xyz_lines = [f"{len(atoms)}", "Generated by ShEPhERD Inference Scaling"]
        for i in range(len(atoms)):
            try:
                atomic_number = int(atoms[i])
                # attempt to get symbol, default to element number if fails
                try:
                    symbol = Chem.Atom(atomic_number).GetSymbol()
                except Exception:
                    symbol = str(atomic_number)
                pos = positions[i]
                xyz_lines.append(f"{symbol} {pos[0]:>15.8f} {pos[1]:>15.8f} {pos[2]:>15.8f}")
            except (ValueError, IndexError) as e:
                logging.warning(f"Skipping atom {i} due to data issue: {e}")
                continue

        xyz_lines[0] = str(len(xyz_lines) - 2)

        return "\n".join(xyz_lines)

    except Exception as e:
        logging.error(f"Error generating XYZ content: {e}")
        return None


def create_rdkit_molecule(sample):
    """
    Create an RDKit molecule from ShEPhERD output using XYZ block approach.
    
    Args:
        sample (dict): ShEPhERD output dictionary with x1 containing atoms and positions.
        
    Returns:
        rdkit.Chem.rdchem.Mol: RDKit molecule object or None if conversion fails.
    """
    if 'x1' not in sample:
        logging.warning("No atom data (x1) found in sample")
        return None

    try:
        # extract atoms and their positions from x1
        atoms = sample['x1']['atoms']
        positions = sample['x1']['positions']

        # create XYZ format string
        xyz = f'{len(atoms)}\n\n'
        for a in range(len(atoms)):
            atomic_number = int(atoms[a])
            position = positions[a]
            symbol = Chem.Atom(atomic_number).GetSymbol()
            xyz += f'{symbol} {position[0]:.3f} {position[1]:.3f} {position[2]:.3f}\n'

        # create molecule from XYZ block
        mol = Chem.MolFromXYZBlock(xyz)
        if mol is None:
            logging.warning("Failed to create molecule from XYZ block")
            return None

        # try different charge states for bond determination
        mol_final = None
        for charge in [0, 1, -1, 2, -2]:
            mol_copy = deepcopy(mol)
            try:
                rdDetermineBonds.DetermineBonds(mol_copy, charge=charge, embedChiral=True)
                logging.debug(f"Bond determination successful with charge {charge}")
                mol_final = mol_copy
                break
            except Exception as e:
                logging.debug(f"Bond determination failed with charge {charge}: {e}")
                continue

        if mol_final is None:
            logging.warning("Bond determination failed for all charge states")
            return None
        
        # validate molecule
        try:
            radical_electrons = sum([a.GetNumRadicalElectrons() for a in mol_final.GetAtoms()])
            if radical_electrons > 0:
                logging.warning(f"Molecule has {radical_electrons} radical electrons")
            
            mol_final.UpdatePropertyCache()
            Chem.GetSymmSSSR(mol_final)
            logging.debug("Molecule validation successful")
        except Exception as e:
            logging.warning(f"Molecule validation failed: {e}")
            return None

        # try to generate SMILES to verify molecule
        try:
            smiles = Chem.MolToSmiles(mol_final)
            logging.debug(f"Generated SMILES: {smiles}")
        except Exception as e:
            logging.warning(f"SMILES generation failed: {e}")

        if '.' in smiles:
            logging.warning("Molecule is a fragment, failed to create molecule")
            return None

        return mol_final
        
    except Exception as e:
        logging.warning(f"Error creating molecule: {e}")
        return None
