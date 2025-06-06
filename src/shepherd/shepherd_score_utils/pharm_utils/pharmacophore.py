"""
Copied from: https://github.com/coleygroup/shepherd-score/blob/main/shepherd_score/pharm_utils/pharmacophore.py

Generate pharmacophores from a rdkit conformer.

Parts of code adapted from Francois Berenger / Tsuda Lab and rdkit.
Code from Tsuda Lab: https://github.com/tsudalab/ACP4/blob/master/bin/acp4_ph4.py
    From https://doi.org/10.1021/acs.jcim.2c01623
Code from rdkit:
    https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/Features/FeatDirUtilsRD.py
    https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/Features/ShowFeats.py
"""

import os
import sys
sys.path.append('.')
from copy import deepcopy
import math
from typing import List, Tuple, Dict, Union

import numpy as np
from scipy.spatial import distance, Delaunay

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
PT = Chem.GetPeriodicTable()

# pharmacophores
from .pharmvec import GetDonorFeatVects, GetAcceptorFeatVects, GetAromaticFeatVects, GetHalogenFeatVects
#from shepherd_score.score.constants import P_TYPES
P_TYPES = ('Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'Halogen', 'Cation', 'Anion', 'ZnBinder')

feature_colors = {
  'Donor': (0, 1, 1),
  'Acceptor': (1, 0, 1),
  'NegIonizable': (1, 0, 0),
  'Anion': (1,0,0),
  'PosIonizable': (0, 0, 1),
  'Cation': (0,0,1),
  'ZnBinder': (1, .5, .5),
  'Zn': (1, .5, .5),
  'Aromatic': (1, .8, .2),
  'LumpedHydrophobe': (.5, .25, 0),
  'Hydrophobe': (.5, .25, 0),
  'Halogen': (.13, .55, .13)
}

# Below is used to get hydrophobic groups
#### From https://github.com/tsudalab/ACP4/blob/master/bin/acp4_ph4.py ####
#### Credit to Francois Berenger and Tsuda Lab ####
#### https://doi.org/10.1021/acs.jcim.2c01623 ####

# These are the same as Pharmer / Pmapper
__hydrophobic_smarts = [
    "a1aaaaa1",
    "a1aaaa1",
    # branched terminals as one point
    "[$([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])&!$(**[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])]",
    "[$(*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])&!$(*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I])]([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]",
    "*([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])([CH3X4,CH2X3,CH1X2,F,Cl,Br,I])[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]",
    # simple rings only; need to combine points to get good results for 3d structures
    "[C&r3]1~[C&r3]~[C&r3]1",
    "[C&r4]1~[C&r4]~[C&r4]~[C&r4]1",
    "[C&r5]1~[C&r5]~[C&r5]~[C&r5]~[C&r5]1",
    "[C&r6]1~[C&r6]~[C&r6]~[C&r6]~[C&r6]~[C&r6]1",
    "[C&r7]1~[C&r7]~[C&r7]~[C&r7]~[C&r7]~[C&r7]~[C&r7]1",
    "[C&r8]1~[C&r8]~[C&r8]~[C&r8]~[C&r8]~[C&r8]~[C&r8]~[C&r8]1",
    # aliphatic chains
    "[CH2X4,CH1X3,CH0X2]~[CH3X4,CH2X3,CH1X2,F,Cl,Br,I]",
    "[$([CH2X4,CH1X3,CH0X2]~[$([!#1]);!$([CH2X4,CH1X3,CH0X2])])]~[CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]",
    "[$([CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]~[$([CH2X4,CH1X3,CH0X2]~[$([!#1]);!$([CH2X4,CH1X3,CH0X2])])])]~[CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]",
    # sulfur (apparently)
    "[$([S]~[#6])&!$(S~[!#6])]"
]

def pattern_of_smarts(s):
    return Chem.MolFromSmarts(s)

__hydrophobic_patterns = list(map(pattern_of_smarts, __hydrophobic_smarts))

# geometric center of a matched pattern
def __average_match(mol, matched_pattern):
    avg_x = 0.0
    avg_y = 0.0
    avg_z = 0.0
    count = float(len(matched_pattern))
    conf0 = mol.GetConformer()
    for i in matched_pattern:
        xyz = conf0.GetAtomPosition(i)
        avg_x += xyz.x
        avg_y += xyz.y
        avg_z += xyz.z
    center = (avg_x / count,
              avg_y / count,
              avg_z / count)
    return center

def __find_matches(mol, patterns):
    res = []
    for pat in patterns:
        # get all matches for that pattern
        matched = mol.GetSubstructMatches(pat)
        for m in matched:
            # get the center of each matched group
            avg = __average_match(mol, m)
            res.append(avg)
    return res

def __euclid(xyz0, xyz1):
    x0, y0, z0 = xyz0
    x1, y1, z1 = xyz1
    dx = x0 - x1
    dy = y0 - y1
    dz = z0 - z1
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def __average(l):
    sum_x = 0.0
    sum_y = 0.0
    sum_z = 0.0
    n = float(len(l))
    for (x, y, z) in l:
        sum_x += x
        sum_y += y
        sum_z += z
    return (sum_x / n,
            sum_y / n,
            sum_z / n)

def find_hydrophobes(mol: rdkit.Chem.rdchem.Mol,
                     cluster_hydrophobic: bool = True):
    """
    Find hydrophobes and cluster them.

    Arguments
    ---------
    mol : rdkit Mol object with a conformer.
    cluster_hydrophobic : bool (default=True) to cluster hydrophobic atoms if they fall within 2A.

    Returns
    -------
    list of tuples containing coordinates for the locations for each hydrophobe.
    """
    all_hydrophobes = __find_matches(mol, __hydrophobic_patterns)
    if not cluster_hydrophobic:
        return all_hydrophobes
    else:
        # regroup all hydrophobic features within 2.0A
        grouped_hydrophobes = []
        n = len(all_hydrophobes)
        idx2cluster = list(range(n))
        for i in range(n):
            h_i = all_hydrophobes[i]
            cluster_id = idx2cluster[i]
            for j in range(i+1, n):
                h_j = all_hydrophobes[j]
                if __euclid(h_i, h_j) <= 2.0: # Angstrom
                    # same cluster
                    idx2cluster[j] = cluster_id
        cluster_ids = set(idx2cluster)
        for cid in cluster_ids:
            group = []
            for i, h in enumerate(all_hydrophobes):
                if idx2cluster[i] == cid:
                    group.append(h)
            grouped_hydrophobes.append(__average(group))
        return grouped_hydrophobes
    
### End Tsuda Lab code


def _get_points_fibonacci(num_samples):
    """
    Generate points on unit sphere using fibonacci approach.
    Adapted from Morfeus:
    https://github.com/digital-chemistry-laboratory/morfeus/blob/main/morfeus/geometry.py
    
    Parameters
    ----------
    num_samples : int
        Number of points to sample from the surface of a sphere
    
    Returns
    -------
    np.ndarray (num_samples,3)
        Coordinates of the sampled points.
    """
    offset = 2.0 / num_samples
    increment = np.pi * (3.0 - np.sqrt(5.0))

    i = np.arange(num_samples)
    y = ((i * offset) - 1) + (offset / 2)
    r = np.sqrt(1 - np.square(y))
    phi = np.mod((i + 1), num_samples) * increment
    x = np.cos(phi) * r
    z = np.sin(phi) * r

    points = np.column_stack((x, y, z))
    return points


def __outside_hull(sample_points: np.ndarray,
                   hull: Delaunay
                   ) -> np.ndarray:
    """
    Test if points in `sample_points` are outside of the convex hull formed by the atoms.

    Arguments
    ---------
    sample_points : (N,3) np.ndarray of the points to check if outside the "interior" of the molecule.
    hull : scipy.spatial.Delaunay object initialized by the positions of the atoms of the molecule.

    Returns
    -------
    (N,) np.ndarray of booleans describing if sample_points are outside of the convex hull
    """
    return hull.find_simplex(sample_points) < 0


def __is_accessible(interaction_sphere, atom_pos, radii, mask_atom_idx):
    """
    Check if at least 2% of sampled points fall within a surface-accessible volume of the molecule.
     This is 2% of the original 200 points (4 points).
    Currently using SAS with a probe radius of 0.8A rather than vdW volume. vdW volume will fail to
     exclude buried pharmacophores. Also experimented with checking if the interaction points fell
     within a convex hull and buried volume with Morpheus which both had limited improvements.

    Arguments
    ---------
    interaction_sphere : np.ndarray (M, 3) of points to check accessibility of a potentially
                    interacting atom. M <= 200
    atom_pos : np.ndarray (N, 3) Positions of atoms in molecule.
    radii : np.ndarray (N,) vdW radii for each corresponding atom.
    mask_atom_idx : np.ndarray of bool (N,) contains atom indices to ignore if the interaction
                    points are within their SA volumes. For example, the acceptor atom or the
                    donating hydrogens.
    
    Returns
    -------
    bool
    """
    # compute distances from each sampled point to all atoms (except excluded)
    dist_matrix = distance.cdist(interaction_sphere, atom_pos[mask_atom_idx])
    mask = np.all(dist_matrix >= radii + 0.8, axis=1) # mask for points within vdW + probe radius
    interaction_sphere = interaction_sphere[mask]

    # if hull is not None:
    #      # If you actually want to include this, then only compute Delaunay ONCE per molecule (outside this func).
    #     hull = Delaunay(mol.GetConformer().GetPositions())
    #     sas_mask = np.all(dist_matrix[mask] >= radii + 0.8, axis=1) # points within SAS defined volume
    #     hull_mask = __outside_hull(interaction_sphere, hull).astype(bool) # points within hull
    #     interaction_sphere = interaction_sphere[hull_mask | sas_mask]

    num_accessible = len(interaction_sphere) # number of non-colliding points
    if num_accessible > 4: # at least 2% accessible from initial total 200 points
        return True
    else:
        return False


def _is_donator_accessible(mol: rdkit.Chem.rdchem.Mol,
                           hydrogens: Union[List[rdkit.Chem.rdchem.Atom], None],
                           pharm_pos: Tuple,
                           unit_vec: Tuple,
                           ) -> bool:
    """
    Check accessbility of donator atoms inspired by protocol of Pharao.
    DOI: 10.1016/j.jmgm.2008.04.003
    Check whether at least 2% of the points sampled on a sphere of 1.8A radius is accessible.
        i.e., beyond the SAS        
    Arguments
    ---------
    mol : rdkit Mol with conformer
    pharm_pos : tuple holding coords of anchor point
    unit_vec : tuple holding coords of releative unit vector
    num_nbrs : int of the number of neighbors to the acceptor (heavy + hydr)

    Returns
    -------
    bool
    """
    if hydrogens is None:
        hyd_atom_ids = []
    else:
        hyd_atom_ids = [h.GetIdx() for h in hydrogens]
    radii = np.array([PT.GetRvdw(atom.GetAtomicNum()) for i, atom in enumerate(mol.GetAtoms()) if i not in hyd_atom_ids])

    # Pharmacophore position is about 1.2A in direction of vector
    pharm_pos = np.array(pharm_pos) + 1.2*np.array(unit_vec)

    # unit sphere
    interaction_sphere = _get_points_fibonacci(200)
    interaction_radius = 1.8 # angstroms
    interaction_sphere *= interaction_radius
    interaction_sphere += pharm_pos # move to position of pharmacophore

    atom_pos = mol.GetConformer().GetPositions()
    # don't include the hydrogens themselves
    mask_atom_idx = np.isin(np.arange(len(atom_pos)), hyd_atom_ids, invert=True)
    return __is_accessible(interaction_sphere, atom_pos, radii, mask_atom_idx)


def _is_acceptor_accessible(mol: rdkit.Chem.rdchem.Mol,
                            acceptor_atom: rdkit.Chem.rdchem.Atom,
                            pharm_pos: Tuple,
                            unit_vec: Tuple,
                            num_nbrs: int,
                            ) -> bool:
    """
    Check accessbility of acceptor atoms inspired by protocol of Pharao.
    DOI: 10.1016/j.jmgm.2008.04.003
    Check whether at least 2% of the points sampled on a sphere of 1.8A radius is accessible.
        i.e., beyond the SAS

    Arguments
    ---------
    mol : rdkit Mol with conformer
    acceptor_atom : rdkit Atom from mol that is the acceptor
    pharm_pos : tuple holding coords of anchor point
    unit_vec : tuple holding coords of releative unit vector
    num_nbrs : int of the number of neighbors to the acceptor (heavy + hydr)

    Returns
    -------
    bool
    """
    acceptor_atom_id = acceptor_atom.GetIdx()
    radii = np.array([PT.GetRvdw(atom.GetAtomicNum()) for i, atom in enumerate(mol.GetAtoms()) if i != acceptor_atom_id])

    pharm_pos = np.array(pharm_pos)

    # unit sphere
    interaction_sphere = _get_points_fibonacci(200)

    # mask out irrelevant parts of the sphere
    if num_nbrs >= 3:
        # hemisphere
        vec = np.array(unit_vec)
        inds = np.where(np.dot(vec, interaction_sphere.T) > 0)[0]
        interaction_sphere = interaction_sphere[inds]
    elif num_nbrs == 2:
        # Little more than a hemisphere, sqrt(2)/2 = -0.7071 -> 180+45 deg
        vec = np.array(unit_vec)
        inds = np.where(np.dot(vec, interaction_sphere.T) > -0.7071)[0]
        interaction_sphere = interaction_sphere[inds]
    # otherwise full sphere

    interaction_radius = 1.8 # angstroms
    interaction_sphere *= interaction_radius
    interaction_sphere += pharm_pos # move to position of pharmacophore

    atom_pos = mol.GetConformer().GetPositions()
    # don't include the atom itself
    mask_atom_idx = np.where(np.arange(len(atom_pos)) != acceptor_atom_id)[0]
    return __is_accessible(interaction_sphere, atom_pos, radii, mask_atom_idx)


### From rdkit:
# https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/Features/FeatDirUtilsRD.py
# https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/Features/ShowFeats.py


def _average_vectors(vectors: List):
    """
    Arguments
    ---------
    vectors : List of rdkit geometry point3d objects. These should be unit vectors.

    Returns
    -------
    rdkit.Geometry.rdGeometry.Point3D object that is an average of the provided vectors.
    """
    avg_vec = 0
    for v in vectors:
        if avg_vec == 0:
            avg_vec = deepcopy(v)
        else:
            avg_vec += v
    avg_vec.Normalize()
    return avg_vec


def get_pharmacophores_dict(mol: rdkit.Chem.rdchem.Mol,
                            multi_vector: bool = True,
                            exclude: List[int] = [],
                            check_access: bool = False,
                            scale: float = 1.0
                            ) -> Dict:
    """
    Get the positions of pharmacophore anchors and their associated unit vectors.
    Returns a dictionary.
    Adapted from rdkit.Chem.Features.ShowFeats.ShowMolFeats.

    Arguments
    ---------
    mol : rdkit Mol object with a conformer.
    multi_vector : bool to choose whether to represent pharmacophores with multiple vectors. (default = True)
    exclude : list of atom indices to not include as a HBD.
    check_access : bool check if HBD/HBA are accessible to the molecular surface. (default = True)
    scale : float as the length of the vector in Angstroms (default = 1.0).

    Returns
    -------
    dictionary : {'FeatureName' : {
                                   'P': [(anchor coord), ... ],
                                   'V': [(rel. vec), ... ]
                                   }
                 }
    """
    pharmacophores = {}
    
    dirname = os.path.dirname(__file__)
    fdef_file = os.path.join(dirname, 'smarts_features.fdef')
    factory = AllChem.BuildFeatureFactory(fdef_file)
    mol_feats = factory.GetFeaturesForMol(mol)

    # Filter only these for rdkit processing, we will compute hydrophobes later
    keep = ('Aromatic', 'ZnBinder', 'Donor', 'Acceptor', 'Cation', 'Anion', 'Halogen')

    # Non-hydrophobe pharmacophore processing
    for feat in mol_feats:
        family = feat.GetFamily() # type of pharmacophore
        if family not in keep:
          continue
        if family not in pharmacophores:
            pharmacophores[family] = {}
            pharmacophores[family]['P'] = []
            pharmacophores[family]['V'] = []

        pos = feat.GetPos() # positions of pharmacophore anchor

        if family.lower() == 'aromatic':
            anchor, vec = GetAromaticFeatVects(conf = mol.GetConformer(), 
                                               featAtoms = feat.GetAtomIds(),
                                               featLoc = pos,
                                               return_both = multi_vector,
                                               scale = scale)
            if not multi_vector:
                anchor = anchor[0]
                vec = vec[0]

        elif family.lower() == 'donor':
            aids = feat.GetAtomIds()
            if len(aids) == 1:
                featAtom = mol.GetAtomWithIdx(aids[0])
                # Multivector by default
                anchor, vec, hydrogen_list = GetDonorFeatVects(conf = mol.GetConformer(),
                                                               featAtoms = aids,
                                                               scale = scale,
                                                               exclude = exclude)
                if vec is not None and len(vec) > 1:
                    avg_vec = _average_vectors(vec)
                else:
                    if vec is None:
                        avg_vec = None
                    else:
                        avg_vec = deepcopy(vec[0])

                if check_access:
                    if anchor is None or avg_vec is None:
                        continue
                    elif not _is_donator_accessible(mol = mol,
                                                    hydrogens = hydrogen_list,
                                                    pharm_pos = anchor if not isinstance(anchor, list) else anchor[0],
                                                    unit_vec = avg_vec
                                                    ):
                        continue # don't keep this pharmacophore
                
                # If only one vector per pharmacophore
                if not multi_vector and anchor is not None:
                    anchor = anchor[0]
                    vec = deepcopy(avg_vec)

        elif family.lower() == 'acceptor':
            aids = feat.GetAtomIds()
            if len(aids) == 1:
                featAtom = mol.GetAtomWithIdx(aids[0])
                # Multivector by default
                anchor, vec = GetAcceptorFeatVects(conf = mol.GetConformer(),
                                                   featAtoms = aids,
                                                   scale = scale)

                if vec is not None and len(vec) > 1:
                    avg_vec = _average_vectors(vec)
                else:
                    if vec is None:
                        avg_vec = None
                    else:
                        avg_vec = deepcopy(vec[0])

                if check_access:
                    if anchor is None or avg_vec is None:
                        continue
                    numNbrs = len(featAtom.GetNeighbors())
                    if not _is_acceptor_accessible(mol = mol,
                                                   acceptor_atom = featAtom,
                                                   pharm_pos = anchor if not isinstance(anchor, list) else anchor[0],
                                                   unit_vec = avg_vec,
                                                   num_nbrs = numNbrs):
                        continue # don't keep this pharmacophore

                # If only one vector per pharmacophore
                if not multi_vector and anchor is not None:
                    anchor = anchor[0]
                    vec = deepcopy(avg_vec)
        
        elif family.lower() == 'halogen':
            aids = feat.GetAtomIds()
            if len(aids) == 1:
                featAtom = mol.GetAtomWithIdx(aids[0])
                anchor, vec = GetHalogenFeatVects(conf = mol.GetConformer(),
                                                  featAtoms = aids, 
                                                  scale = scale)
                anchor = anchor[0]
                vec = vec[0]

        else:
            anchor = tuple(pos)
            vec = (0,0,0)

        if anchor is not None and vec is not None:
            if isinstance(anchor, list):
                pharmacophores[family]['P'].extend([tuple(a) for a in anchor])
                pharmacophores[family]['V'].extend(tuple(v) for v in vec)
            else:
                pharmacophores[family]['P'].append(tuple(anchor))
                pharmacophores[family]['V'].append(tuple(vec))

    # Hydrophobe processing
    hydrophobes = find_hydrophobes(mol=mol, cluster_hydrophobic=True)
    pharmacophores['Hydrophobe'] = {}
    pharmacophores['Hydrophobe']['P'] = hydrophobes
    pharmacophores['Hydrophobe']['V'] = [(0,0,0) for _ in range(len(hydrophobes))]
    return pharmacophores


def get_pharmacophores(mol: rdkit.Chem.rdchem.Mol,
                       multi_vector: bool = True,
                       exclude: List[int] = [],
                       check_access: bool = False,
                       scale: float = 1.0
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the identity, anchor positions, and relative unit vectors for each pharmacophore.
    Pharmacophore ordering for indexing:
        ('Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'Cation', 'Anion', 'ZnBinder')

    Note: `check_access` is currently based on whether interaction points sampled from a sphere's
     surface with a radius of 1.8A from the acceptor/donor atom falls outside the solvent
     accessible surface defined by the vdW radius + 0.8A of the neighboring atoms. This works for
     buried acceptors/donors, but may be prone to false positives. For example, CN(C)C would have
     its sole HBA rejected. Other approaches such as buried volume should be considered in the
     future.
    
    Arguments
    ---------
    mol : rdkit Mol object with conformer.
    multi_vector : bool to choose whether to represent pharmacophores with multiple vectors. (default = True)
    exclude : list of hydrogen indices to not include as a HBD.
    check_access : bool check if HBD/HBA are accessible to the molecular surface. (default = True)
    scale : float as the length of a pharmacophore vector in Angstroms (default = 1.0).

    Returns
    -------
    tuple
        X : np.ndarray (N,)
            Identity of pharmacophore corresponding to the indexing order.
        P : np.ndarray (N, 3)
            Anchor positions of each pharmacophores
        V : np.ndarray (N, 3)
            Unit vectors in a relative position to the anchor positions.
            Adding P and V results in the position of the vector's extended point.
    """
    pharmacophores_dict = get_pharmacophores_dict(mol=mol,
                                                  multi_vector=multi_vector,
                                                  check_access=check_access,
                                                  scale=scale,
                                                  exclude=exclude)

    X, P, V = [], [], []
    for family in pharmacophores_dict:
        anchor_pos = pharmacophores_dict[family]['P']
        P.extend(anchor_pos)
        V.extend(pharmacophores_dict[family]['V'])
        type_embed = P_TYPES.index(family)
        X.extend([type_embed]*len(anchor_pos))

    return np.array(X), np.array(P), np.array(V)