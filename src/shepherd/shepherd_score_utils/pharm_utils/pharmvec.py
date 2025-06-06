"""
Copied from: https://github.com/coleygroup/shepherd-score/blob/main/shepherd_score/pharm_utils/pharmvec.py

Generate the vector features for pharmacophores from a rdkit conformer.

Adapted from rdkit:
    https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/Features/FeatDirUtilsRD.py
    https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/Features/ShowFeats.py

Changed to return anchor position and relative unit vector.
"""

from rdkit import Chem
from rdkit.Geometry.rdGeometry import Point3D

# pharmacophores
from rdkit.Chem.Features import FeatDirUtilsRD as FeatDirUtils

## Streamlined functions

def GetAromaticFeatVects(conf, featAtoms, featLoc, return_both: bool = False, scale=1.0):
    """
    Compute the direction vector for an aromatic feature

    Changed: only return one vector, process later for visualization and scoring

    Arguments
    ---------
        conf : a conformer
        featAtoms : list of atom IDs that make up the feature
        featLoc : location of the aromatic feature specified as point3d
        return_both : bool for whether to return both vectors or just one.
        scale : the size of the direction vector

    Returns
    -------
    Tuple
        list of anchor position(s) as rdkit Point3D
        list of relative unit vector(s) as rdkit Point3D
    """
    head = featLoc
    ats = [conf.GetAtomPosition(x) for x in featAtoms]

    v1 = ats[0] - head
    v2 = ats[1] - head
    norm1 = v1.CrossProduct(v2)
    norm1.Normalize()
    norm1 *= scale
    if return_both:
        norm2 = Point3D(0, 0, 0) - norm1
        return [head]*2, [norm1, norm2]
    else:
        return [head], [norm1]


def GetDonorFeatVects(conf, featAtoms, scale=1., exclude=[]):
    """
    Get vectors for hydrogen bond donors in the direction of the hydrogens.

    Arguments
    ---------
    conf : rdkit Mol object with a conformer.
    featAtoms : list containing rdkit Atom object of atom attributed as a donor.
    scale : float (default = 1.) length of direction vector.
    exclude : list of atom indices that should not be included as a donatable H.

    Returns
    -------
    Tuple
        list of anchor position(s) as rdkit Point3D or None
        list of relative unit vector(s) as rdkit Point3D or None
        list of neighboring hydrogens or None
    """
    assert len(featAtoms) == 1
    aid = featAtoms[0]
    mol = conf.GetOwningMol()
    cpt = conf.GetAtomPosition(aid)
    nbrs = mol.GetAtomWithIdx(aid).GetNeighbors()

    hyd_nbrs = [] # hydrogens
    vectors = []
    for nbr in nbrs:
        if nbr.GetAtomicNum() == 1:
            if nbr.GetIdx() in exclude:
                continue
            hyd_nbrs.append(nbr)
            hnid = nbr.GetIdx()
            vec = conf.GetAtomPosition(hnid)
            vec -= cpt
            vec.Normalize()
            vec *= scale
            vectors.append(vec)

    if hyd_nbrs != []:
        return [cpt]*len(vectors), vectors, hyd_nbrs
    else:
        return None, None, None
    

def GetAcceptorFeatVects(conf: Chem.rdchem.Mol,
                         featAtoms: Chem.rdchem.Atom,
                         scale: float = 1.0):
    """
    Get the anchor positions and relative unit vectors of an acceptor atom.
    Assumes HBA's are only O, and N as defined by smarts_featurs.fdef.
    If HBA is not one of those, then it assumes the atom has one lone pair.

    Arguments
    ---------
    conf : rdkit Mol object with a conformer.
    featAtoms : list containing rdkit Atom object of atom attributed as an acceptor.
    scale : float (default = 1.) length of direction vector.

    Returns
    -------
    Tuple
        list of anchor position(s) as rdkit Point3D or [None]
        list of relative unit vector(s) as rdkit Point3D or [None]
    """
    assert len(featAtoms) == 1
    atom_id = featAtoms[0]

    mol = conf.GetOwningMol()
    atom = mol.GetAtomWithIdx(atom_id)
    nbrs = atom.GetNeighbors()
    num_lone_pairs = 3
    if atom.GetAtomicNum() == 7: # N
        num_lone_pairs = 1
    elif atom.GetAtomicNum() == 8: # O
        num_lone_pairs = 2

    hydrogens = []
    heavy = []
    for nbr in nbrs:
        if nbr.GetAtomicNum() == 1:
            hydrogens.append(nbr)
        else:
            heavy.append(nbr)

    cpt = conf.GetAtomPosition(atom_id)

    if num_lone_pairs == 1:
        if len(nbrs) == 1: # linear
            # triple bond
            v1 = conf.GetAtomPosition(nbrs[0].GetIdx())
            v1 -= cpt
            v1.Normalize()
            v1 *= (-1.0 * scale)
            return [cpt], [v1]

        elif len(nbrs) == 2: # sp2
            v1 = FeatDirUtils._findAvgVec(conf, cpt, nbrs)
            v1 *= (-1.0 * scale)
            return [cpt], [v1]
        
        elif len(nbrs) == 3: # sp3
            cpt = conf.GetAtomPosition(atom_id)
            out = FeatDirUtils._GetTetrahedralFeatVect(conf, atom_id, scale)
            if out != ():
                v1 = out[0][1] - cpt # need to subtract out the center to be relative
                v1.Normalize()
                v1 *= scale
                return [cpt], [v1] 
            else:
                # Means that this is planar so likely not an acceptor, remove it after
                return None, None

        else:
            return None, None

    elif num_lone_pairs == 2:
        heavy_nbr = heavy[0]
        if len(nbrs) == 1: # sp2
            for a in heavy_nbr.GetNeighbors():
                if a.GetIdx() != atom_id:
                    heavy_nbr_nbr = a # heavy atom's neighbor that isn't the acceptor
                    break

            pt1 = conf.GetAtomPosition(heavy_nbr_nbr.GetIdx())
            v1 = conf.GetAtomPosition(heavy_nbr.GetIdx())
            pt1 -= v1
            v1 -= cpt
            rotAxis = v1.CrossProduct(pt1)
            rotAxis.Normalize()
            bv1 = FeatDirUtils.ArbAxisRotation(120, rotAxis, v1)
            bv1.Normalize()
            bv1 *= scale
            bv2 = FeatDirUtils.ArbAxisRotation(-120, rotAxis, v1)
            bv2.Normalize()
            bv2 *= scale
            return [cpt]*2, [bv1, bv2]

        if len(nbrs) == 2: # sp3
            bvec = FeatDirUtils._findAvgVec(conf, cpt, nbrs)
            bvec *= (-1.0 * scale)
            # we will create two vectors by rotating bvec by half the tetrahedral angle in either directions
            v1 = conf.GetAtomPosition(nbrs[0].GetIdx())
            v1 -= cpt
            v2 = conf.GetAtomPosition(nbrs[1].GetIdx())
            v2 -= cpt
            rotAxis = v1 - v2
            rotAxis.Normalize()
            bv1 = FeatDirUtils.ArbAxisRotation(54.5, rotAxis, bvec)
            bv2 = FeatDirUtils.ArbAxisRotation(-54.5, rotAxis, bvec)
            bv1.Normalize()
            bv2.Normalize()
            bv1 *= scale
            bv2 *= scale
            return [cpt]*2, [bv1, bv2]
        
        else:
            return None, None
        
    elif num_lone_pairs == 3: # sp3 but do linear
        # Just do opposite of single bond (i.e., F)
        heavy_nbr = heavy[0]
        if len(hydrogens) == 0:
            v1 = conf.GetAtomPosition(heavy_nbr.GetIdx())
            v1 -= cpt
            v1.Normalize()
            v1 *= (-1.0 * scale)
            return [cpt], [v1]
    else:
        return None, None
    

def GetHalogenFeatVects(conf: Chem.rdchem.Mol,
                        featAtoms: Chem.rdchem.Atom,
                        scale: float = 1.0):
    """
    Get the anchor positions and relative unit vectors of a halogen atom.
    Assumes only one connection.

    Arguments
    ---------
    conf : rdkit Mol object with a conformer.
    featAtoms : list containing rdkit Atom object of atom attributed as an acceptor.
    scale : float (default = 1.) length of direction vector.

    Returns
    -------
    Tuple
        list of anchor position(s) as rdkit Point3D or [None]
        list of relative unit vector(s) as rdkit Point3D or [None]
    """
    assert len(featAtoms) == 1
    atom_id = featAtoms[0]

    mol = conf.GetOwningMol()
    atom = mol.GetAtomWithIdx(atom_id)
    nbrs = atom.GetNeighbors()

    cpt = conf.GetAtomPosition(atom_id)

    # Just do opposite of single bond
    heavy_nbr = nbrs[0]
    v1 = conf.GetAtomPosition(heavy_nbr.GetIdx())
    v1 -= cpt
    v1.Normalize()
    v1 *= (-1.0 * scale)
    return [cpt], [v1]


###################################################
## Modular functions from rdkit that were edited ##
## No longer used, but potentially more general ###
###################################################

# Hydrogen Bond Donors
def GetDonor1FeatVects_single(conf, featAtoms, scale=1.):
    """
    Get the direction vectors for Donor of type 1.
    Made to generate a single vector representation.
    
    This is a donor with one heavy atom. It is not clear where we should we should be putting the
    direction vector for this. It should probably be a cone. In this case we will just use the
    direction vector from the donor atom to the heavy atom.

    Changed: conditioning based on the number of hydrogens
    1. If 1 hydrogen, vector should point in the direction of the hydrogen.
    2. If 2 hydrogens, vector should point in a bisecting direction of the two hydrogens.
    3. If 3 hydrogens, point in the direction of the bond.
    
    Arguments
    ---------
    conf - rdkit Mol object with conformer
    featAtoms - list of atoms that are part of the feature
    scale - float for length of the direction vector (default = 1.0)

    Returns
    -------
    Tuple
        anchor position as rdkit Point3D or None
        relative unit vector(s) as rdkit Point3D or None
        list of hydrogen rdkit Atom objects
    """
    assert len(featAtoms) == 1
    aid = featAtoms[0]
    mol = conf.GetOwningMol()
    nbrs = mol.GetAtomWithIdx(aid).GetNeighbors()
    
    hyd_nbrs = [] # hydrogens
    heavy_nbr = -1 # hnbr in rdkit
    for nbr in nbrs:
        if nbr.GetAtomicNum() == 1:
            hyd_nbrs.append(nbr)
        else:
            heavy_nbr = nbr.GetIdx()

    cpt = conf.GetAtomPosition(aid)
    # Point in direction of hydrogen
    if len(hyd_nbrs) == 1:
        bvec = conf.GetAtomPosition(hyd_nbrs[0].GetIdx())
        bvec -= cpt
        bvec.Normalize()
        bvec *= scale
        return cpt, bvec, hyd_nbrs

    # Point in "average" (bisected) direction of 2 hydrogens
    elif len(hyd_nbrs) == 2:
        bvec = FeatDirUtils._findAvgVec(conf, cpt, hyd_nbrs)
        bvec *= scale
        # Removed conditional for generating 2 vectors for sp3 oxygen to maintain one average vector    
        return cpt, bvec, hyd_nbrs

    # Otherwise, vector is the one from donor atom to heavy atom. 3 hydrogens
    cpt = conf.GetAtomPosition(aid)
    v1 = conf.GetAtomPosition(heavy_nbr)
    v1 -= cpt
    v1.Normalize()
    v1 *= (-1.0 * scale)
    return cpt, v1, hyd_nbrs


def GetDonor2FeatVects_single(conf, featAtoms, scale=1.):
    """
    Get the direction vectors for Donor of type 2.
    Made to generate a single vector representation.
    
    This is a donor with two heavy atoms as neighbors. The atom may are may not have
    hydrogen on it. Here are the situations with the neighbors that will be considered here
    1. two heavy atoms and two hydrogens: we will assume a sp3 arrangement here
    2. two heavy atoms and one hydrogen: this can either be sp2 or sp3
    3. two heavy atoms and no hydrogens

    Changed: conditioning based on the number of hydrogens
    1. For case 1, point in the direction bisecting the two hydrogens.
    2. For case 2, point in the direction of the hydrogen.
    3. For case 3, no changes.
    
    Arguments
    ---------
    conf : rdkit Mol object with conformer
    featAtoms : list of atoms that are part of the feature
    scale : float for length of the direction vector (default = 1.0)

    Returns
    -------
    Tuple
        anchor position as rdkit Point3D or None
        relative unit vector(s) as rdkit Point3D or None
        list of hydrogen rdkit Atom objects
    """
    assert len(featAtoms) == 1
    aid = featAtoms[0]
    mol = conf.GetOwningMol()
    cpt = conf.GetAtomPosition(aid)
    
    # find the two atoms that are neighbors of this atoms
    nbrs = list(mol.GetAtomWithIdx(aid).GetNeighbors())
    assert len(nbrs) >= 2
    
    hydrogens = []
    heavy = []
    for nbr in nbrs:
        if nbr.GetAtomicNum() == 1:
          hydrogens.append(nbr)
        else:
          heavy.append(nbr)
    
    # Case 3: not sure when this would be triggered
    if len(nbrs) == 2:
        # there should be no hydrogens in this case
        assert len(hydrogens) == 0
        # in this case the direction is the opposite of the average vector of the two neighbors
        bvec = FeatDirUtils._findAvgVec(conf, cpt, heavy)
        bvec *= (-1.0 * scale)
        return cpt, bvec, None
    
    # Case 2
    if len(nbrs) == 3:
        assert len(hydrogens) == 1
        # this is a little more tricky we have to check if the hydrogen is in the plane of the
        # two heavy atoms (i.e. sp2 arrangement) or out of plane (sp3 arrangement)
        # One of the directions will be from hydrogen atom to the heavy atom
        hid = hydrogens[0].GetIdx()
        bvec = conf.GetAtomPosition(hid)
        bvec -= cpt
        bvec.Normalize()
        bvec *= scale
        if FeatDirUtils._checkPlanarity(conf, cpt, nbrs, tol=1.0e-2):
            # only the hydrogen atom direction needs to be used
            return cpt, bvec, hydrogens

        # we have a non-planar configuration - we will assume sp3 and compute a second direction vector
        # Changed since we constrain to 1 vector
        return cpt, bvec, hydrogens
    
    # Case 1
    if len(nbrs) >= 4:
        # Changed -> take the bisecting vector (average) to only use 1 vector
        vecs = []
        for hid in hydrogens:
            hid = hid.GetIdx()
            bvec = conf.GetAtomPosition(hid)
            bvec -= cpt
            vecs.append(bvec.Normalize())
        bisec_vec = sum(*vecs)
        bisec_vec.Normalize()
        return cpt, bisec_vec, hydrogens
    return None, None, None


def GetDonor3FeatVects_single(conf, featAtoms, scale=1.0):
    """
    Get the direction vectors for Donor of type 3.
    Made to generate a single vector representation.

    This is a donor with three heavy atoms as neighbors. We will assume
    a tetrahedral arrangement of these neighbors. So the direction we are seeking
    is the last fourth arm of the sp3 arrangement

    Changed: Return anchor and relative unit vector tuple

    Arguments
    ---------
    conf : rdkit Mol object with conformer
    featAtoms : list of atoms that are part of the feature
    scale : float for length of the direction vector (default = 1.0)

    Returns
    -------
    Tuple
        anchor position as rdkit Point3D or None
        relative unit vector(s) as rdkit Point3D or None
        list of hydrogen rdkit Atom objects
    """
    assert len(featAtoms) == 1
    aid = featAtoms[0]
    cpt = conf.GetAtomPosition(aid)
    out = FeatDirUtils._GetTetrahedralFeatVect(conf, aid, scale)
    nbrs = conf.GetAtomWithIdx(aid).GetNeighbors()
    hydrogens = []
    for nbr in nbrs:
        if nbr.GetAtomicNum() == 1:
            hydrogens.append(nbr)
    if out != ():
        if hydrogens == []:
            hydrogens = None
        return cpt, out[0][1] - cpt, hydrogens # need to subtract out the center
    else:
        # Means that this is planar so likely not an donor, remove it after
        return None, None, None


# Hydrogen bond acceptors
def GetAcceptor1FeatVects_single(conf, featAtoms, scale=1.):
    """
    Get the direction vectors for Acceptor of type 1.
    Made to generate a single vector representation.
    
    This is a acceptor with one heavy atom neighbor. There are two possibilities we will
    consider here
    1. The bond to the heavy atom is a single bond e.g. CO
     In this case we don't know the exact direction and we just use the inversion of this bond direction
     and mark this direction as a 'cone'
    2. The bond to the heavy atom is a double bond e.g. C=O
     In this case the we have two possible direction except in some special cases e.g. SO2
     where again we will use bond direction

    Changed: conditioning based on the number of hydrogens, and methanamine fix
    1. Case 1
        If one hydrogen, vector points in the opposite direction of the bisection of the acute
         angle formed by the heavy-acceptor-hydrogen.
        If two hydrogens, assume sp3 and project in that lone-pair direction. If not tetrahedral,
         then return None.
    2. Case 2: Return the bisecting vector of the two lone-pairs.
     
    Arguments
    ---------
    conf : rdkit Mol object with conformer
    featAtoms : list of atoms that are part of the feature
    scale : float for length of the direction vector (default = 1.0)

    Returns
    -------
    Tuple
        anchor position as rdkit Point3D or None
        relative unit vector(s) as rdkit Point3D or None
    """
    assert len(featAtoms) == 1
    aid = featAtoms[0]
    mol = conf.GetOwningMol()
    nbrs = mol.GetAtomWithIdx(aid).GetNeighbors()
    
    cpt = conf.GetAtomPosition(aid)
    
    hyd_nbrs = [] # hydrogens
    heavyAt = -1
    # find the adjacent heavy atom and hydrogens
    for nbr in nbrs:
        if nbr.GetAtomicNum() == 1:
            hyd_nbrs.append(nbr)
        else:
            heavyAt = nbr

    # before this was ``>`` which doesn't make sense
    singleBnd = mol.GetBondBetweenAtoms(aid, heavyAt.GetIdx()).GetBondType() == Chem.BondType.SINGLE

    # special scale - if the heavy atom is a sulfur (we should proabably check phosphorous as well)
    sulfur = heavyAt.GetAtomicNum() == 16
    
    methanamine = mol.GetBondBetweenAtoms(aid, heavyAt.GetIdx()).GetBondType() == Chem.BondType.DOUBLE and len(hyd_nbrs) == 1
    
    if singleBnd or sulfur or methanamine:
        if len(hyd_nbrs) == 1:
            bvec = FeatDirUtils._findAvgVec(conf, cpt, [heavyAt, hyd_nbrs[0]])
            bvec *= (-1 * scale)
            return cpt, bvec
    
        elif len(hyd_nbrs) == 2:
            out = FeatDirUtils._GetTetrahedralFeatVect(conf, aid, scale)
            if out != ():
                return cpt, out[0][1] - cpt # need to subtract out the center
            else:
                # Means that this is planar so likely not an acceptor, remove it after
                return None, None

    # Changed -> Assume sp2 (like the original code) but rather than get the two
    #  lone pair directions just do the vector along the double bond axis to use only one vector
    v1 = conf.GetAtomPosition(heavyAt.GetIdx())
    v1 -= cpt
    v1.Normalize()
    v1 *= (-1.0 * scale)
    return cpt, v1


def GetAcceptor2FeatVects_single(conf, featAtoms, scale=1.):
    """
    Get the direction vectors for Acceptor of type 2.
    Made to generate a single vector representation.
    
    This is the acceptor with two adjacent heavy atoms. We will special case a few things here.
    If the acceptor atom is an oxygen we will assume a sp3 hybridization
    the acceptor directions (two of them)
    reflect that configurations. Otherwise the direction vector in plane with the neighboring
    heavy atoms

    Changed: Only generate one vector
    Rather than generating 2 vectors for sp3 oxygen, just keep the average vector.
    
    Arguments
    ---------
    conf : rdkit Mol object with conformer
    featAtoms : list of atoms that are part of the feature
    scale : float for length of the direction vector (default = 1.0)

    Returns
    -------
    Tuple
        anchor position as rdkit Point3D or None
        relative unit vector(s) as rdkit Point3D or None
    """
    assert len(featAtoms) == 1
    aid = featAtoms[0]
    cpt = conf.GetAtomPosition(aid)
    
    mol = conf.GetOwningMol()
    nbrs = list(mol.GetAtomWithIdx(aid).GetNeighbors())
    hydrogens = []
    heavy = []
    for nbr in nbrs:
        if nbr.GetAtomicNum() == 1:
          hydrogens.append(nbr)
        else:
          heavy.append(nbr)
    
    if len(hydrogens) == 0:
        bvec = FeatDirUtils._findAvgVec(conf, cpt, heavy)
    elif len(hydrogens) == 1:
        bvec = FeatDirUtils._findAvgVec(conf, cpt, nbrs)
    bvec *= (-1.0 * scale)

    # Changed -- Removed conditional for generating 2 vectors for sp3 oxygen to maintain one average vector    
    return cpt, bvec


def GetAcceptor3FeatVects_single(conf, featAtoms, scale=1.0):
    """
    Get the direction vectors for Donor of type 3.
    Made to generate a single vector representation.

    This is a donor with three heavy atoms as neighbors. We will assume
    a tetrahedral arrangement of these neighbors. So the direction we are seeking
    is the last fourth arm of the sp3 arrangement

    Changed: to return anchor and relative unit vector tuple

    Arguments
    ---------
    conf : rdkit Mol object with conformer
    featAtoms : list of atoms that are part of the feature
    scale : float for length of the direction vector (default = 1.0)

    Returns
    -------
    Tuple
        anchor position as rdkit Point3D or None
        relative unit vector(s) as rdkit Point3D or None
    """
    assert len(featAtoms) == 1
    aid = featAtoms[0]
    cpt = conf.GetAtomPosition(aid)
    out = FeatDirUtils._GetTetrahedralFeatVect(conf, aid, scale)
    if out != ():
        return cpt, out[0][1] - cpt # need to subtract out the center to be relative
    else:
        # Means that this is planar so likely not an acceptor, remove it after
        return None, None


################### MULTI-VECTOR ##################
## THIS IS MORE TRUTHFUL TO RDKIT IMPLEMENTATION ##

# Hydrogen bond acceptors
def GetAcceptor1FeatVects(conf, featAtoms, scale=1.):
    """
    Get the direction vectors for Acceptor of type 1.
    Made to generate a multi-vector representation.
    
    This is a acceptor with one heavy atom neighbor. There are two possibilities we will
    consider here
    1. The bond to the heavy atom is a single bond e.g. CO
     In this case we don't know the exact direction and we just use the inversion of this bond direction
     and mark this direction as a 'cone'
    2. The bond to the heavy atom is a double bond e.g. C=O
     In this case the we have two possible direction except in some special cases e.g. SO2
     where again we will use bond direction

    Changed return format, added fixes for methanamine and two vectors for hydroxyls
     
    Arguments
    ---------
    conf : rdkit Mol object with a conformer.
    featAtoms : list containing rdkit Atom object of atom attributed as an acceptor.
    scale : float (default = 1.) length of direction vector.

    Returns
    -------
    Tuple
        list of anchor position(s) as rdkit Point3D or None
        list of relative unit vector(s) as rdkit Point3D or None
    """
    assert len(featAtoms) == 1
    aid = featAtoms[0]
    mol = conf.GetOwningMol()
    nbrs = mol.GetAtomWithIdx(aid).GetNeighbors()
    
    cpt = conf.GetAtomPosition(aid)
    
    hyd_nbrs = [] # hydrogens
    heavyAt = -1
    # find the adjacent heavy atom and hydrogens
    for nbr in nbrs:
        if nbr.GetAtomicNum() == 1:
            hyd_nbrs.append(nbr)
        else:
            heavyAt = nbr

    # before this was ``>`` which doesn't make sense
    singleBnd = mol.GetBondBetweenAtoms(aid, heavyAt.GetIdx()).GetBondType() == Chem.BondType.SINGLE

    # special scale - if the heavy atom is a sulfur (we should proabably check phosphorous as well)
    sulfur = heavyAt.GetAtomicNum() == 16

    methanamine = mol.GetBondBetweenAtoms(aid, heavyAt.GetIdx()).GetBondType() == Chem.BondType.DOUBLE and len(hyd_nbrs) == 1
    
    if singleBnd or sulfur or methanamine:
        if len(hyd_nbrs) == 0 or sulfur:
            v1 = conf.GetAtomPosition(heavyAt.GetIdx())
            v1 -= cpt
            v1.Normalize()
            v1 *= (-1.0 * scale)
            return [cpt], [v1]

        elif len(hyd_nbrs) == 1 and methanamine:
            bvec = FeatDirUtils._findAvgVec(conf, cpt, [heavyAt, hyd_nbrs[0]])
            bvec *= (-1 * scale)
            return [cpt], [bvec]
        
        elif len(hyd_nbrs) == 1 and singleBnd:
            # hydroxyl group
            bvec = FeatDirUtils._findAvgVec(conf, cpt, nbrs)
            bvec *= (-1.0 * scale)

            # assume sp3
            # we will create two vectors by rotating bvec by half the tetrahedral angle in either directions
            v1 = conf.GetAtomPosition(hyd_nbrs[0].GetIdx())
            v1 -= cpt
            v2 = conf.GetAtomPosition(heavyAt.GetIdx())
            v2 -= cpt
            rotAxis = v1 - v2
            rotAxis.Normalize()
            bv1 = FeatDirUtils.ArbAxisRotation(54.5, rotAxis, bvec)
            bv2 = FeatDirUtils.ArbAxisRotation(-54.5, rotAxis, bvec)
            return [cpt]*2, [bv1, bv2]
    
        elif len(hyd_nbrs) == 2:
            out = FeatDirUtils._GetTetrahedralFeatVect(conf, aid, scale)
            if out != ():
                return [cpt], [out[0][1] - cpt] # need to subtract out the center
            else:
                # Means that this is planar so likely not an acceptor, remove it after
                return None, None

    # ok in this case we will assume that
    # heavy atom is sp2 hybridized and the direction vectors (two of them)
    # are in the same plane, we will find this plane by looking for one
    # of the neighbors of the heavy atom
    hvNbrs = heavyAt.GetNeighbors()
    hvNbr = -1
    for nbr in hvNbrs:
        if nbr.GetIdx() != aid:
            hvNbr = nbr
            break

    pt1 = conf.GetAtomPosition(hvNbr.GetIdx())
    v1 = conf.GetAtomPosition(heavyAt.GetIdx())
    pt1 -= v1
    v1 -= cpt
    rotAxis = v1.CrossProduct(pt1)
    rotAxis.Normalize()
    bv1 = FeatDirUtils.ArbAxisRotation(120, rotAxis, v1)
    bv1.Normalize()
    bv1 *= scale
    bv2 = FeatDirUtils.ArbAxisRotation(-120, rotAxis, v1)
    bv2.Normalize()
    bv2 *= scale
    return [cpt]*2, [bv1, bv2]


def GetAcceptor2FeatVects(conf, featAtoms, scale=1.):
    """
    Get the direction vectors for Acceptor of type 2.
    Made to generate a single vector representation.
    
    This is the acceptor with two adjacent heavy atoms. We will special case a few things here.
    If the acceptor atom is an oxygen we will assume a sp3 hybridization
    the acceptor directions (two of them)
    reflect that configurations. Otherwise the direction vector in plane with the neighboring
    heavy atoms

    Changed: return format
    
    Arguments
    ---------
    conf : rdkit Mol object with a conformer.
    featAtoms : list containing rdkit Atom object of atom attributed as an acceptor.
    scale : float (default = 1.) length of direction vector.

    Returns
    -------
    Tuple
        list of anchor position(s) as rdkit Point3D or None
        list of relative unit vector(s) as rdkit Point3D or None
    """
    assert len(featAtoms) == 1
    aid = featAtoms[0]
    cpt = conf.GetAtomPosition(aid)
    
    mol = conf.GetOwningMol()
    nbrs = list(mol.GetAtomWithIdx(aid).GetNeighbors())
    hydrogens = []
    heavy = []
    for nbr in nbrs:
        if nbr.GetAtomicNum() == 1:
          hydrogens.append(nbr)
        else:
          heavy.append(nbr)
    
    if len(hydrogens) == 0:
        bvec = FeatDirUtils._findAvgVec(conf, cpt, heavy)
    elif len(hydrogens) == 1:
        bvec = FeatDirUtils._findAvgVec(conf, cpt, nbrs)
    bvec *= (-1.0 * scale)

    if mol.GetAtomWithIdx(aid).GetAtomicNum() == 8:
        # assume sp3
        # we will create two vectors by rotating bvec by half the tetrahedral angle in either directions
        v1 = conf.GetAtomPosition(heavy[0].GetIdx())
        v1 -= cpt
        v2 = conf.GetAtomPosition(heavy[1].GetIdx())
        v2 -= cpt
        rotAxis = v1 - v2
        rotAxis.Normalize()
        bv1 = FeatDirUtils.ArbAxisRotation(54.5, rotAxis, bvec)
        bv2 = FeatDirUtils.ArbAxisRotation(-54.5, rotAxis, bvec)
        return [cpt]*2, [bv1, bv2]
    return [cpt], [bvec]


def GetAcceptor3FeatVects(conf, featAtoms, scale=1.0):
    """
    Get the direction vectors for Donor of type 3.
    Made to generate a single vector representation.

    This is a donor with three heavy atoms as neighbors. We will assume
    a tetrahedral arrangement of these neighbors. So the direction we are seeking
    is the last fourth arm of the sp3 arrangement

    Changed: return format

    Arguments
    ---------
    conf : rdkit Mol object with a conformer.
    featAtoms : list containing rdkit Atom object of atom attributed as an acceptor.
    scale : float (default = 1.) length of direction vector.

    Returns
    -------
    Tuple
        list of anchor position(s) as rdkit Point3D or None
        list of relative unit vector(s) as rdkit Point3D or None
    """
    assert len(featAtoms) == 1
    aid = featAtoms[0]
    cpt = conf.GetAtomPosition(aid)
    out = FeatDirUtils._GetTetrahedralFeatVect(conf, aid, scale)
    if out != ():
        return [cpt], [out[0][1] - cpt] # need to subtract out the center to be relative
    else:
        # Means that this is planar so likely not an acceptor, remove it after
        return None, None