"""
Copied from: https://github.com/coleygroup/shepherd-score/blob/main/shepherd_score/generate_point_cloud.py

Module contains functions to generate point clouds of
molecular surfaces.

Surface generation workflow adapted from Open Drug Discovery Toolkit:
    https://oddt.readthedocs.io/en/latest/_modules/oddt/surface.html
"""
from typing import Tuple, List

import numpy as np
#from shepherd_score.score.constants import COULOMB_SCALING
COULOMB_SCALING = 1e4/(4*55.263*np.pi)
import open3d as o3d

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import distance
from typing import Union

PT = Chem.GetPeriodicTable()


def get_atom_coords(mol: rdkit.Chem.Mol,
                    MMFF_optimize: bool = True
                    ) -> Tuple[Chem.Mol, np.ndarray]:
    """
    Get the coordinates of all atoms in a molecule using rdkit.
    If the rdkit.Chem.mol object already has a conformer it just retrieves the coordinates
    without optimizing using MMFF.

    Parameters
    ----------
    mol : rdkit.Chem.Mol object
        RDKit molecule object
    
    MMFF_optimize : bool
        Whether or not to use MMFF to optimize geometry
    
    Returns
    -------
    tuple
        rdkit.Chem.Mol
            Mol object. If the input mol did not have a conformer, it uses MMFF to optimize an
            embeded molecule and includes hydrogens.
        np.ndarray: shape = (N,3)
            Positions of each atom's center.
    """
    try:
        mol.GetConformer()
    except ValueError:
        try:
            mol = Chem.AddHs(mol)
            Chem.AllChem.EmbedMolecule(mol, maxAttempts = 200)
            if MMFF_optimize:
                Chem.AllChem.MMFFOptimizeMolecule(mol)

            mol.GetConformer() # test whether conformer generation succeeded
        except Exception as e:
            print('Failed to embed molecule:', e)
            return None
    return mol, mol.GetConformer().GetPositions()


def get_atomic_vdw_radii(mol: rdkit.Chem.Mol) -> np.ndarray:
    """
    Get the van der Waals radii of all atoms in a molecule using rdkit.

    Parameters
    ----------
    mol : rdkit.Chem.Mol object
    
    Returns
    -------
    np.ndarray : shape = (N,)
        vdW radii for each atom.
    """
    radii = np.zeros((mol.GetNumAtoms(),))
    for i, _ in enumerate(radii):
        # get the van der Waals radii of each atom
        radii[i] = PT.GetRvdw(mol.GetAtomWithIdx(i).GetAtomicNum())
    return radii


###################################
# Sampling from molecular surface #
###################################
def sample_molecular_surface_with_radius(centers: np.ndarray,
                                         radii: Union[np.ndarray, List],
                                         probe_radius: float = 1.2,
                                         num_samples_per_atom: int = 20
                                         ) -> np.ndarray:
    """
    Samples points from the surface of vdW radius of atoms and combines it into one molecule (Vectorized).

    Parameters
    ----------
    centers : np.ndarray (N, 3)
        Cartesian coordinates of the atom centers of a molecule.
    
    radii : np.ndarray (N,)
        van der Waals radii of each atom (in Angstrom) in the same order as the centers parameter.
    
    probe_radius : float (default = 1.2)
        The radius of a probe atom to act as a "solvent accessible surface".
        Default = 1.2 angstroms which is the radius of a Hydrogen atom.
    
    num_samples_per_atom : int (default = 20)
        Number of points to sample from the surface of each atom.
        Note that this value is scaled by a given atom's relative vdW radius to a carbon and
        SQUARED. Typically choose a value between 15 and 35.
            E.g., if set to 20, a carbon atom would have 400 sampled points.
    
    Returns
    -------
    np.ndarray (N*num_points_per_atom, 3)
        Array containing the coordinates of each point sampled from each atom.
    """
    # get surface radius based on vdW radii, cutoff
    cutoff = 1.4
    if num_samples_per_atom > 50:
        raise ValueError('Do not set num_samples_per_atom to be larger than 50 for performance\
                         issues. The number is squared internally.')
    if not isinstance(radii, np.ndarray):
        radii = np.array(radii)
    # surf_radii = cutoff * (radii / 1.7) + probe_radius # scaled vdW radius + probe radius
    surf_radii = radii + probe_radius # vdW radius + probe radius
    # get number of samples per atom dependent on vdw radii
    num_samples_per_atom = np.ceil((num_samples_per_atom * (radii / 1.7))**2)

    # Sample unit surfaces from normal distribution
    x = np.random.normal(size=(int(num_samples_per_atom.sum()), 3))
    # reformat radii and center arrays to match number of samples per atom
    Rs = np.repeat(surf_radii.reshape((-1,1)), [int(n) for n in num_samples_per_atom], axis=0)
    centers = np.repeat(centers, [int(n) for n in num_samples_per_atom], axis=0)

    # Scale surfaces by radii and translate to centers
    surface = ((x / np.linalg.norm(x, axis=1)[:, np.newaxis]) * Rs) + centers
    return surface


def sample_molecular_surface_with_radius_fibonacci(centers: np.ndarray,
                                                   radii: Union[np.ndarray, List],
                                                   probe_radius: float = 1.2,
                                                   num_samples_per_atom: int = 20
                                                   ) -> np.ndarray:
    """
    Samples points from the surface of vdW radius of atoms and combines it into one molecule (Vectorized).

    Parameters
    ----------
    centers : np.ndarray (N, 3)
        Cartesian coordinates of the atom centers of a molecule.
    
    radii : np.ndarray (N,)
        van der Waals radii of each atom (in Angstrom) in the same order as the centers parameter.
    
    probe_radius : float (default = 1.2)
        The radius of a probe atom to act as a "solvent accessible surface".
        Default = 1.2 angstroms which is the radius of a Hydrogen atom.
    
    num_samples_per_atom : int (default = 20)
        Number of points to sample from the surface of each atom.
        Note that this value is scaled by a given atom's relative vdW radius to a carbon and
        SQUARED. Typically choose a value between 15 and 35.
            E.g., if set to 20, a carbon atom would have 400 sampled points.
    
    Returns
    -------
    np.ndarray (N*num_points_per_atom, 3)
        Array containing the coordinates of each point sampled from each atom.
    """
    # get surface radius based on vdW radii, cutoff
    cutoff = 1.4
    if num_samples_per_atom > 50:
        raise ValueError('Do not set num_samples_per_atom to be larger than 50 for performance\
                         issues. The number is squared internally.')
    if not isinstance(radii, np.ndarray):
        radii = np.array(radii)
    # surf_radii = cutoff * (radii / 1.7) + probe_radius # scaled vdW radius + probe radius
    surf_radii = radii + probe_radius # vdW radius + probe radius
    # get number of samples per atom dependent on vdw radii
    num_samples_per_atom = np.ceil((num_samples_per_atom * (radii / 1.7))**2)

    # store points for spheres generated by same sized radii since deterministic
    spheres = {num_samples : _get_points_fibonacci(num_samples) for num_samples in set(num_samples_per_atom)}
    # Apply a random SO(3) rotation to each atom's sphere seperately
    # spheres = np.vstack([np.dot(spheres[num_samples], special_ortho_group.rvs(3).T) for num_samples in num_samples_per_atom])
    # Don't apply random SO(3) rotation since mesh sampling is stochastic and 5-8x faster.
    spheres = np.vstack([spheres[num_samples] for num_samples in num_samples_per_atom])
    
    # reformat radii and center arrays to match number of samples per atom for elementwise mult and add
    Rs = np.repeat(surf_radii.reshape((-1,1)), [int(n) for n in num_samples_per_atom], axis=0)
    centers = np.repeat(centers, [int(n) for n in num_samples_per_atom], axis=0)

    # Scale surfaces by radii and translate to centers
    surface = (spheres * Rs) + centers
    return surface


def _get_points_fibonacci(num_samples: int):
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


def get_point_cloud(points: np.ndarray,
                    color: List[float] = [0.0, 0.0, 0.0]
                    ) -> o3d.geometry.PointCloud:
    """
    Convert np.ndarray of points to a Open3D Point Cloud object.

    Parameters
    ----------
    points : np.ndarray (N, 3)
        Coordinates of points.
        
    color : list[float] (default=[0,0,0] (black))

    Returns
    -------
    open3d.geometry.PointCloud object
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.paint_uniform_color(color)
    return pcd


def _get_molecular_surface_mesh(centers: np.ndarray,
                                radii: Union[np.ndarray, List],
                                num_samples_per_atom: int = 25,
                                probe_radius: float = 1.2,
                                ball_radii: List[float] = [1.2],
                                color: List[float] = [1.0, 0.0, 0.0]
                                ) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]:
    """
    Generate a surface mesh representation of a molecule's surface. The dense point cloud is also
    returned.

    Parameters
    ----------
    centers : np.ndarray (N, 3)
        Cartesian coordinates of the atom centers of a molecule.
    
    radii : np.ndarray (N,)
        van der Waals radii of each atom (in Angstrom) in the same order as the centers parameter.
    
    num_samples_per_atom : int (default = 20)
        Number of points to sample from the surface of each atom.
        Note that this value is scaled by a given atom's relative vdW radius to a carbon and
        SQUARED. Typically choose a value between 15 and 35.
            E.g., if set to 20, a carbon atom would have 400 sampled points.

    probe_radius : float (default = 1.2)
        The radius of a probe atom to act as a "solvent accessible surface".
        Default = 1.2 angstroms which is the radius of a Hydrogen atom.
        
    ball_radii : list[float] (default = [1.2])
        The radius of the ball(s) used in Open3D's ball pivoting algorithm to generate a triangle
        mesh.

    color : list[float] (default = [1., 0., 0.])
        RGB color values for the point cloud (default is red).
    
    Returns
    -------
    tuple
        o3d.geometry.TriangleMesh : Mesh representing the molecular surface.
        o3d.geometry.PointCloud : Dense point cloud representing the molecular surface.
    """
    points = sample_molecular_surface_with_radius_fibonacci(centers=centers,
                                                            radii=radii,
                                                            probe_radius=probe_radius,
                                                            num_samples_per_atom=num_samples_per_atom
                                                            )
    # distances of every point with respect to the centers of each atom
    dist_matrix = distance.cdist(points, centers)
    # mask out the points within vdw radius of each atom
    mask = np.where(np.all(dist_matrix >= radii + probe_radius - 0.01, axis=1), 1., 0.).astype(bool)

    # generate point cloud
    pcd = get_point_cloud(points[mask], color=color)

    # Generate surface mesh and sample from it evenly
    pcd.estimate_normals()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(ball_radii))
    return mesh, points


def get_molecular_surface_point_cloud(centers: np.ndarray,
                                      radii: Union[np.ndarray, List],
                                      num_points: Union[int, None] = None,
                                      num_samples_per_atom: int = 25,
                                      probe_radius: float = 1.2,
                                      ball_radii: List[float] = [1.2],
                                      color: List[float] = [1.0, 0.0, 0.0]
                                      ) -> o3d.geometry.PointCloud:
    """
    Gets the point cloud representation of a molecule's van der Waals surface. Takes into account
    the vdW radii of different atoms. Removes overlapping points within vdW radii of neighboring
    atoms.

    Parameters
    ----------
    centers : np.ndarray (N, 3)
        Cartesian coordinates of the atom centers of a molecule.
    
    radii : np.ndarray (N,)
        van der Waals radii of each atom (in Angstrom) in the same order as the centers parameter.

    num_points : int (default = None)
        The total number of points in the final point cloud. If None, it returns as many as what
        was left after cleaning up the atom-sampled surface point cloud.
    
    num_samples_per_atom : int (default = 25)
        Number of points to sample from the surface of each atom.
        Note that this value is scaled by a given atom's relative vdW radius to a carbon and
        SQUARED. Typically choose a value between 15 and 35.
            E.g., if set to 20, a carbon atom would have 400 sampled points.

    probe_radius : float (default = 1.2)
        The radius of a probe atom to act as a "solvent accessible surface".
        Default = 1.2 angstroms which is the radius of a Hydrogen atom.

    ball_radii : list[float] (default = [1.2])
        The radius of the ball(s) used in Open3D's ball pivoting algorithm to generate a triangle
        mesh.

    color : list[float] (default = [1., 0., 0.])
        RGB color values for the point cloud (default is red).
    
    Returns
    -------
    o3d.geometry.PointCloud
        Point cloud object representation of the molecular surface.
    """
    mesh, points = _get_molecular_surface_mesh(centers=centers,
                                               radii=radii,
                                               num_samples_per_atom=num_samples_per_atom,
                                               probe_radius=probe_radius,
                                               ball_radii=ball_radii,
                                               color=color)
    if num_points is None:
        num_points = len(points)
    pcd = mesh.sample_points_poisson_disk(num_points)
    return pcd


def get_molecular_surface(centers:np.ndarray,
                          radii:Union[np.ndarray, List],
                          num_points:Union[int, None] = None,
                          num_samples_per_atom:int = 25,
                          probe_radius: float = 1.2,
                          ball_radii: List[float] = [1.2],
                          ) -> np.ndarray:
    """
    Gets the point cloud representation of a molecule's van der Waals surface and outputs a
    numpy array. Takes into account the vdW radii of different atoms. Removes overlapping points
    within vdW radii of neighboring atoms.

    Parameters
    ----------
    centers : np.ndarray (N, 3)
        Cartesian coordinates of the atom centers of a molecule.
    
    radii : np.ndarray (N,)
        van der Waals radii of each atom (in Angstrom) in the same order as the centers parameter.

    num_points : int (default = None)
        The total number of points in the final point cloud. If None, it returns as many as what
        was left after cleaning up the atom-sampled surface point cloud.
    
    num_samples_per_atom : int (default = 25)
        Number of points to sample from the surface of each atom.
        Note that this value is scaled by a given atom's relative vdW radius to a carbon and
        SQUARED. Typically choose a value between 15 and 35.
            E.g., if set to 20, a carbon atom would have 400 sampled points.

    probe_radius : float (default = 1.2)
        The radius of a probe atom to act as a "solvent accessible surface".
        Default = 1.2 angstroms which is the radius of a Hydrogen atom.
        
    ball_radii : list[float] (default = [1.2])
        The radius of the ball(s) used in Open3D's ball pivoting algorithm to generate a triangle
        mesh.
    
    Returns
    -------
    np.ndarray
        Coordinates of points representing the molecular surface.
    """
    pcd = get_molecular_surface_point_cloud(centers=centers, radii=radii,
                                            num_points=num_points, 
                                            num_samples_per_atom=num_samples_per_atom,
                                            probe_radius=probe_radius,
                                            ball_radii=ball_radii)
    return np.asarray(pcd.points)


def get_molecular_surface_point_cloud_const_density(centers: np.ndarray,
                                                    radii: Union[np.ndarray, List],
                                                    density: float = 0.3,
                                                    num_samples_per_atom: int = 25,
                                                    probe_radius: float = 1.2,
                                                    ball_radii: List[float] = [1.2],
                                                    color: List[float] = [1.0, 0.0, 0.0]
                                                    ) -> o3d.geometry.PointCloud:
    """
    Gets the point cloud representation of a molecule's van der Waals surface. Takes into account
    the vdW radii of different atoms. Removes overlapping points within vdW radii of neighboring
    atoms.

    Parameters
    ----------
    centers : np.ndarray (N, 3)
        Cartesian coordinates of the atom centers of a molecule.
    
    radii : np.ndarray (N,)
        van der Waals radii of each atom (in Angstrom) in the same order as the centers parameter.

    density : float (default = 0.3)
        The density of points on the surface. The number of points is calculated from the solvent
        accessible surface area approximately computed by the surface area of the generated mesh.
    
    num_samples_per_atom : int (default = 35)
        Number of points to sample from the surface of each atom.
        Note that this value is scaled by a given atom's relative vdW radius to a carbon and
        SQUARED. Typically choose a value between 15 and 35.
            E.g., if set to 20, a carbon atom would have 400 sampled points.

    probe_radius : float (default = 1.2)
        The radius of a probe atom to act as a "solvent accessible surface".
        Default = 1.2 angstroms which is the radius of a Hydrogen atom.

    ball_radii : list[float] (default = [1.2])
        The radius of the ball(s) used in Open3D's ball pivoting algorithm to generate a triangle
        mesh.

    color : list[float] (default = [1., 0., 0.])
        RGB color values for the point cloud (default is red).
    
    Returns
    -------
    o3d.geometry.PointCloud
        Point cloud object representation of the molecular surface.
    """
    mesh, _ = _get_molecular_surface_mesh(centers=centers,
                                          radii=radii,
                                          num_samples_per_atom=num_samples_per_atom,
                                          probe_radius=probe_radius,
                                          ball_radii=ball_radii,
                                          color=color)
    # solv_acc_surf_area = rdFreeSASA.CalcSASA(mol, radii) # solvent accessible surface area
    solv_acc_surf_area = mesh.get_surface_area() # Approximate solvent accessible surface area
    num_points = int(density * solv_acc_surf_area)
    pcd = mesh.sample_points_poisson_disk(num_points)
    return pcd


def get_molecular_surface_const_density(centers:np.ndarray,
                                        radii:Union[np.ndarray, List],
                                        density: float = 0.3,
                                        num_samples_per_atom:int = 25,
                                        probe_radius: float = 1.2,
                                        ball_radii: List[float] = [1.2],
                                        ) -> np.ndarray:
    """
    Gets the point cloud representation of a molecule's van der Waals surface and outputs a
    numpy array. Takes into account the vdW radii of different atoms. Removes overlapping points
    within vdW radii of neighboring atoms.

    Parameters
    ----------
    centers : np.ndarray (N, 3)
        Cartesian coordinates of the atom centers of a molecule.
    
    radii : np.ndarray (N,)
        van der Waals radii of each atom (in Angstrom) in the same order as the centers parameter.

    density : float (default = 0.3)
        The density of points on the surface. The number of points is calculated from the solvent
        accessible surface area approximately computed by the surface area of the generated mesh.
    
    num_samples_per_atom : int (default = 20)
        Number of points to sample from the surface of each atom.
        Note that this value is scaled by a given atom's relative vdW radius to a carbon and
        SQUARED. Typically choose a value between 15 and 35.
            E.g., if set to 20, a carbon atom would have 400 sampled points.

    probe_radius : float (default = 1.2)
        The radius of a probe atom to act as a "solvent accessible surface".
        Default = 1.2 angstroms which is the radius of a Hydrogen atom.
        
    ball_radii : list[float] (default = [1.2])
        The radius of the ball(s) used in Open3D's ball pivoting algorithm to generate a triangle
        mesh.
    
    Returns
    -------
    np.ndarray
        Coordinates of points representing the molecular surface.
    """
    pcd = get_molecular_surface_point_cloud_const_density(centers=centers,
                                                          radii=radii,
                                                          density=density, 
                                                          num_samples_per_atom=num_samples_per_atom,
                                                          probe_radius=probe_radius,
                                                          ball_radii=ball_radii)
    return np.asarray(pcd.points)


def get_electrostatics(mol: Chem.Mol, points: np.ndarray) -> np.ndarray:
    """
    Compute the Coulomb potential values at each point for a given molecule.
    Assumes the input "mol" already has an optimized conformer. Gets partial charges from
    MMFF or Gasteiger.

    Parameters
    ----------
    mol : rdkit.Chem.Mol object
        RDKit molecule object with an optimized geometry in conformers.
    
    points : np.ndarray (N, 3)
        Coordinates of sampled points to compute Coulomb potential at.

    Returns
    -------
    np.ndarray (N)
        Electrostatic potential values corresponding to each point.
    """
    try:
        mol.GetConformer()
    except ValueError as e:
        raise ValueError(f"Provided rdkit.Chem.Mol object did not have conformer embedded.", e)

    molec_props = Chem.AllChem.MMFFGetMoleculeProperties(mol)
    if molec_props:
        charges = np.array([molec_props.GetMMFFPartialCharge(i) for i, _ in enumerate(mol.GetAtoms())])
    else:
        print("MMFF charges not available for the input molecule, defaulting to Gasteiger charges.")
        AllChem.ComputeGasteigerCharges(mol)
        charges=np.array([a.GetDoubleProp('_GasteigerCharge') for a in mol.GetAtoms()])
    
    centers = mol.GetConformer().GetPositions()

    distances = np.linalg.norm(points[:, np.newaxis] - centers, axis=2)
    # Calculate the potentials
    E_pot = np.dot(charges, 1 / distances.T) * COULOMB_SCALING
    # Ensure that invalid distances (where distance is 0) are handled
    E_pot[np.isinf(E_pot)] = 0    
    return E_pot


def get_electrostatics_given_point_charges(charges: np.ndarray, positions: np.ndarray, points: np.ndarray)-> np.ndarray:
    """
    Compute the Coulomb potential values at each point for a given set of charges at defined positions.

    Parameters
    ----------
    charges : np.ndarray (N,)
        Charges, with units [V]
    
    positions : np.ndarray (N, 3)
        Coordinates of point charges, with units of A.
        
    points : np.ndarray (M, 3)
        Coordinates of point cloud at which to compute electrostatic potential, with units of A.

    Returns
    -------
    np.ndarray (M,)
        Electrostatic potential values corresponding to each point.
    """
    
    distances = np.linalg.norm(points[:, np.newaxis] - positions, axis=2)
    # Calculate the potentials
    E_pot = np.dot(charges, 1. / distances.T) * COULOMB_SCALING # [eV/e] = [V]
    # Ensure that invalid distances (where distance is 0) are handled
    E_pot[np.isinf(E_pot)] = 0    
    return E_pot


def color_pcd_with_electrostatics(pcd: o3d.geometry.PointCloud, E_pot: np.ndarray) -> None:
    """
    Color the point cloud based on elecrostatic potential. Colors are only scaled to the molecule
    itself (i.e., the colors are not comparable to different molecules).
    Red is positive, blue is negative, black is neutral.
    """

    colors = np.zeros((len(E_pot), 3))
    colors[:,0] = np.where(E_pot < 0, 0, E_pot/np.max(E_pot)).squeeze()
    colors[:,2] = np.where(E_pot >= 0, 0, -E_pot/np.max(-E_pot)).squeeze()
    pcd.colors = o3d.utility.Vector3dVector(colors)


def get_sample_atom_volume(R: float, num_samples: int) -> np.ndarray:
    """
    Sample points uniformly from a sphere.
    https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume

    Parameters
    ----------
    R : float
        Radius
    num_samples : int
        Number of samples
    
    Returns
    -------
    np.ndarray (num_samples, 3)
    """
    phi = np.random.uniform(0, 2*np.pi, num_samples)
    cos_theta = np.random.uniform(-1, 1, num_samples)
    u = np.random.uniform(0,1, num_samples)
    
    theta = np.arccos(cos_theta)
    r = R * u**(1/3)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.column_stack((x, y, z))


def get_molecular_volume(centers: np.ndarray, radii: np.ndarray, num_samples: int = None, num_samples_per_atom: int=15):
    """
    Sample points uniformly from a molecule's volume. Does not include probe radius.
    Ignores possible overlap of vdW radius of atoms when randomly sub-selecting points for the
    volume.

    Parameters
    ----------
    centers : np.ndarray (N, 3)
        Cartesian coordinates of the atom centers of a molecule.
    
    radii : np.ndarray (N,)
        van der Waals radii of each atom (in Angstrom) in the same order as the centers parameter.

    num_samples_per_atom : int (default = 15)
        Number of points to sample from the surface of each atom.
        Note that this value is scaled by a given atom's relative vdW radius to a carbon and
        CUBED. Typically choose a value between 10 and 20.
            E.g., if set to 20, a carbon atom would have 8000 sampled points.
    
    Returns
    -------
    np.ndarray
        Coordinates of points representing the molecular surface.
    """
    if num_samples_per_atom > 20:
        raise ValueError('Do not set num_samples_per_atom to be larger than 20 for performance\
                         issues. The number is cubed internally.')
    cutoff = 1.4
    # radii = cutoff * (radii / 1.7)
    num_samples_per_atom = np.ceil((num_samples_per_atom * (radii / 1.7))**3)

    points = np.vstack([get_sample_atom_volume(r, int(num_samples_per_atom[i])) for i, r in enumerate(radii)])
    centers = np.repeat(centers, [int(n) for n in num_samples_per_atom], axis=0)
    points += centers # translate atoms to their centers

    # Subselect points by masking
    if num_samples is not None:
        idx = np.arange(len(points))
        np.random.shuffle(idx)
        if num_samples > len(points):
            num_samples = len(points)
        points = points[idx[:num_samples]]
    return points