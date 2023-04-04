from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core import Molecule, Structure
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import pymatgen as pmt
import argparse
import time

def compute_voronoi_neighbor(struct, cutoff=13, d_thresh=4.0, w_thresh=0.2):
    """
        Compute Voronoi neighbors and weight from solid angle using VoronoiNN package
    Args:
        struct (pmt.Structure/Molecule): pymatgen structure or molecule
        cutoff (float, optional):   Cut off threshold for VoronoiNN. Defaults to 13.0 A.
        d_thresh (float, optional): Cut off threshold for distance. Defaults to 4.0 A.
        w_thresh (float, optional): Cut off threshold for Voronoi weight. Defaults to 0.2.

    Returns:
        list:  Neighbors information for each Local structure:
                Ex: [['H', 1, 0.4, 3.5],...]
    """

    # Initialize VoronoiNN 
    voronoi = VoronoiNN(cutoff=cutoff, allow_pathological=True)

    # Filter atoms neighbors with high Voronoi weights: solid_angle/max(solid_angle)
    local_xyz = [[
        [nn['site'].species_string, nn['site_index'], nn['weight'], np.sqrt(np.sum((struct[i].coords - nn['site'].coords)**2))]
        for nn in voronoi.get_nn_info(struct, i)
        if nn['weight'] >= w_thresh and np.sqrt(np.sum((struct[i].coords - nn['site'].coords)**2)) <= d_thresh
    ] for i in range(len(struct))]
    return local_xyz

# define a function to compute the voronoi neighbor using a single structure
def compute_voronoi_neighbor_wrapper(s, d_t, w_t):
    system_atom = s['Atoms']
    system_coord = np.array(s['Coords'], dtype='float32')

    if 'Lattice' in s:
        lattice = s['Lattice']
        struct = Structure(lattice=lattice, coords_are_cartesian=True,
                           coords=system_coord, species=system_atom)
        cutoff = 13
    else:
        struct = Molecule(system_atom, system_coord)
        size = 30
        struct = struct.get_boxed_structure(size, size, size, reorder=False)
        cutoff = 13 + size

    return compute_voronoi_neighbor(struct, cutoff, d_t, w_t)

def parallel_compute_neighbor(dataset_path, save_path,
                             d_t=4.0, w_t=0.2, pool=8):
    """
        Parallel compute Voronoi neighbors
    Args:
        dataset (dict): dictionary of  a structure with format
                    {'Atoms':[....],'Coords':[....],'Lattice':[....],'Properties':[...]}
        savepath (str): Path to save neighbor information for dataset'
        d_t (float, optional): Cut off distance for compute neighbor. Defaults to 4.0.
        w_t (float, optional): Cut off Voronoi weight for compute neighbor. Defaults to 0.2.
        pool (int, optional): Parallel process for computing. Defaults to 8.
    """

    dataset = np.load(dataset_path, allow_pickle=True)

    all_data = []
    print('Computing Voronoi neighbor for dataset ', dataset_path,' using ', pool,' parallel process')
    
    #Using multiprocess for faster computing, change the number process based on system capacity
    with ProcessPoolExecutor(pool) as executor:
        for i in range(0, len(dataset), pool):
            print(i)
            batch = dataset[i:i+pool]
            batch_futures = [executor.submit(compute_voronoi_neighbor_wrapper, s,d_t,w_t) for s in batch]
            batch_results = [future.result() for future in batch_futures]
            all_data.extend(batch_results)
            
    np.save(save_path, all_data)
    print('Finished computing Voronoi neighbor for dataset ', dataset_path)
    