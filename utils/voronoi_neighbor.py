from pymatgen.analysis.local_env import VoronoiNN
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import pymatgen as pmt
import argparse

def compute_voronoi_neighbor(d_thresh=4.0, w_thresh=0.2, 
                            system_atoms=None, system_coord=None,lattice=None):
    """
        Compute Voronoi neighbors and weight from solid angle using VoronoiNN package
        Args:
            d_thresh: Cut off threshold for distance
            w_thresh: Cut off threshold for Voronoi weight
            system_atoms: List of atoms name in the structure
            system_coord: 3D coordinates of all atoms in the structure
            lattice: Lattice information for crystal material
        Return:
            Neighbors information for each Local structure:
            Ex: [['H', 1, 0.4, 3.5],...]
    """
    if lattice:
        lattice_info = lattice
        struct = pmt.Structure(lattice=lattice_info, coords_are_cartesian=True,
                           coords=system_coord, species=system_atoms)
    else:
        struct = pmt.Molecule(system_atoms, system_coord)
        size = 30
        struct = struct.get_boxed_structure(size, size, size, reorder=False)
        cutoff = 13 + size

    local_xyz = []
    for i in range(len(system_atoms)):
        atom_xyz = []
        site = struct[i]

        voronoi = VoronoiNN(cutoff=cutoff, allow_pathological=True)
        neighbors = voronoi.get_nn_info(struct,i)
   
        for nn in neighbors:
            site_x = nn['site']
            w = nn['weight']

             # Filter atoms neighbors with high Voronoi weights: solid_angle/max(solid_angle)
            if (w > w_thresh):
                d = np.sqrt(np.sum((site.coords - site_x.coords)**2))

                # Filter atoms neighbors with small distance
                if d < d_thresh:
                    site_x_label = site_x.species_string
                    atom_xyz += [(site_x_label, nn['site_index'], w, d)]

        local_xyz.append(atom_xyz)

    return local_xyz

def main(args):
    """
        dataset: dictionary of  a structure with format
                 {'Atoms':[....],'Coords':[....],'Lattice':[....],'Properties':[...]}
    """
    dataset = np.load(args.dataset, allow_pickle=True)

    all_data = []
    future = []
    print('Computing Voronoi neighbor for dataset')
    
    #Using multiprocess for faster computing, change the number process based on system capacity
    with ProcessPoolExecutor(8) as excutor:
        i = 0
        print(i)
        for d in dataset:
            system_atom = d['Atoms']
            system_coord = np.array(d['Coords'], dtype='float32')

            if 'Lattice' in d:
                lattice = d['Lattice']
            else:
                lattice = None

            future.append(excutor.submit(compute_voronoi_neighbor, 
                                        args.d_t, args.w_t, 
                                        system_atom, system_coord, lattice))
            i += 1
            if i % 16 == 0:
                for f in future:
                    all_data.append(f.result())
                future.clear()
                print(i)

        for f in future:
            all_data.append(f.result())
        future.clear()

    np.save(args.savepath, all_data)
    print('Finished computing Voronoi neighbor for dataset')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('dataset', type=str, help='Path to dataset')
    parser.add_argument('savepath', type=str, help='Path to save neighbor information for dataset')
    parser.add_argument('--d_t', type=float, default=4.0,
                    help='Cut off distance for compute neighbor')
    parser.add_argument('--w_t', type=float, default=0.2,
                    help='Cut off Voronoi weight for compute neighbor')

    args = parser.parse_args()
    main(args)
