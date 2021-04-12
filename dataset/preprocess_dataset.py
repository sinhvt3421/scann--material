from pymatgen.analysis.local_env import VoronoiNN
import numpy as np
import pandas
import ase
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import pymatgen as pmt
import time
from scipy.spatial import distance_matrix

def compute_voronoi(data: dict,datatype: str) -> list:
    """
    Compute Voronoi neighbors and weight from solid angle using VoronoiNN package
    Params  
        data: dictionary of  a structure with format
                 {'Atoms':[....],'Coords':[....],'Lattice':[....],'Properties':[...]}
        datatype: Molecule or Crystal
    Return
        local_xyz: list of atoms centers with its neighbors and information
    """
    if datatype == 'C':
        system = data['Atoms']
        system_coord = np.array(data['Coords'], dtype='float32') 
        struct = pmt.Structure(lattice=data['Lattice'],
                            coords=system_coord, species=system, coords_are_cartesian=True)
    else:
        system = data['Atoms']
        system_coord = data['Coords']
        mol = pmt.Molecule(system, system_coord)
        size = 10
        mol = mol.get_boxed_structure(size, size, size, reorder=False)
        cutoff = 13 + size

    local_xyz = []
    for i in range(len(system)):
        atom_xyz = []
        site = struct[i]

        voronoi = VoronoiNN(cutoff=cutoff, allow_pathological=True)
        neighbors = voronoi.get_nn_info(struct,i)
   
        for nn in neighbors:
            site_x = nn['site']
            weight_voronoi = nn['weight']
            solid_angle = nn['poly_info']['solid_angle']
            neighbor_index = nn['site_index']
            face_distance = nn['poly_info']['face_dist']
            # Filter atoms neighbors with high Voronoi weights: solid_angle/max(solid_angle)
            if weight_voronoi > 0.1:
                distance = np.sqrt(np.sum((site.coords - site_x.coords)**2))
                site_x_label = site_x.species_string

                atom_xyz += [(site_x_label, weight_voronoi, neighbor_index, distance,
                              solid_angle, face_distance)]

        local_xyz.append(atom_xyz)

    return local_xyz


def main(args):
    dataset = np.load(args['path'],allow_pickle=True)

    all_data = []
    future = []
    #Using multiprocess for faster computing, change the number process based on system capacity
    with ProcessPoolExecutor(8) as excutor:
        i = 0
        print(i)
        for data in dataset:
            future.append(excutor.submit(get_neighboor_2, data, args['type']))
            i += 1
            if i % 16 == 0:
                for f in future:
                    all_data.append(f.result())
                future.clear()
                print(i)

        for f in future:
            all_data.append(f.result())
        future.clear()

    np.save(args['save_path'], all_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset for computing Voronoi neighbors and weights')
    parser.add_argument('destpath', type=str,
                        help='Destiation path dataset')
    parser.add_argument('savepath', type=str,
                        help='Save path processed dataset')
    parser.add_argument('--type', default='M',help='Dataset type M-molecule or C-crystal')
    args = parser.parse_args()
    main(args)
