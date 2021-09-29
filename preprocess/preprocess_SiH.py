from pymatgen.analysis.local_env import VoronoiNN
import numpy as np
import pandas
import ase
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import pymatgen as pmt
import time
from scipy.spatial import distance_matrix


def get_neighboor(step):
    system = [x[0] for x in step[0]]
    system_coord = np.array([x[1:4] for x in step[0]], dtype='float64') + 200
    struct = pmt.Structure(lattice=[[1000, 0, 0], [0, 1000, 0], [0, 0, 1000]],
                           coords=system_coord, species=system, coords_are_cartesian=True)

    atoms = np.array([site.species_string for site in struct])

    local_xyz = []
    for i_atom, atom in enumerate(atoms):

        coordinator_finder = VoronoiNN(allow_pathological=True, cutoff=26.0)
        neighbors = coordinator_finder.get_nn_info(structure=struct, n=i_atom)

        site = struct[i_atom]
        env_vector = []

        atom_xyz = []
        for nn in neighbors:
            site_x = nn['site']
            w = nn['weight']
            site_x_label = site_x.species_string
            d = np.sqrt(np.sum((site.coords - site_x.coords)**2))
            if w > 0.01:
                atom_xyz += [(site_x_label, w, nn['site_index'], d,
                              nn['poly_info']['solid_angle'], nn['poly_info']['face_dist'])]

        local_xyz.append(atom_xyz)
    return local_xyz


def get_neighboor_2(step):
    system = step['Atoms']
    system_coord = np.array(step['Coords'], dtype='float32') 
    min_coord = np.min(system_coord,0)
    system_coord_new = system_coord - min_coord
    struct = pmt.Structure(lattice=[[30, 0, 0], [0, 30, 0], [0, 0, 30]],
                           coords=system_coord_new, species=system, coords_are_cartesian=True)

    size = 30
    cutoff = 13 + size

    local_xyz = []
    for i in range(len(system)):
        atom_xyz = []
        site = struct[i]

        voronoi = VoronoiNN(cutoff=cutoff, allow_pathological=True)
        neighbors = voronoi.get_nn_info(struct,i)
   
        for nn in neighbors:
            site_x = nn['site']
            w = nn['weight']
            # if (w > 0.2) & (np.min(site_x.coords) > 0) & (np.max(site_x.coords) < size):
            if (w > 0.2) & (np.min(site_x.coords) > 0) & (np.max(site_x.coords) < size):
                d = np.sqrt(np.sum((site.coords - site_x.coords)**2))
                site_x_label = site_x.species_string
                atom_xyz += [(site_x_label, w, nn['site_index'], d,
                              nn['poly_info']['solid_angle'], nn['poly_info']['face_dist'])]

        local_xyz.append(atom_xyz)

    return local_xyz


def main():
    new_step = np.load('preprocess/ptcnt/pt_full_info.npy',
                       allow_pickle=True)

    all_data = []
    future = []
    with ProcessPoolExecutor(8) as excutor:
        i = 0
        print(i)
        for step in new_step:
            future.append(excutor.submit(get_neighboor_2, step))
            i += 1
            if i % 16 == 0:
                for f in future:
                    all_data.append(f.result())
                future.clear()
                print(i)

        for f in future:
            all_data.append(f.result())
        future.clear()

    np.save('preprocess/ptcnt/pt_data_voroinn_neigh_7k.npy', all_data)


if __name__ == '__main__':
    main()
