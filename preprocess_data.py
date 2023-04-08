import logging
import os
import numpy as np
from utils.dataset import *
import argparse
from utils.voronoi_neighbor import parallel_compute_neighbor

# Define dictionary mapping dataset names to functions
dataset_functions = {
    'qm9': process_qm9,
    'fullerene': process_fullerene,
    'ptgp': process_gp,
    'smfe': process_smfe,
    'ternary': process_ternary
}

def init_dataset(dataset='qm9', data_path='', save_path='', d_t=4.0, w_t=0.2, p=8):

    # Call the appropriate function based on the dataset name
    if dataset in dataset_functions:
        print(f'Init dataset {dataset}:')
        if not os.path.exists(os.path.join(save_path, dataset)):
            dataset_functions[dataset](data_path, save_path)
    else:
        print(f'Dataset {dataset} is not recognized.')

    parallel_compute_neighbor(dataset_path=os.path.join(save_path, dataset, dataset+'_data_energy.npy'),
                              save_path=os.path.join(
                                  save_path, dataset, '{}_data_neighbor_dt{}_wt{}.npy'.format(dataset, d_t, w_t)),
                              d_t=d_t, w_t=w_t, pool=p)

def main(args):

    init_dataset(args.dataset, args.data_path,
                 args.save_path, args.dt, args.wt, args.p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('dataset', type=str, default='qm9',
                        help='Target dataset, support [qm9, fullerene, ptgp, smfe]')
    
    parser.add_argument('save_path', type=str, default='processed_data',
                        help='Whether to save processed data')

    parser.add_argument('--data_path', type=str, default='experiments/fullerene',
                        help='Whether to load xyz data')
    
    parser.add_argument('--dt', type=float, default=4.0,
                        help='Cutoff distance')

    parser.add_argument('--wt', type=float, default=0.2,
                        help='Cutoff weight')

    parser.add_argument('--p', type=int, default=8,
                        help='Using multiprocess (8 pool) for faster computing')

    args = parser.parse_args()
    main(args)
