import logging
import os
import numpy as np
from utils.dataset import *
import argparse
from utils.voronoi_neighbor import parallel_compute_neighbor


def init_dataset(dataset='qm9', data_path='', save_path='', d_t=4.0, w_t=0.2, p=8):

    if dataset == 'qm9':
        print('Init dataset QM9:')
        if not os.path.exists(os.path.join(save_path, dataset)):
            process_qm9(save_path)

    if dataset == 'fullerence':
        print('Init dataset Fullerence:')
        if not os.path.exists(os.path.join(save_path, dataset)):
            process_fullerence(data_path, save_path)

    if dataset == 'smfe':
        print('Init dataset SmFe12:')
        if not os.path.exists(os.path.join(save_path, dataset)):
            process_smfe(data_path, save_path)

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
                        help='Target dataset')
    
    parser.add_argument('save_path', type=str, default='processed_data',
                        help='Whether to save processed data')

    parser.add_argument('--data_path', type=str, default='experiments/fullerence',
                        help='Whether to load xyz data')
    
    parser.add_argument('--dt', type=float, default=4.0,
                        help='Cutoff distance')

    parser.add_argument('--wt', type=float, default=0.2,
                        help='Cutoff weight')

    parser.add_argument('--p', type=int, default=8,
                        help='Cutoff weight')

    args = parser.parse_args()
    main(args)
