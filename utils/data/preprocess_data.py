import logging
import os
import numpy as np
from qm9 import process_qm9
import argparse
from voronoi_neighbor import parallel_compute_neighbor

def init_dataset(dataset='qm9', save_path=''):

    if dataset == 'qm9':
        process_qm9(dataset, save_path)

    
    parallel_compute_neighbor(dataset=os.path.join(save_path,dataset,dataset+'_data_energy.npy'),
                             savepath=os.path.join(save_path, dataset,dataset+'_data_neighbor.npy'))

def main(args):

    save_path = os.path.join(args.save_path, args.dataset)

    init_dataset(args.dataset, args.file_name, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str, default='qm9',
                        help='Target dataset')

    parser.add_argument('--file_name', type=str, default='qm9_data_energy',
                        help='Whether to use ring as extra emedding')

    parser.add_argument('--dataset_path', type=str, default='processed_data',
                    help='Whether to use ref optimization energy')
                    
    args = parser.parse_args()
    main(args)
