from model.SCANNet import create_model
import numpy as np
from utils.datagenerator import DataIterator
from utils.general import *
import tensorflow as tf
import os
import yaml
import pickle
import argparse


def main(args):
    config = yaml.safe_load(open(os.path.join(args.trained_model,'config.yaml')))

    model = create_model(config, mode='infer')
    model.load_weights(os.path.join(args.trained_model,'models','model.h5'))

    print('Load data for trained model: ', config['hyper']['data_energy_path'])
    data_energy, data_neighbor = load_dataset(use_ref=config['hyper']['use_ref'], use_ring=config['model']['use_ring'],
                                              dataset=config['hyper']['data_energy_path'],
                                              dataset_neighbor=config['hyper']['data_nei_path'],
                                              target_prop=config['hyper']['target'])

    infer = range(0, 100000)
    indices={'infer': infer}
    datasetIter = DataIterator(batch_size=config['hyper']['batch_size'],
                                indices=indices, data_neighbor=data_neighbor,
                                data_energy=data_energy, converter=True,
                                use_ring=config['model']['use_ring'])

    local_reps = []
    ga_scores  = []
    struct_energy = []

    idx = 0
    for i in range(datasetIter.num_batch['infer']):
        inputs, target = datasetIter.get_batch(infer[idx:idx+datasetIter.batch_size])
        energy, attn_global, local_rep = model.predict(inputs)
    
        ga_scores.extend(attn_global)
        struct_energy.extend(energy)
        local_reps.append(local_rep)
        idx += datasetIter.batch_size
        print(idx)

    print('Save prediction and GA score')
    pickle.dump(ga_scores, open(os.path.join(args.trained_model,'ga_scores.pickle'), 'wb'))

    pickle.dump(local_reps, open(os.path.join(args.trained_model,'local_reps.pickle'), 'wb'))

    pickle.dump(struct_energy, open(os.path.join(args.trained_model, 'energy_pre.pickle'), 'wb'))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('trained_model', type=str,
                        help='Target trained model path for loading')

    args = parser.parse_args()
    main(args)
