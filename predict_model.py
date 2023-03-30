import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import pickle
import yaml
from utils.general import load_dataset
from utils.datagenerator import DataIterator
from scannet.models import SCANNet
import tensorflow as tf



def main(args):
    config = yaml.safe_load(
        open(os.path.join(args.trained_model, 'config.yaml')))

    print('Load pretrained weight for target ', config['hyper']['target'])
    model = SCANNet.load_model_infer(os.path.join(
        args.trained_model, 'models', 'model.h5'))

    print('Load data for trained model: ', config['hyper']['data_energy_path'])
    data_energy, data_neighbor = load_dataset(use_ref=config['hyper']['use_ref'],
                                              use_ring=config['model']['use_ring'],
                                              dataset=config['hyper']['data_energy_path'],
                                              dataset_neighbor=config['hyper']['data_nei_path'],
                                              target_prop=config['hyper']['target'])

    datasetIter = DataIterator(batch_size=config['hyper']['batch_size'],
                               data_neighbor=data_neighbor,
                               data_energy=data_energy, converter=True,
                               use_ring=config['model']['use_ring'])

    ga_scores = []
    struct_energy = []

    idx = 0
    for i in range(len(datasetIter)):
        inputs, _ = datasetIter.__getitem__(i)
        energy, attn_global = model.predict(inputs)

        ga_scores.extend(attn_global)
        struct_energy.extend(energy)

        idx += datasetIter.batch_size
        print(idx)

    print('Save prediction and GA score')
    pickle.dump(ga_scores, open(os.path.join(args.trained_model,
                'ga_scores_{}.pickle'.format(config['hyper']['target'])), 'wb'))

    pickle.dump(struct_energy, open(os.path.join(args.trained_model,
                'energy_pre_{}.pickle'.format(config['hyper']['target'])), 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('trained_model', type=str,
                        help='Target trained model path for loading')

    args = parser.parse_args()
    main(args)
