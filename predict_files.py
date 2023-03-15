from model.SCANNet import create_model
from utils.datagenerator import DataIterator
from utils.general import process_xyz, prepare_input
import tensorflow as tf

import numpy as np
import os
import yaml
import argparse
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(args):
    config = yaml.safe_load(
        open(os.path.join(args.trained_model, 'config.yaml')))

    print('Reading input from file: ', args.file_name)
    struct = process_xyz(args.file_name)

    inputs = prepare_input(
        struct, use_ring=config['model']['use_ring'], use_hyp=config['model']['use_hyp'])

    print('Loading pretrained model:')
    model = create_model(config, mode='infer')
    model.load_weights(os.path.join(args.trained_model, 'models', 'model.h5'))

    energy, attn_global, local_rep, attn_local = model.predict(inputs)

    print('Save prediction and GA score')
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    struct_name = os.path.splitext(os.path.basename(args.file_name))[0]
    save_xyz = '{}_ga_scores_{}.xyz'.format(struct_name, config['hyper']['target'])

    with open(os.path.join(args.save_path,  save_xyz), 'w') as f:
        f.write(str(len(struct['Atoms'])) + '\n')
        f.write('XXX \n')
        for i in range(len(struct['Atoms'])):
            f.write(struct['Atoms'][i] + ' ' + str(struct['Coords'][i][0]) + ' '
                    + str(struct['Coords'][i][1]) + ' ' + str(struct['Coords'][i][2]) + ' ' + str(attn_global[0][i][0]) + '\n')

    pickle.dump([inputs, energy, attn_global, local_rep, attn_local], open(
        os.path.join(args.save_path, struct_name + '_ga_scores.pickle'), 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('trained_model', type=str,
                        help='Target trained model path for loading')

    parser.add_argument('save_path', type=str,
                        help='Save path for prediction')

    parser.add_argument('file_name', type=str,
                        help='Path to structure data xyz files')

    args = parser.parse_args()
    main(args)
