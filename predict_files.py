import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import argparse
import yaml
import numpy as np
from utils.general import process_xyz_pmt, prepare_input_pmt
from scannet.models import SCANNet

def main(args):
    config = yaml.safe_load(
        open(os.path.join(args.trained_model, 'config.yaml')))

    print('Reading input from file: ', args.file_name)
    struct = process_xyz_pmt(args.file_name)

    inputs = prepare_input_pmt(struct)
    
    print('Load pretrained weight for target ', config['hyper']['target'])
    model = SCANNet.load_model_infer(os.path.join(
        args.trained_model, 'model_{}.h5'.format(config['hyper']['target'])))

    energy, attn_global= model.predict(inputs)

    print('Save prediction and GA score')
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    struct_name = os.path.splitext(os.path.basename(args.file_name))[0]
    save_xyz = '{}_ga_scores_{}.xyz'.format(
        struct_name, config['hyper']['target'])

    with open(os.path.join(args.save_path,  save_xyz), 'w') as f:
        f.write(str(len(struct['Atoms'])) + '\n')
        f.write('XXX \n')
        for i in range(len(struct['Atoms'])):
            f.write('{}\t{}\t{}\t{}\t{}\n'.format(
                struct['Atoms'][i], struct['Coords'][i][0],
                struct['Coords'][i][1], struct['Coords'][i][2], attn_global[0][i][0]))

    pickle.dump([inputs, energy, attn_global], open(
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
