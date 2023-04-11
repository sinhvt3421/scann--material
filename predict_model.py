import tensorflow as tf
from sklearn.metrics import r2_score, mean_absolute_error
from scannet.models import SCANNet
import yaml
import pickle
import argparse
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(args):

    config = yaml.safe_load(
        open(os.path.join(args.trained_model, 'config.yaml')))

    print('Load pretrained weight for target ', config['hyper']['target'])
    scannet = SCANNet(config, os.path.join(
        args.trained_model, 'models', 'model_{}.h5'.format(config['hyper']['target'])), mode='infer')

    print('Load data for trained model: ', config['hyper']['data_energy_path'])
    scannet.prepare_dataset(split=False)

    ga_scores = []
    struct_energy = []
    y = []
    idx = 0
    data = scannet.dataIter

    for i in range(len(data)):
        inputs, target = data.__getitem__(i)
        energy, attn_global = scannet.predict_data(inputs)

        ga_scores.extend(attn_global)

        struct_energy.extend(list(np.squeeze(energy)))
        y.extend(list(target))

        idx += data.batch_size
        if i % 10 == 0:
            print(idx)

    print(r2_score(struct_energy, y), mean_absolute_error(struct_energy, y))

    print('Save prediction and GA score')
    pickle.dump(ga_scores, open(os.path.join(args.trained_model,
                                             'ga_scores_{}.pickle'.format(config['hyper']['target'])), 'wb'))

    pickle.dump([y, struct_energy], open(os.path.join(args.trained_model,
                                                      'energy_pre_{}.pickle'.format(config['hyper']['target'])), 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('trained_model', type=str,
                        help='Target trained model path for loading')

    args = parser.parse_args()
    main(args)
