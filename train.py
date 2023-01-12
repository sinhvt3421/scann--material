from model.model import create_model
import tensorflow as tf
from tensorflow.keras.callbacks import *
from model.custom_layer import SGDR

import numpy as np
from utils.datagenerator import DataIterator
from utils.general import *

from sklearn.metrics import r2_score, mean_absolute_error
import os
from ase.db import connect
import yaml
import shutil

from ase.units import Hartree, eV
import argparse
import time


def main(args):
    
    start = time.time()
    config = yaml.safe_load(open(args.dataset))

    config['model']['use_ring'] = args.use_ring

    print('Create model use ring information: ', args.use_ring)

    model = create_model(config, mode='train')
    if args.pretrained:
        print('load pretrained weight')
        model.load_weights(config['hyper']['pretrained'])

    print('Load data for dataset: ', args.dataset)
    data_energy, data_neighbor = load_dataset(use_ref=args.use_ref, use_ring=args.use_ring,
                                              dataset=config['hyper']['data_energy_path'],
                                              dataset_neighbor=config['hyper']['data_nei_path'],
                                              target_prop=args.target)

    config['hyper']['data_size'] = len(data_energy)

    train, valid, test, extra = split_data(len_data=len(data_energy),
                                           test_size=config['hyper']['test_size'])

    assert(len(extra) == 0), 'Split was inexact {} {} {} {}'.format(
        len(train), len(valid), len(test), len(extra))

    print("Number of train data : ", len(train), " , Number of valid data: ", len(valid),
          " , Number of test data: ", len(test))

    indices = {'train': train, 'valid': valid, 'test': test}

    datasetIter = DataIterator(batch_size=config['hyper']['batch_size'],
                               indices=indices, data_neighbor=data_neighbor,
                               data_energy=data_energy, converter=True,
                               use_ring=args.use_ring)

    callbacks = []
    if not os.path.exists(config['hyper']['save_path'] + '_' + args.target):
        os.makedirs(os.path.join(
            config['hyper']['save_path'] + '_' + args.target, 'models/'))

    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(config['hyper']['save_path'] + '_' + args.target,
                                                                              'models/', "model.h5"),
                                                        monitor='val_mae',
                                                        save_weights_only=True, verbose=1,
                                                        save_best_only=True))

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_mae', patience=100)

    lr = SGDR(min_lr=config['hyper']['min_lr'], max_lr=config['hyper']
              ['lr'], base_epochs=50, mul_epochs=2)

    callbacks.append(lr)
    callbacks.append(early_stop)

    yaml.safe_dump(config, open(config['hyper']['save_path'] + '_' + args.target + '/config.yaml', 'w'),
                   default_flow_style=False)

    shutil.copy('model/model.py',
                config['hyper']['save_path'] + '_' + args.target + '/model.py')
    shutil.copy('train.py',
                config['hyper']['save_path'] + '_' + args.target + '/train.py')

    hist = model.fit(datasetIter.iterator('train'), epochs=1000,
                     steps_per_epoch=datasetIter.num_batch['train'],
                     validation_data=datasetIter.iterator('valid'),
                     validation_steps=datasetIter.num_batch['valid'],
                     callbacks=callbacks,
                     verbose=2)

    print('Training time: ', time.time()-start)

    # Predict for testdata
    print('Load best validation weight for predicting testset')
    model.load_weights(os.path.join(config['hyper']['save_path'] + '_' + args.target,
                                    'models/', 'model.h5'))

    y_predict = []
    y = []
    idx = 0
    for i in range(datasetIter.num_batch['test']):
        inputs, target = datasetIter.get_batch(
            test[idx:idx+datasetIter.batch_size])
        output = model.predict(inputs)

        y.extend(list(target['gam_net']))
        y_predict.extend(list(np.squeeze(output)))

        idx += datasetIter.batch_size

    print('Result for testset: R2 score: ', r2_score(y, y_predict),
          ' and MAE: ', mean_absolute_error(y, y_predict))

    save_data = [y_predict, y, test, hist.history]

    np.save(config['hyper']['save_path'] + '_' +
            args.target + '/hist_data.npy', save_data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('target', type=str,
                        help='Target energy for training')

    parser.add_argument('dataset', type=str, help='Path to dataset configs')

    parser.add_argument('--use_ring', type=bool, default=False,
                        help='Whether to use ring as extra emedding')

    parser.add_argument('--use_ref', type=bool, default=False,
                        help='Whether to use ref optimization energy')

    parser.add_argument('--pretrained', type=bool, default=False,
                        help='Whether to use pretrained model')

    args = parser.parse_args()
    main(args)
