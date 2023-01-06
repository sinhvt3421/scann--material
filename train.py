from model.model import create_model
import numpy as np
from dataset.datagenerator import DataIterator
from sklearn.metrics import r2_score, mean_absolute_error
import tensorflow as tf
import os
from ase.db import connect
import yaml
import shutil
import pickle
import math
from ase.units import Hartree, eV
import argparse
import time
from model.custom_layer import SGDR
from utils.general import *


def main(args):
    a = time.time()
    config = yaml.safe_load(open(args.dataset))

    config['model']['use_ofm'] = args.use_ofm
    config['model']['use_ring'] = args.use_ring
    config['model']['use_bonds'] = args.use_bonds

    print('Create model use ring ', args.use_ring, ' use bonds ',
          args.use_bonds, ' use ofm ', args.use_ofm)

    model = create_model(config,mode='train')
    if args.pretrained:
        print('load pretrained weight')
        model.load_weights(config['hyper']['pretrained'])

    print('Load data for dataset ', args.dataset)
    data_energy, data_neighbor = load_dataset(use_ref=args.use_ref,use_ring=args.use_ring,
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

    trainIter = DataIterator(type='train', batch_size=config['hyper']['batch_size'],
                         indices=train, data_neighbor=data_neighbor,
                         data_energy=data_energy,converter=True, 
                         use_ring=args.use_ring, use_bonds=args.use_bonds)
  
    validIter = DataIterator(type='valid', batch_size=config['hyper']['batch_size'],
                         indices=valid, data_neighbor=data_neighbor,
                         data_energy=data_energy,converter=True, 
                         use_ring=args.use_ring, use_bonds=args.use_bonds)

    testIter = DataIterator(type='test', batch_size=config['hyper']['batch_size'],
                            indices=test, data_neighbor=data_neighbor,
                            data_energy=data_energy,converter=True, 
                            use_ring=args.use_ring, use_bonds=args.use_bonds)

    callbacks = []
    if not os.path.exists(config['hyper']['save_path'] + '_' + args.target):
        os.makedirs(os.path.join(
            config['hyper']['save_path'] + '_' + args.target, 'models/'))

    shutil.copy('model/model.py',
                config['hyper']['save_path'] + '_' + args.target + '/model.py')
    shutil.copy('train.py',
                    config['hyper']['save_path'] + '_' + args.target + '/train.py')

    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(config['hyper']['save_path'] + '_' + args.target,
                                                                              'models/', "model-{epoch}.h5"),
                                                        monitor='val_mae', save_weights_only=True, verbose=1,
                                                        save_best_only=True))

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(
        config['hyper']['save_path'] + '_' + args.target), profile_batch=0, embeddings_freq=0, histogram_freq=0)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_mae', patience=100)

    lr = SGDR(min_lr=config['hyper']['min_lr'], max_lr=config['hyper']
              ['lr'], base_epochs=200, mul_epochs=2)
    # callbacks.append(reduce_lr)
    callbacks.append(lr)
    callbacks.append(early_stop)
    callbacks.append(tensorboard)

    yaml.safe_dump(config, open(config['hyper']['save_path'] + '_' + args.target + '/config.yaml', 'w'),
                   default_flow_style=False)

    hist = model.fit(trainIter.iterator(), epochs=1000,
                     steps_per_epoch=trainIter.num_batch,
                     validation_data=validIter.iterator(),
                     callbacks=callbacks,
                     validation_steps=validIter.num_batch,
                     verbose=2, workers=1)


    y_predict = []
    y = []
    idx = 0
    for i in range(testIter.num_batch):
        inputs, target = testIter._get_batches_of_transformed_samples(
            test[idx:idx+testIter.batch_size])
        output = model.predict(inputs)
        y.extend(list(target['gam_net'] ))
        y_predict.extend(list(np.squeeze(output)))
        idx += testIter.batch_size

    print('prediction r2 score:', r2_score(y, y_predict),
          ' and MAE:', mean_absolute_error(y, y_predict))

    save_data = [y_predict, test, hist.history]
    np.save(config['hyper']['save_path'] + '_' +
            args.target + '/hist_data', save_data)

    print('Training time: ', time.time()-a)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('target', type=str,
                        help='Target energy for training')
    parser.add_argument('dataset', type=str, help='Path to dataset configs')
    parser.add_argument('--use_ofm', type=bool, default=False,
                        help='Whether to use ofm as extra embedding')
    parser.add_argument('--use_ring', type=bool, default=False,
                        help='Whether to use ring as extra emedding')
    parser.add_argument('--use_bonds', type=bool, default=False,
                        help='Whether to use bond type as extra emedding bonds')
    parser.add_argument('--use_ref', type=bool, default=False,
                    help='Whether to use ref optimization energy')
    parser.add_argument('--pretrained', type=bool, default=False,
                    help='Whether to use pretrained model')
    args = parser.parse_args()
    main(args)
