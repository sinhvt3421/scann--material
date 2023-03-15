import tensorflow as tf
from tensorflow.keras.callbacks import *

import shutil
import numpy as np
import random
import os
import yaml
import argparse
import time

from model.SCANNet import create_model
from model.custom_layer import SGDRC

from utils.datagenerator import DataIterator
from utils.general import *

from sklearn.metrics import r2_score, mean_absolute_error


def set_seed(seed=2134):
    # tf.keras.utils.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

def main(args):
    
    set_seed(2134)
    
    start = time.time()
    config = yaml.safe_load(open(args.dataset))

    config['model']['use_ring'] = args.use_ring
    config['model']['use_hyp'] = args.use_hyp
    config['hyper']['use_ref'] = args.use_ref
    config['hyper']['target'] = args.target

    print('Create model use Ring Information: ', args.use_ring, ' use Acceptor/Donor Information: ', args.use_hyp)

    model = create_model(config, mode='train')
    if args.pretrained:
        print('load pretrained weight')
        model.load_weights(config['hyper']['pretrained'])

    print('Load data for dataset: ', args.dataset)
    data_energy, data_neighbor = load_dataset(use_ref=args.use_ref, use_ring=args.use_ring,use_hyp=args.use_hyp,
                                              dataset=config['hyper']['data_energy_path'],
                                              dataset_neighbor=config['hyper']['data_nei_path'],
                                              target_prop=args.target)

    config['hyper']['data_size'] = len(data_energy)

    train, valid, test, extra = split_data(len_data=len(data_energy), test_percent=config['hyper']['test_percent'],
                                           train_size=config['hyper']['train_size'],
                                           test_size=config['hyper']['test_size'])

    assert (len(extra) == 0), 'Split was inexact {} {} {} {}'.format(
        len(train), len(valid), len(test), len(extra))

    print("Number of train data : ", len(train), " , Number of valid data: ", len(valid),
          " , Number of test data: ", len(test))

    trainIter = DataIterator(batch_size=config['hyper']['batch_size'],
                             data_neighbor=data_neighbor[train],
                             data_energy=data_energy[train], converter=True,
                             use_ring=args.use_ring, use_hyp=args.use_hyp, shuffle=True)

    validIter = DataIterator(batch_size=config['hyper']['batch_size'],
                             data_neighbor=data_neighbor[valid],
                             data_energy=data_energy[valid], converter=True,
                             use_ring=args.use_ring,use_hyp=args.use_hyp)

    testIter = DataIterator(batch_size=config['hyper']['batch_size'],
                            data_neighbor=data_neighbor[test],
                            data_energy=data_energy[test], converter=True,
                            use_ring=args.use_ring,use_hyp=args.use_hyp)

    if not os.path.exists(config['hyper']['save_path'] + '_' + args.target):
        os.makedirs(os.path.join(
            config['hyper']['save_path'] + '_' + args.target, 'models/'))

    callbacks = []
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(config['hyper']['save_path'] + '_' + args.target,
                                                                              'models/', "model.h5"),
                                                        monitor='val_mae',
                                                        save_weights_only=True, verbose=2,
                                                        save_best_only=True))

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_mae', patience=200)

    lr = SGDRC(lr_min=config['hyper']['min_lr'],
               lr_max=config['hyper']['lr'], t0=50, tmult=2,
               lr_max_compression=1.2, trigger_val_mae=80)

    sgdr = LearningRateScheduler(lr.lr_scheduler)
    callbacks.append(sgdr)

    callbacks.append(lr)
    callbacks.append(early_stop)

    yaml.safe_dump(config, open(config['hyper']['save_path'] + '_' + args.target + '/config.yaml', 'w'),
                   default_flow_style=False)

    shutil.copy('model/SCANNet.py',
                config['hyper']['save_path'] + '_' + args.target + '/SCANNet.py')
    
    shutil.copy('model/custom_layer.py',
                config['hyper']['save_path'] + '_' + args.target + '/custom_layer.py')
    
    shutil.copy('train.py',
                config['hyper']['save_path'] + '_' + args.target + '/train.py')

    hist = model.fit(trainIter, epochs=1000,
                     validation_data=validIter,
                     callbacks=callbacks,
                     verbose=2, shuffle=False,
                     use_multiprocessing=True,
                     workers=4)

    print('Training time: ', time.time()-start)

    # Predict for testdata
    print('Load best validation weight for predicting testset')
    model.load_weights(os.path.join(config['hyper']['save_path'] + '_' + args.target,
                                    'models/', 'model.h5'))

    y_predict = []
    y = []
    for i in range(len(testIter)):
        inputs, target = testIter.__getitem__(i)
        output = model.predict(inputs)

        y.extend(list(target))
        y_predict.extend(list(np.squeeze(output)))

    print('Result for testset: R2 score: ', r2_score(y, y_predict),
          ' and MAE: ', mean_absolute_error(y, y_predict))

    save_data = [y_predict, y, test, hist.history]

    np.save(config['hyper']['save_path'] + '_' +
            args.target + '/hist_data.npy', save_data)

    with open(config['hyper']['save_path'] + '_' +
              args.target + '/report.txt', 'w') as f:
        f.write('Training MAE: ' + str(min(hist.history['mae'])) + '\n')
        f.write('Val MAE: ' + str(min(hist.history['val_mae'])) + '\n')
        f.write('Test MAE: ' + str(mean_absolute_error(y, y_predict)) +
                ', Test R2: ' + str(r2_score(y, y_predict)))

    print('Saved model record for dataset')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('target', type=str,
                        help='Target energy for training')

    parser.add_argument('dataset', type=str, help='Path to dataset configs')

    parser.add_argument('--use_ring', type=bool, default=False,
                        help='Whether to use ring as extra emedding')
    
    parser.add_argument('--use_hyp', type=bool, default=False,
                        help='Whether to use acceptor or donor informatio as extra emedding')

    parser.add_argument('--use_ref', type=bool, default=False,
                        help='Whether to use ref optimization energy')

    parser.add_argument('--pretrained', type=bool, default=False,
                        help='Whether to use pretrained model')

    args = parser.parse_args()
    main(args)
