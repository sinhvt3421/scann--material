from model.model_dam import create_model
import numpy as np
from sklearn.model_selection import train_test_split
from dataset.datagenerator_latxb import DataIterator
from sklearn.preprocessing import MinMaxScaler
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


class SGDR(tf.keras.callbacks.Callback):
    """This callback implements the learning rate schedule for
    Stochastic Gradient Descent with warm Restarts (SGDR),
    as proposed by Loshchilov & Hutter (https://arxiv.org/abs/1608.03983).

    The learning rate at each epoch is computed as:
    lr(i) = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * i/num_epochs))

    Here, num_epochs is the number of epochs in the current cycle, which starts
    with base_epochs initially and is multiplied by mul_epochs after each cycle.

    # Example
        ```python
            sgdr = CyclicLR(min_lr=0.0, max_lr=0.05,
                                base_epochs=10, mul_epochs=2)
            model.compile(optimizer=keras.optimizers.SGD(decay=1e-4, momentum=0.9),
                          loss=loss)
            model.fit(X_train, Y_train, callbacks=[sgdr])
        ```

    # Arguments
        min_lr: minimum learning rate reached at the end of each cycle.
        max_lr: maximum learning rate used at the beginning of each cycle.
        base_epochs: number of epochs in the first cycle.
        mul_epochs: factor with which the number of epochs is multiplied
                after each cycle.
    """

    def __init__(self, min_lr=0.0, max_lr=0.05, base_epochs=10, mul_epochs=2):
        super(SGDR, self).__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.base_epochs = base_epochs
        self.mul_epochs = mul_epochs
        # self.list_max_lr = [self.max_lr, self.max_lr /
        #                     10, self.max_lr/15, self.max_lr/20]
        self.cycles = 0.
        self.cycle_iterations = 0.
        self.trn_iterations = 0.

        self._reset()

    def _reset(self, new_min_lr=None, new_max_lr=None,
               new_base_epochs=None, new_mul_epochs=None):
        """Resets cycle iterations."""

        if new_min_lr != None:
            self.min_lr = new_min_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_base_epochs != None:
            self.base_epochs = new_base_epochs
        if new_mul_epochs != None:
            self.mul_epochs = new_mul_epochs
        self.cycles = 0.
        self.cycle_iterations = 0.

    def sgdr(self):

        cycle_epochs = self.base_epochs * (self.mul_epochs ** self.cycles)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * (self.cycle_iterations + 1) / cycle_epochs))

    def on_train_begin(self, logs=None):

        if self.cycle_iterations == 0:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.max_lr)
        else:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.sgdr())

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)

        self.trn_iterations += 1
        self.cycle_iterations += 1
        if self.cycle_iterations >= self.base_epochs * (self.mul_epochs ** self.cycles):
            self.cycles += 1
            self.cycle_iterations = 0
            self.max_lr = self.max_lr / 2
            tf.keras.backend.set_value(self.model.optimizer.lr, self.max_lr)
        else:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.sgdr())


class CosineAnnealingScheduler(tf.keras.callbacks.Callback):
    """Cosine annealing scheduler.
    """

    def __init__(self, T_max, eta_max, eta_min=0, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * \
            (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)


def main(args):
    config = yaml.safe_load(open(args.dataset))
    print('Create model')
    model = create_model(config)

    print('Load data neighbors for dataset ', args.dataset)
    all_data = np.load(config['hyper']['data_nei_path'], allow_pickle=True)

    print('Load data target: ', args.target)
    data_full = np.load(
        config['hyper']['data_energy_path'], allow_pickle=True)

    data_energy = []
    for d in data_full:
        data_energy.append([d['Atomic'], d['Properties']
                            [args.target], d['Ring'], d['Aromatic']])

    data_energy = np.array(data_energy, dtype='object')

    data_ofm_raw = pickle.load(open(
        config['hyper']['data_ofm_path'], 'rb'))
    struct_ofm = np.array(pickle.load(open(
        config['hyper']['struct_ofm_path'], 'rb')))
    # print(data_energy[0])
    idx = ~np.all(struct_ofm == 0, axis=0)
    data_ofm = []
    for ofm in data_ofm_raw:
        data_ofm.append(ofm[:, idx])
    data_ofm = np.array(data_ofm)

    N_train = 100000
    N_test = int(config['hyper']['data_size'] * 0.1)
    N_val = config['hyper']['data_size'] - N_train - N_test

    np.random.seed(0)
    data_perm = np.random.permutation(config['hyper']['data_size'])

    train, valid, test, extra = np.split(
        data_perm, [N_train, N_train+N_val, N_train+N_val+N_test])

    assert(len(extra) == 0), 'Split was inexact {} {} {} {}'.format(
        len(train), len(valid), len(test), len(extra))

    # his = np.load(config['hyper']['save_path'] +
    #               '/hist_data.npy', allow_pickle=True)
    # test = his[1]
    print(len(train), len(valid), len(test))
    histo = []
    train = DataIterator(type='train', batch_size=config['hyper']['batch_size'],
                         indices=train, data_neigh=all_data, data_energy=data_energy, data_ofm=data_ofm, use_ofm=False, converter=True)
    # # mean, std = train.get_scaler()
    mean = 0
    std = 1
    print(mean, std)
    valid = DataIterator(type='valid', batch_size=config['hyper']['batch_size'],
                         indices=valid, data_neigh=all_data, data_energy=data_energy, data_ofm=data_ofm, use_ofm=False, mean=mean, std=std, converter=True)

    testIter = DataIterator(type='test', batch_size=config['hyper']['batch_size'],
                            indices=test, data_neigh=all_data, data_energy=data_energy, data_ofm=data_ofm, use_ofm=False, mean=mean, std=std, converter=True)

    callbacks = []
    if not os.path.exists(config['hyper']['save_path']):
        os.makedirs(os.path.join(config['hyper']['save_path'], 'models/'))

    shutil.copy('model/model_dam.py',
                config['hyper']['save_path'] + '/model_dam.py')

    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(config['hyper']['save_path'],
                                                                              'models/', "model-{epoch}.h5"),
                                                        monitor='val_mae', save_weights_only=True, verbose=1,
                                                        save_best_only=True))

    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae',
    #                                                  factor=config['hyper']['decay'],
    #                                                  patience=5,  min_lr=1e-5, verbose=1)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(
        config['hyper']['save_path']), profile_batch=0, embeddings_freq=100, histogram_freq=50)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_mae', patience=100)

    # lr = CosineAnnealingScheduler(T_max=50,eta_max=config['hyper']['lr'],eta_min=1e-4)
    lr = SGDR(min_lr=config['hyper']['min_lr'], max_lr=config['hyper']
              ['lr'], base_epochs=50, mul_epochs=2)
    # callbacks.append(reduce_lr)
    callbacks.append(lr)
    callbacks.append(early_stop)
    callbacks.append(tensorboard)

    yaml.safe_dump(config, open(config['hyper']['save_path']+'/config.yaml', 'w'),
                   default_flow_style=False)

    hist = model.fit(train.iterator(), epochs=500,
                     steps_per_epoch=train.num_batch,
                     validation_data=valid.iterator(),
                     callbacks=callbacks,
                     validation_steps=valid.num_batch,
                     verbose=2)
    # model.load_weights(os.path.join(
    #     config['hyper']['save_path'], 'models/', "model-288.h5"))
    y_predict = []
    y = []
    idx = 0
    for i in range(testIter.num_batch):
        inputs, target = testIter._get_batches_of_transformed_samples(
            test[idx:idx+testIter.batch_size])
        output = model.predict(inputs)
        y.extend(list(target['dam_net'] * std + mean))
        y_predict.extend(list(np.squeeze(output) * std + mean))
        idx += testIter.batch_size
    print(y[:10], y_predict[:10])
    print('prediction r2 score:', r2_score(y, y_predict),
          ' and MAE:', mean_absolute_error(y, y_predict))
    save_data = [y_predict, test, hist.history]
    np.save(config['hyper']['save_path'] + '/hist_data', save_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('target', type=str,
                        help='Target energy for training')
    parser.add_argument('dataset',type=str,help='Path to dataset configs')
    args = parser.parse_args()
    main(args)
