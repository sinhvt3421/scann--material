import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Embedding, Dense, Dropout, Input, Lambda

from scannet.layers import *
from scannet.losses import *
from scannet.layers import _CUSTOM_OBJECTS

from utils.general import load_dataset, split_data
from utils.datagenerator import DataIterator
from tensorflow.keras.callbacks import *

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import os
import time
import yaml
import gc


class SCANNet:
    """
        Implements main SCANNet 
    """

    def __init__(self, config):
        """
        Args:
            config:
                    use_ring: Whether using extra ring information for molecule data
                    use_attn_norm: Whether using normalization layer for local attention as in Transformer

                    n_attention: Number of local attention layer
                    n_embedding: Embedding dims for atomic number of atoms

                    dense_embed: Dims for computing linear layer
                    dim: dim for computing atttention layer
                    num_head: total head for attention as in Transformer, set to 1 will not using head attention
                    scale: attention scale factor, default to 0.5. 
        """
        self.config = config

    def __getattr__(self, p):
        return getattr(self.model, p)

    def init_model(self, pretrained=''):
        if pretrained:
            print('load pretrained model from ', self.config['hyper']['pretrained'])
            self.model = create_model_pretrained(self.config)
        else:
            self.model = create_model(self.config)

        self.model.compile(loss=root_mean_squared_error,
                           optimizer=tf.keras.optimizers.Adam(self.config['hyper']['lr'],
                                                              gradient_transformers=[AutoClipper(10)]),
                           metrics=['mae', r2_square])

    @classmethod
    def load_model_infer(cls, path):
        model = load_model(path, custom_objects=_CUSTOM_OBJECTS)

        attention_output = model.get_layer('global_attention').output[0]
        model_infer = tf.keras.Model(inputs=model.input, outputs=[
                                     model.output, attention_output])
        return model_infer

    def prepare_dataset(self):

        data_energy, data_neighbor = load_dataset(use_ref=self.config['hyper']['use_ref'], 
                                                  use_ring=self.config['model']['use_ring'],
                                                  dataset=self.config['hyper']['data_energy_path'],
                                                  dataset_neighbor=self.config['hyper']['data_nei_path'],
                                                  target_prop=self.config['hyper']['target'])

        self.config['hyper']['data_size'] = len(data_energy)

        train, valid, test, extra = split_data(len_data=len(data_energy),
                                               test_percent=self.config['hyper']['test_percent'],
                                               train_size=self.config['hyper']['train_size'],
                                               test_size=self.config['hyper']['test_size'])

        assert (len(extra) == 0), 'Split was inexact {} {} {} {}'.format(
            len(train), len(valid), len(test), len(extra))

        print("Number of train data : ", len(train), " , Number of valid data: ", len(valid),
              " , Number of test data: ", len(test))

        self.trainIter, self.validIter, self.testIter = [DataIterator(batch_size=self.config['hyper']['batch_size'],
                                                                      data_neighbor=data_neighbor[indices],
                                                                      data_energy=data_energy[indices],
                                                                      use_ring=self.config['model']['use_ring'],
                                                                      shuffle=(len(indices) == len(train)))
                                                         for indices in (train, valid, test)]

    def create_callbacks(self):

        callbacks = []
        callbacks.append(ModelCheckpoint(filepath='{}_{}/models/model.h5'.format(self.config['hyper']['save_path'], self.config['hyper']['target']),
                                         monitor='val_mae',
                                         save_weights_only=False, verbose=2,
                                         save_best_only=True))

        callbacks.append(EarlyStopping(monitor='val_mae', patience=200))

        lr = SGDRC(lr_min=self.config['hyper']['min_lr'],
                   lr_max=self.config['hyper']['lr'], t0=50, tmult=2,
                   lr_max_compression=1.2, trigger_val_mae=80)
        sgdr = LearningRateScheduler(lr.lr_scheduler)

        callbacks.append(lr)
        callbacks.append(sgdr)

        return callbacks

    def train(self, epochs=1000):

        if not os.path.exists('{}_{}/models/'.format(self.config['hyper']['save_path'], self.config['hyper']['target'])):
            os.makedirs(
                '{}_{}/models/'.format(self.config['hyper']['save_path'], self.config['hyper']['target']))
            
        callbacks = self.create_callbacks()

        yaml.safe_dump(self.config, open('{}_{}/config.yaml'.format(self.config['hyper']['save_path'], self.config['hyper']['target']), 'w'),
                       default_flow_style=False)

        self.hist = self.model.fit(self.trainIter, epochs=epochs,
                                   validation_data=self.validIter,
                                   callbacks=callbacks,
                                   verbose=2, shuffle=False,
                                   use_multiprocessing=True,
                                   workers=4)

        tf.keras.backend.clear_session()
        del self.model
        gc.collect()

    def evaluate(self):
        # Predict for testdata
        print('Load best validation weight for predicting testset')
        self.model = load_model('{}_{}/models/model.h5'.format(self.config['hyper']['save_path'], self.config['hyper']['target']), custom_objects=_CUSTOM_OBJECTS)

        y_predict = []
        y = []
        for i in range(len(self.testIter)):
            inputs, target = self.testIter.__getitem__(i)
            output = self.model.predict(inputs)

            y.extend(list(target))
            y_predict.extend(list(np.squeeze(output)))

        print('Result for testset: R2 score: ', r2_score(y, y_predict),
              ' and MAE: ', mean_absolute_error(y, y_predict))

        save_data = [y_predict, y, self.hist.history]

        np.save(
            '{}_{}/hist_data.npy'.format(self.config['hyper']['save_path'], self.config['hyper']['target']), save_data)

        with open('{}_{}/report.txt'.format(self.config['hyper']['save_path'], self.config['hyper']['target']), 'w') as f:
            f.write('Training MAE: ' +
                    str(min(self.hist.history['mae'])) + '\n')
            f.write('Val MAE: ' +
                    str(min(self.hist.history['val_mae'])) + '\n')
            f.write('Test MAE: ' + str(mean_absolute_error(y, y_predict)) +
                    ', Test R2: ' + str(r2_score(y, y_predict)))

        print('Saved model record for dataset')


def create_model_pretrained(config):

    model = load_model(config['hyper']['pretrained'], custom_objects=_CUSTOM_OBJECTS)
    model.summary()

    return model


def create_model(config):

    atomic = Input(name='atomic', shape=(None,), dtype='int32')
    atom_mask = Input(shape=[None, 1], name='atom_mask')

    neighbor = Input(name='neighbors', shape=(None, None), dtype='int32')
    neighbor_mask = Input(name='neighbor_mask',
                          shape=(None, None), dtype='float32')

    neighbor_weight = Input(name='neighbor_weight',
                            shape=(None, None, 1), dtype='float32')
    neighbor_distance = Input(name='neighbor_distance', shape=(
        None, None, 20), dtype='float32')

    inputs = []
    cfm = config['model']
    if cfm['use_ring']:
        shape = 2
        ring_info = Input(name='ring_aromatic', shape=(
            None, shape), dtype='float32')

        inputs = [atomic,  atom_mask, neighbor, neighbor_mask,
                  neighbor_weight, neighbor_distance, ring_info]
    else:
        inputs = [atomic,  atom_mask, neighbor,
                  neighbor_mask, neighbor_weight, neighbor_distance]

    # Embedding atom and extra information as ring, aromatic
    centers = Embedding(cfm['n_atoms'],
                        cfm['embedding_dim'],
                        name='embed_atom',
                        dtype='float32')(atomic)
    if cfm['use_ring']:
        ring_embed = Dense(
            10, name='extra_embed', dtype='float32')(ring_info)

        # Shape embed_atom [B, M, n_embedding + 10]
        centers = tf.concat([centers, ring_embed], -1)

    centers = Dense(cfm['local_dim'], activation='swish', name='dense_embed',
                    dtype='float32')(centers)
    centers = Dropout(0.1)(centers)

    neighbor_indices = Lambda(gather_shape, name='get_neighbor')(neighbor)

    def local_attention_block(c, n_in, n_d, n_w, n_m):

        # Local attention for local structure representation
        attn_local, context = LocalAttention(v_proj=False, kq_proj=True,
                                             dim=cfm['local_dim'],
                                             num_head=cfm['num_head'],
                                             activation='swish')(c, n_in, n_d, n_w, n_m)
        if cfm['use_attn_norm']:
            # 2 Forward Norm layers
            centers = ResidualNorm(cfm['local_dim'])(context)
        else:
            centers = context

        return centers, attn_local

    # list_centers_layers = []
    # Local Attention recursive layers
    for i in range(cfm['n_attention']):
        centers, attn_local = local_attention_block(centers, neighbor_indices,
                                                    neighbor_distance, neighbor_weight,
                                                    neighbor_mask)
        # list_centers_layers.append(centers)

     # Dense layer after Local Attention -> representation for each local structure [B, M, d]
    centers = Dense(cfm['global_dim'], activation='swish',
                    name='after_Lc',
                    kernel_regularizer=regularizers.l2(1e-4))(centers)

    # Using weighted attention score for combining structures representation
    attn_global, struc_rep = GlobalAttention(v_proj=False, kq_proj=True,
                                             dim=cfm['global_dim'], scale=cfm['scale'],
                                             norm=cfm['use_ga_norm'])(centers, atom_mask)

    # Shape struct representation [B, d]
    struc_rep = Dense(cfm['dense_out'], activation='swish', name='bf_property',
                      kernel_regularizer=regularizers.l2(1e-4))(struc_rep)

    # Shape predict_property [B, 1]
    predict_property = Dense(1, name='predict_property')(struc_rep)

    model = tf.keras.Model(inputs=inputs, outputs=[predict_property])

    model.summary()

    return model
