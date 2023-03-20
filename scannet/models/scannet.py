import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from scannet.layers import *
from scannet.losses import *


class SCANNet(tf.keras.layers.Layer):
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

        super(SCANNet, self).__init__()

        # Number of attention layer
        self.n_attention = config['n_attention']

        # Input is whether molecule or crystal material
        self.use_ring = config['use_ring']

        # Whether using normalization layer for local attention as in Transformer
        self.attn_norm = config['use_attn_norm']

        # Embeding for atomic number and other extra information as ring, aromatic,...
        self.embed_atom = tf.keras.layers.Embedding(config['n_atoms'],
                                                    config['embedding_dim'],
                                                    name='embed_atom',
                                                    dtype='float32')
        if self.use_ring:
            self.extra_embed = tf.keras.layers.Dense(
                10, name='extra_embed', dtype='float32')

        self.dense_embed = tf.keras.layers.Dense(config['local_dim'],
                                                 activation='swish', name='dense_embed',
                                                 dtype='float32')
        self.dropout = tf.keras.layers.Dropout(0.1)

        # L layers Local Attention
        self.local_attention = [LocalAttention(v_proj=False, kq_proj=True,
                                               dim=config['local_dim'], num_head=config['num_head'],
                                               activation='swish')
                                for i in range(config['n_attention'])]

        if self.attn_norm:
            self.residual_norm = [ResidualNorm(
                config['local_dim']) for i in range(config['n_attention'])]

        # Dense layer before Global Attention
        self.dense_afterLc = tf.keras.layers.Dense(config['global_dim'], activation='swish',
                                                   name='after_Lc',
                                                   kernel_regularizer=regularizers.l2(1e-4))

        # Global Attention layer
        self.global_attention = GlobalAttention(v_proj=False, kq_proj=True,
                                                dim=config['global_dim'], scale=config['scale'],
                                                norm=config['use_ga_norm'])

        # Dense layer on structure representation
        self.dense_bftotal = tf.keras.layers.Dense(
            config['dense_out'], activation='swish', name='bf_property',
            kernel_regularizer=regularizers.l2(1e-4))

        # Output property
        self.predict_property = tf.keras.layers.Dense(
            1, name='predict_property')

    def get_config(self):
        config = super(SCANNet, self).get_config()
        config.update({
            'n_attention': self.n_attention,
            'use_ring': self.use_ring,
            'attn_norm': self.attn_norm,
            'embed_atom': self.embed_atom,
            'extra_embed': self.extra_embed,
            'dense_embed': self.dense_embed,
            'dropout': self.dropout,
            'local_attention': self.local_attention,
            'residual_norm': self.residual_norm,
            'dense_afterLc': self.dense_afterLc,
            'global_attention': self.global_attention,
            'dense_bftotal': self.dense_bftotal,
            'predict_property': self.predict_property,
        })
        return config

    def local_attention_loop(self, centers, neighbor, neighbor_mask, neigbor_weight, neighbor_distance):

        neighbor_shape = tf.shape(neighbor)

        B, M, N = neighbor_shape[0], neighbor_shape[1], neighbor_shape[2]
        D = centers.shape[-1]

        range_B = tf.range(B)[:, None, None, None]
        range_B_t = tf.tile(range_B, [1, M, N, 1])
        neighbor_indices = tf.concat(
            [range_B_t, tf.expand_dims(neighbor, -1)], -1)

        list_context = []
        # Local Attention recursive layers
        for i in range(self.n_attention):

            # Get neighbor vector from indices, size [B, M, N, d] and  Multiply weight Voronoi with neighbor
            neighbors = tf.gather_nd(centers, neighbor_indices)

            # Shape neighbor_weighted [B, M, N, embedding_dim ]
            neighbors = tf.reshape(neighbors, [B, M, N, D])

            # Local attention for local structure representation
            attn_local, context = self.local_attention[i](centers, neighbors,
                                                          neighbor_distance, neigbor_weight,  neighbor_mask)
            if self.attn_norm:
                # 2 Forward Norm layers
                centers = self.residual_norm[i](context)
            else:
                centers = context

            list_context.append(centers)

        return centers, list_context, attn_local

    def call(self, inputs, train=True, global_attn=True):
        """     
                B: Batch size
                M: Number of centers atoms
                N: Number of neighbor atoms of the center
            Args:
                inputs:
                    atoms: Atomic number for each atom in the structure, size [B, M, 1]
                    ring_info: Whether an atom in a Ring or Aromatic, size [B, M, 2]
                    mask_atom: Masking for padding atoms in a batch up to M atoms, size [B, M, 1]
                    neighbor: Indexes of neighbor atoms for each center in the structure, size [B, M, N, 1]
                    mask: Masking for padding neighbor indexes in a batch up to N neighbors, size [B, M, N, 1]
                    neighbor_weight: Voronoi weight of each neighbor atoms for each center, size [B, M, N, 1]
                    neighbor_distance: Distance from each neighbor atoms to their center, size [B, M, N, 1]
        """
        if self.use_ring:
            centers, center_mask, neighbor, neighbor_mask, neighbor_weight, neighbor_distance, ring_info = inputs
        else:
            centers, center_mask, neighbor, neighbor_mask, neighbor_weight, neighbor_distance = inputs

        # Embedding atom and extra information as ring, aromatic
        centers = self.embed_atom(centers)

        if self.use_ring:
            ring_embed = self.extra_embed(ring_info)
            # Shape embed_atom [B, M, n_embedding + 10]
            centers = tf.concat([centers, ring_embed], -1)

        centers = self.dense_embed(centers)
        centers = self.dropout(centers)

        centers, list_center_rep, attn_local = self.local_attention_loop(centers, neighbor, neighbor_mask,
                                                                         neighbor_weight, neighbor_distance)

        # Dense layer after Local Attention -> representation for each local structure [B, M, d]
        centers = self.dense_afterLc(centers)

        # Using weighted attention score for combining structures representation
        if global_attn:
            attn_global, struc_rep = self.global_attention(
                centers, center_mask)
        else:
            struc_rep = tf.reduce_sum(centers * center_mask, axis=1)

        # Shape struct representation [B, d]
        struc_rep = self.dense_bftotal(struc_rep)

        # Shape predict_property [B, 1]
        predict_property = self.predict_property(struc_rep)

        if train:
            return predict_property

        else:
            if global_attn:
                return predict_property, attn_global, list_center_rep, attn_local
            else:
                return predict_property


def create_model(config, mode='train'):

    atomic = tf.keras.layers.Input(name='atomic', shape=(None,), dtype='int32')
    atom_mask = tf.keras.layers.Input(shape=[None, 1], name='atom_mask')

    neighbor = tf.keras.layers.Input(
        name='neighbors', shape=(None, None), dtype='int32')
    neighbor_mask = tf.keras.layers.Input(
        name='neighbor_mask', shape=(None, None), dtype='float32')

    neighbor_weight = tf.keras.layers.Input(
        name='neighbor_weight', shape=(None, None, 1), dtype='float32')
    neighbor_distance = tf.keras.layers.Input(
        name='neighbor_distance', shape=(None, None, 20), dtype='float32')

    if config['model']['use_ring']:
        shape = 2

        ring_info = tf.keras.layers.Input(
            name='ring_aromatic', shape=(None, shape), dtype='float32')

        inputs = [atomic,  atom_mask, neighbor,
                  neighbor_mask, neighbor_weight, neighbor_distance, ring_info]

    else:
        inputs = [atomic,  atom_mask, neighbor,
                  neighbor_mask, neighbor_weight, neighbor_distance]

    gammodel = SCANNet(config['model'])

    if mode == 'train':
        out_energy = gammodel(inputs)

        model = tf.keras.Model(inputs=inputs, outputs=[out_energy])

        model.summary()
        model.compile(loss=root_mean_squared_error,
                      optimizer=tf.keras.optimizers.Adam(config['hyper']['lr'],
                                                         gradient_transformers=[AutoClipper(10)]),
                      metrics=['mae', r2_square])

    if mode == 'infer':
        out_energy, attn_global, list_center_rep, attn_local = gammodel(
            inputs, train=False)

        model = tf.keras.Model(inputs=inputs, outputs=[
                               out_energy, attn_global, list_center_rep, attn_local])
        model.summary()

    return model
