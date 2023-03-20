import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from scannet.layers import *
from scannet.losses import *
from tensorflow.keras.layers import Embedding, Dense, Dropout, Input, Lambda


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

        self.model = create_model(config)

        self.model.compile(loss=root_mean_squared_error,
                           optimizer=tf.keras.optimizers.Adam(config['hyper']['lr'],
                                                              gradient_transformers=[AutoClipper(10)]),
                           metrics=['mae', r2_square])

    def __getattr__(self, p):
        return getattr(self.model, p)

    @classmethod
    def load_model_infer(cls, path):
        from scannet.layers import _CUSTOM_OBJECTS

        model = tf.keras.models.load_model(
            path, custom_objects=_CUSTOM_OBJECTS)

        attention_output = model.get_layer('global_attention').output[0]
        model_infer = tf.keras.Model(inputs=model.input, outputs=[
                                     model.output, attention_output])
        return model_infer

    @classmethod
    def load_model(cls, path):
        from scannet.layers import _CUSTOM_OBJECTS

        model = tf.keras.models.load_model(
            path, custom_objects=_CUSTOM_OBJECTS)

        return model


def gather_shape(x):
    x_shape = tf.shape(x)
    B, M, N = x_shape[0], x_shape[1], x_shape[2]

    range_B = tf.range(B)[:, None, None, None]
    range_B_t = tf.tile(range_B, [1, M, N, 1])
    x_indices = tf.concat(
        [range_B_t, tf.expand_dims(x, -1)], -1)

    return x_indices


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
