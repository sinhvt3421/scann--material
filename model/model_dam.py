import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers


def root_mean_squared_error(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))


def mean_squared_error(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))


def shifted_softplus(x):
    """
    Softplus nonlinearity shifted by -log(2) such that shifted_softplus(0.) = 0.

    y = log(0.5e^x + 0.5)

    """
    return tf.math.softplus(x) - tf.cast(tf.math.log(2.), tf.float32)


def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


class LocalAttention(tf.keras.layers.Layer):
    def __init__(self, v_dim=16, dim=16,
                 mode='dot', v_proj=True, scale=0.5,
                 num_head=8, tranformer=True, name='attn_local'):
        super(LocalAttention, self).__init__(name)

        # Setup
        self.v_dim = v_dim
        self.dim = dim
        self.scale = scale
        self.tranformer = tranformer
        self.num_head = num_head
        # Linear proj. before attention
        self.proj_q = tf.keras.layers.Dense(
            dim * num_head, name='query', kernel_regularizer=regularizers.l2(1e-4))

        self.proj_k = tf.keras.layers.Dense(
            dim * num_head,  name='key', kernel_regularizer=regularizers.l2(1e-4))

        self.proj_v = tf.keras.layers.Dense(
            dim * num_head, name='value', kernel_regularizer=regularizers.l2(1e-4))

        # Filter gaussian distance
        self.filter_dis = tf.keras.layers.Dense(
            v_dim * num_head, name='filter_dis', activation='relu',
            kernel_regularizer=regularizers.l2(1e-4))

    def call(self, atom_query, atom_neighbor, local_distance, mask):
        local_distance = self.filter_dis(local_distance)
        atom_neighbor = atom_neighbor * local_distance

        # Query centers atoms shape [bs, len_atom_centers, dim]
        query = self.proj_q(atom_query)

        # Key neighbor atoms shape [bs, len_atom_centers, num_neighbors, dim]
        key = self.proj_k(atom_neighbor)

        value = self.proj_v(atom_neighbor)
        # print(key)
        sh = tf.shape(atom_neighbor)
        bs = sh[0]
        qlen = sh[1]
        nlen = sh[2]
        # shape query_t [bs, heads, len_atom_centers, dim]
        query_t = tf.reshape(query, [bs, -1, self.num_head, self.dim])
        query_t = tf.transpose(query_t, [0, 2, 1, 3])

        # shape key [bs, heads, len_atom_centers, num_neighbors, dim]
        key = tf.reshape(key, [bs, -1, nlen, self.num_head, self.dim])
        key = tf.transpose(key, [0, 3, 1, 2, 4])

        value = tf.reshape(value, [bs, -1, nlen, self.num_head, self.dim])
        value = tf.transpose(value, [0, 3, 1, 2, 4])

        # shape query_t [bs, heads, len_atom_centers, 1, dim] * [bs, heads, len_atom_centers, num_neighbors, dim]
        # shape energy [bs, heads, len_atom_centers, num_neighbors , 1]
        # energy = tf.matmul(key, tf.expand_dims(query_t, 3), transpose_b=True)
        energy = tf.expand_dims(query_t, 3) * key
        # shape mask [bs, 1, len_atom_centers, num_neighbors , 1]
        energy = tf.multiply(tf.expand_dims(
            tf.expand_dims(mask, 1), -1), energy)

        dk = tf.cast(tf.shape(key)[-1], tf.float32)**(-self.scale)
        scaled_energy = energy * dk

        # shape attn [bs, heads, len_atom_centers, num_neighbors , 1] -> softmax over num_neighbors
        attn = tf.nn.softmax(scaled_energy, -2)
        context = attn * \
            tf.multiply(tf.expand_dims(tf.expand_dims(mask, 1), -1), value)
        context = tf.reshape(tf.transpose(context, [0, 2, 3, 1, 4]), [
            bs, qlen, nlen, self.num_head * self.dim])

        context = tf.reduce_sum(context, 2) + query

        return attn, context


class GlobalAttention(tf.keras.layers.Layer):
    def __init__(self, v_dim=16, dim=16,
                 mode='dot', v_proj=True, scale=0.5, num_head=8, name='attn'):
        super(GlobalAttention, self).__init__(name)

        # Setup
        self.v_dim = v_dim
        self.dim = dim
        self.scale = scale

        # Linear proj. before attention
        self.proj_q = tf.keras.layers.Dense(
            dim*num_head, name='query', kernel_regularizer=regularizers.l2(1e-4))

        self.proj_k = tf.keras.layers.Dense(
            dim*num_head,  name='key', kernel_regularizer=regularizers.l2(1e-4))

        self.proj_v = tf.keras.layers.Dense(
            dim*num_head, name='value', kernel_regularizer=regularizers.l2(1e-4))

        self.out = tf.keras.layers.Dense(1, name='out')

    def call(self, atom_query, mask):
        # Query centers atoms shape [bs, len_atom_centers, dim]
        query = self.proj_q(atom_query)

        # Key centers atoms shape [bs, len_atom_centers, dim]
        key = self.proj_k(atom_query)

        # shape energy [bs, len_atom_centers, len_atom_centers]
        energy = tf.matmul(query, key, transpose_b=True)
        energy = tf.multiply(mask, energy)

        dk = tf.cast(tf.shape(key)[-1], tf.float32)**(-self.scale)
        scaled_energy = energy * dk

        # shape attn [bs, len_atom_centers, len_atom_centers]
        attn = tf.nn.softmax(scaled_energy)
        # shape mask [bs, len_atom_centers, 1]
        attn = tf.multiply(mask, attn)

        # value = tf.matmul(attn, query)
        # #Global score [bs, len_atom_centers, 1]
        # score = self.out(value)
        # score = tf.nn.softmax(tf.multiply(mask,score),1)

        # # Sum over masked centers -> shape context [bs, 1]
        # context = tf.reduce_sum(
        #     score * tf.multiply(mask, self.proj_v(atom_query)), 1)

        context = tf.reduce_sum(tf.matmul(attn,  self.proj_v(atom_query)), 1)
        return attn, context


class DAMNet(tf.keras.models.Model):
    def __init__(self, config):
        super(DAMNet, self).__init__()
        self.n_attention = config['n_attention']

        # n layers Local Attention
        self.local_attention = [LocalAttention(name='attn_Lc_'+str(i), v_dim=config['v_dim'],
                                               dim=config['dim'], num_head=config['num_head'])
                                for i in range(config['n_attention'])]

        self.forward_trans = [tf.keras.layers.Dense(config['dense_embed'],
                                                    name='forward_trans' + str(i), dtype='float32',
                                                    kernel_regularizer=regularizers.l2(1e-4))
                              for i in range(config['n_attention'])]

        self.layer_norm = [tf.keras.layers.LayerNormalization(name='layer_norm_' + str(i), epsilon=1e-6)
                           for i in range(config['n_attention'])]

        self.forward_norm = [tf.keras.layers.LayerNormalization(name='forward_norm_' + str(i), epsilon=1e-6)
                             for i in range(config['n_attention'])]

        # Embeding for atomic number and other extra information as ring, aromatic,...
        self.embed_atom = tf.keras.layers.Embedding(config['n_atoms'],
                                                    config['n_embedding'],
                                                    name='embed_atom', dtype='float32')

        self.dense_embed = tf.keras.layers.Dense(config['dense_embed'],
                                                 activation='relu', name='dense_embed',
                                                 dtype='float32')
        self.extra_embed = tf.keras.layers.Dense(
            10, name='extra_embed', dtype='float32')

        # Dropout for embedding and attention layers
        self.dropout_em = tf.keras.layers.Dropout(0.2)

        self.dropout_att = [tf.keras.layers.Dropout(
            0.2) for i in range(config['n_attention'])]

        # Dense layer before Global Attention
        self.dense_afterLc = tf.keras.layers.Dense(
            config['dense_out'], activation='relu', name='after_Lc',
            kernel_regularizer=regularizers.l2(1e-4))

        self.global_attention = GlobalAttention(name='attn_Gl', v_dim=config['v_dim'],
                                                dim=config['dim'], num_head=config['num_head'])
        # Dense layer on structure representation
        self.dense_bftotal = tf.keras.layers.Dense(
            config['dense_out'], activation='relu', name='bf_property',
            kernel_regularizer=regularizers.l2(1e-4))

        self.predict_property = tf.keras.layers.Dense(
            1, name='predict_property')

    def call(self, inputs, train=True, lats_attn=True):
        atoms, ring_info, mask_atom, local, mask, local_weight, local_distance = inputs

        # embedding atom and extra information as ring, aromatic
        embed_atom = self.embed_atom(atoms)
        embed_ring = self.extra_embed(ring_info)

        # shape embed_atom [bs, len_atom_centers, n_embedding + 10]
        embed_atom = tf.concat([embed_atom, embed_ring], -1)
        dense_embed = self.dense_embed(embed_atom)

        # get neighbor vector from local indices
        sh = tf.shape(local)
        rang = tf.range(sh[0])[:, None, None, None]
        rang_t = tf.tile(rang, [1, sh[1], sh[2], 1])
        indices = tf.concat([rang_t, tf.expand_dims(local, -1)], -1)

        neighbors = tf.gather_nd(dense_embed, indices)

        # multiply weight Voronoi with neibor
        neighbor_weighted = neighbors * local_weight

        # shape neighbor_weighted [bs, len_atom_centers, num_neighbors, embedding_dim ]
        neighbor_weighted = tf.reshape(
            neighbor_weighted, [sh[0], sh[1], sh[2], dense_embed.shape[-1]])

        # Local Attention loop layers
        for i in range(self.n_attention-1):

            attn_local, context = self.local_attention[i](
                dense_embed, neighbor_weighted, local_distance,  mask)

            # 2 Forward Norm layers
            attention_norm = self.layer_norm[i](context+dense_embed)
            f_out = self.forward_trans[i](attention_norm)

            dense_embed = self.forward_norm[i](f_out+attention_norm)

            # dense_embed = self.dropout_att[i](dense_embed)

            # Get neighbor_weighted from changed centers
            neighbor_weighted = tf.gather_nd(
                dense_embed, indices) * local_weight
            neighbor_weighted = tf.reshape(
                neighbor_weighted, [sh[0], sh[1], sh[2], dense_embed.shape[-1]])

        # Last layer Local Attention, don't need gather_nd step
        attn_local, context = self.local_attention[self.n_attention-1](
            dense_embed, neighbor_weighted, local_distance, mask)

        attention_norm = self.layer_norm[self.n_attention -
                                         1](context+dense_embed)
        f_out = self.forward_trans[self.n_attention-1](attention_norm)

        dense_embed = self.forward_norm[self.n_attention -
                                        1](f_out+attention_norm)
        # Dense layer after Local Attention -> representation for each atoms [bs, len_atoms_centers, dim]
        dense_embed = self.dense_afterLc(dense_embed)
        # dense_embed = self.dropout_em(dense_embed)
        # Using weighted attention score for combining structures representation
        if lats_attn:
            attn_global, struc_rep = self.global_attention(
                dense_embed, mask_atom)
        else:
            struc_rep = tf.reduce_sum(dense_embed * mask_atom, axis=1)

        # shape struct_rep [bs, dim_out]
        struc_rep = self.dense_bftotal(struc_rep)

        # shape predict_property [bs, 1]
        predict_property = self.predict_property(struc_rep)

        if train:
            return predict_property
        else:
            return predict_property, context, attn_local, attn_global, struc_rep, dense_embed


def create_model(config):

    atomic = tf.keras.layers.Input(name='atomic', shape=(None,), dtype='int32')
    ring_info = tf.keras.layers.Input(
        name='ring_aromatic', shape=(None, 2), dtype='float32')

    mask_atom = tf.keras.layers.Input(shape=[None, 1], name='mask_atom')

    local = tf.keras.layers.Input(
        name='locals', shape=(None, None), dtype='int32')
    mask_local = tf.keras.layers.Input(
        name='mask_local', shape=(None, None), dtype='float32')
    local_weight = tf.keras.layers.Input(
        name='local_weight', shape=(None, None, 1), dtype='float32')
    local_distance = tf.keras.layers.Input(
        name='local_distance', shape=(None, None, 20), dtype='float32')

    dammodel = DAMNet(config['model'])
    out = dammodel([atomic, ring_info, mask_atom, local, mask_local,
                    local_weight, local_distance])

    model = tf.keras.Model(
        inputs=[atomic, ring_info,  mask_atom, local, mask_local, local_weight, local_distance], outputs=[out])

    model.summary()
    model.compile(loss=root_mean_squared_error,
                  optimizer=tf.keras.optimizers.Adam(config['hyper']['lr'],
                                                     decay=1e-5, clipnorm=100),
                  metrics=['mae', coeff_determination])
    return model


def create_model_infer(config):
    atomic = tf.keras.layers.Input(name='atomic', shape=(None,), dtype='int32')
    ring_info = tf.keras.layers.Input(
        name='ring_aromatic', shape=(None, 2), dtype='int32')
    mask_atom = tf.keras.layers.Input(shape=[None, 1], name='mask_atom')

    local = tf.keras.layers.Input(
        name='locals', shape=(None, None), dtype='int32')
    mask_local = tf.keras.layers.Input(
        name='mask_local', shape=(None, None), dtype='float32')
    local_weight = tf.keras.layers.Input(
        name='local_weight', shape=(None, None, 1), dtype='float32')
    local_distance = tf.keras.layers.Input(
        name='local_distance', shape=(None, None, 20), dtype='float32')

    dammodel = DAMNet(config['model'])
    out_energy, context, attn_local, attn_global, struc_rep, dense_embed = dammodel([atomic, ring_info, mask_atom, local, mask_local,
                                                                                     local_weight, local_distance], False, True)

    model = tf.keras.Model(
        inputs=[atomic, ring_info, mask_atom, local, mask_local, local_weight, local_distance], outputs=[out_energy, context, attn_local, attn_global, struc_rep, dense_embed])

    model.summary()
    return model
