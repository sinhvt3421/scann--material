import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
import tensorflow_addons as tfa


def root_mean_squared_error(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))


def mean_squared_error(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))

def r2_square(y_true, y_pred):
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


class LocalAttention(tf.keras.layers.Layer):

    """
    Implements a local attention block
    """

    def __init__(self, dim=16, num_head=8, v_proj=True, scale=0.5,
                  name='LA_layer'):
        super(LocalAttention, self).__init__(name)
        """
        Args:
            dim:        Dimension of projection for query and key attention.
            num_head:   Number of head attention use. Total dim will be dim * num_head
            v_proj:     A Boolen for whether using value project or not
            scale:      A scalar for normalization attention value (default to Transformer paper)
        """

        # Init hyperparameter
        self.dim = dim
        self.scale = scale
        self.num_head = num_head

        self.v_proj = v_proj

        # Linear projection before attention
        self.proj_q = tf.keras.layers.Dense(
            dim * num_head, name='query', 
            kernel_regularizer=regularizers.l2(1e-4))

        self.proj_k = tf.keras.layers.Dense(
            dim * num_head,  name='key', 
            kernel_regularizer=regularizers.l2(1e-4))
        
        if self.v_proj:
            self.proj_v = tf.keras.layers.Dense(
                dim * num_head, name='value', 
                kernel_regularizer=regularizers.l2(1e-4))

        # Filter gaussian distance - Distance embedding
        self.filter_dis = tf.keras.layers.Dense(
            dim * num_head, name='filter_dis', activation='swish',
            kernel_regularizer=regularizers.l2(1e-4))

    def call(self, atom_query, atom_neighbor, local_distance, mask):
        """
        Args:
            atom_query:     A tensor of size [batch_size, len_atom_centers, dim]. Center representation 
                            for all local structure
            atom_neighbor:  A tensor of size [batch_size,len_atom_centers, num_neighbors, dim].
                            Representation for all neighbor of center atoms
            local_distance: A tensor of size [batch_size, len_atom_centers, num_neighbors, 1]
                            Distance from neighbor to center atoms 
            mask:           A Boolen tensor for masking different number of neighbors for each center atoms
        """

        local_distance = self.filter_dis(local_distance)
        atom_neighbor = atom_neighbor * local_distance

        # Query centers atoms shape [bs, len_atom_centers, dim]
        query = self.proj_q(atom_query)

        # Key neighbor atoms shape [bs, len_atom_centers, num_neighbors, dim]
        key = self.proj_k(atom_neighbor)

        if self.v_proj:
            value = self.proj_v(atom_neighbor)

        sh = tf.shape(atom_neighbor)
        bs = sh[0]
        qlen = sh[1]
        nlen = sh[2]
        # shape query_t [bs, len_atom_centers, heads dim]
        query_t = tf.reshape(query, [bs, -1, self.num_head, self.dim])

        # shape key [bs, len_atom_centers, num_neighbors, heads dim]
        key = tf.reshape(key, [bs, -1, nlen, self.num_head, self.dim])

        value = tf.reshape(value, [bs, -1, nlen, self.num_head, self.dim])


        # shape query_t [bs, len_atom_centers, heads, dim] * [bs, len_atom_centers, num_neighbors, heads, dim]
        # shape energy [bs, heads, len_atom_centers, num_neighbors]
        dk = tf.cast(tf.shape(key)[-1], tf.float32)**(-self.scale)
        query_t = tf.multiply(query_t , dk)

        energy = tf.einsum('bchd,bcnhd->bhcn', query_t, key)

        # shape attn [bs, heads, len_atom_centers, num_neighbors] -> softmax over num_neighbors
        mask_scaled = (1.0 - tf.expand_dims(mask, 1)) * -1e9
        energy += mask_scaled
        
        attn = tf.nn.softmax(energy, -1)

        if self.v_proj:
            v = value
        else:
            v = key

        context = tf.einsum('bcn, bcnhd -> bcnhd', mask, tf.einsum('bhcn, bcnhd -> bcnhd',attn,v))

        context = tf.reshape(context, [bs, qlen, nlen, self.num_head * self.dim])

        #Taking sum over weighted neighbor representation and query representation for center representation
        context = tf.reduce_sum(context, 2) + query

        return  attn, context


class GlobalAttention(tf.keras.layers.Layer):

    """
    Implements a global attention block
    """

    def __init__(self,  dim=16, num_head=8, 
                v_proj=True, scale=0.5,  norm=False, name='GA_layer'):
        super(GlobalAttention, self).__init__(name)

        # Setup
        self.dim = dim
        self.scale = scale
        self.norm = norm

        self.v_proj = v_proj

        # Linear proj. before attention
        self.proj_q = tf.keras.layers.Dense(
            dim*num_head, name='query', kernel_regularizer=regularizers.l2(1e-4))

        self.proj_k = tf.keras.layers.Dense(
            dim*num_head,  name='key', kernel_regularizer=regularizers.l2(1e-4))

        if self.v_proj:
            self.proj_v = tf.keras.layers.Dense(
                dim*num_head, name='value', kernel_regularizer=regularizers.l2(1e-4))


    def call(self, atom_query, mask):
        # Query centers atoms shape [bs, len_atom_centers, dim]
        query = self.proj_q(atom_query)

        # Key centers atoms shape [bs, len_atom_centers, dim]
        key = self.proj_k(atom_query)

        if self.v_proj:
            value = self.proj_v(atom_query)

        # shape energy [bs, len_atom_centers, len_atom_centers]
        dk = tf.cast(tf.shape(key)[-1], tf.float32)**(-self.scale)
        query = tf.multiply(query, dk)

        energy = tf.einsum('bqd,bkd->bqk',query,key)
        energy = tf.multiply(mask, energy)

        # Taking the sum of attention from all local structures
        # shape transform_energy [bs, len_atom_centers, 1]
        agg_attention = tf.reduce_sum(energy, -1)
        agg_attention = tf.reshape(
            agg_attention, [tf.shape(atom_query)[0], -1, 1])

        agg_attention = tf.multiply(mask, agg_attention)

        # Normalize the score for better softmax behaviors
        if self.norm:
            #Normalize score
            agg_attention, _ = tf.linalg.normalize(
                agg_attention, ord='euclidean', axis=1, name=None
            )

        mask_scale = (1.0 - mask) * -1e9
        agg_attention += mask_scale

        attn = tf.nn.softmax(agg_attention, 1)
    
        if self.v_proj:
            v = value
        else:
            v = key

        # Multiply the attention score and local structure representation
        context = tf.multiply(mask, tf.einsum('bqj,bqd -> bqd',attn,v))

        context = tf.reduce_sum(context, 1)

        return attn, context


class GAMNet(tf.keras.models.Model):
    """
        Implements main GAMNet 
    """

    def __init__(self, config, mean=0.0, std=1.0):
        """
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
        super(GAMNet, self).__init__()

        # Number of attention layer
        self.n_attention = config['n_attention']

        # Input is whether molecule or crystal material
        self.mol = config['use_ring']

        # Whether using normalization layer for local attention as in Transformer
        self.attn_norm = config['use_attn_norm']

        self.mean = mean
        self.std = std

        # Embeding for atomic number and other extra information as ring, aromatic,...
        self.embed_atom = tf.keras.layers.Embedding(config['n_atoms'],
                                        config['n_embedding'],
                                        name='embed_atom', 
                                        dtype='float32')

        self.extra_embed = tf.keras.layers.Dense(10, name='extra_embed', dtype='float32')

        self.dense_embed = tf.keras.layers.Dense(config['dense_embed'],
                                                 activation='swish', name='dense_embed',
                                                 dtype='float32')

        # L layers Local Attention
        self.local_attention = [LocalAttention(name='LA_layer_'+str(i), 
                                               dim=config['dim'], num_head=config['num_head'])
                                for i in range(config['n_attention'])]

        if self.attn_norm:
            self.forward_trans = [tf.keras.layers.Dense(config['dense_embed'],
                                                        name='forward_trans' + str(i), dtype='float32',
                                                        kernel_regularizer=regularizers.l2(1e-4))
                                for i in range(config['n_attention'])]

            self.layer_norm = [tf.keras.layers.LayerNormalization(name='layer_norm_' + str(i), epsilon=1e-6)
                            for i in range(config['n_attention'])]

            self.forward_norm = [tf.keras.layers.LayerNormalization(name='forward_norm_' + str(i), epsilon=1e-6)
                                for i in range(config['n_attention'])]

       

        # Dense layer before Global Attention
        self.dense_afterLc = tf.keras.layers.Dense(
            config['dense_out'], activation='swish', name='after_Lc',
            kernel_regularizer=regularizers.l2(1e-4))

        # Global Attention layer
        self.global_attention = GlobalAttention(name='GA_layer', 
                                                dim=config['dim'], num_head=config['num_head'],scale=config['scale'],
                                                norm=config['use_norm'],softmax=config['softmax'])

        # Dense layer on structure representation
        self.dense_bftotal = tf.keras.layers.Dense(
            config['dense_out'], activation='swish', name='bf_property',
            kernel_regularizer=regularizers.l2(1e-4))

        # Output property
        self.predict_property = tf.keras.layers.Dense(1, name='predict_property')

    def call(self, inputs, train=True, global_attn=True):
        """
            Args:
                M: Number of centers atoms
                N: Number of local atoms
                atoms: Atomic number for each atom in the structure, size [B, M, 1]
                ring_info: Whether an atom in a Ring or Aromatic, size [B, M, 2]
                mask_atom: Masking for padding atoms in a batch up to M atoms, size [B, M, 1]
                local: Indexes of local atoms for each center in the structure, size [B, M, N, 1]
                mask: Masking for padding local indexes in a batch up to N neighbors, size [B, M, N, 1]
                local_weight: Voronoi weight of each local atoms for each center, size [B, M, N, 1]
                local_distance: Distance from each local atoms to their center, size [B, M, N, 1]
        """
        if self.mol:
            atoms, ring_info, mask_atom, local, mask, local_weight, local_distance = inputs
        else:
            atoms, mask_atom, local, mask, local_weight, local_distance = inputs

        # Embedding atom and extra information as ring, aromatic
        embed_atom = self.embed_atom(atoms)

        if self.mol:
            embed_ring = self.extra_embed(ring_info)
            # Shape embed_atom [B, M, n_embedding + 10]
            embed_atom = tf.concat([embed_atom, embed_ring], -1)

        dense_embed = self.dense_embed(embed_atom)

        # Get neighbor vector from local indices
        sh = tf.shape(local)
        rang = tf.range(sh[0])[:, None, None, None]
        rang_t = tf.tile(rang, [1, sh[1], sh[2], 1])
        indices = tf.concat([rang_t, tf.expand_dims(local, -1)], -1)

        # Get neighbor vector from indices, size [B, M, N, d]
        neighbors = tf.gather_nd(dense_embed, indices)

        # Multiply weight Voronoi with neighbor
        neighbor_weighted = neighbors * local_weight

        # Shape neighbor_weighted [B, M, N, embedding_dim ]
        neighbor_weighted = tf.reshape(
            neighbor_weighted, [sh[0], sh[1], sh[2], dense_embed.shape[-1]])

        # Local Attention recursive layers
        for i in range(self.n_attention-1):
            
            # Local attention for local structure representation
            attn_local, context = self.local_attention[i](dense_embed, neighbor_weighted, local_distance,  mask)

            if self.attn_norm:
                # 2 Forward Norm layers
                attention_norm = self.layer_norm[i](context+dense_embed)

                f_out = self.forward_trans[i](attention_norm)

                dense_embed = self.forward_norm[i](f_out+attention_norm)
            else:
                dense_embed = context

            # Get neighbor representation after updated from attention
            neighbor_weighted = tf.gather_nd(dense_embed, indices) * local_weight
                
            neighbor_weighted = tf.reshape(
                neighbor_weighted, [sh[0], sh[1], sh[2], dense_embed.shape[-1]])

        
        # Last layer Local Attention, don't need gather_nd step
        attn_local, context = self.local_attention[self.n_attention-1](
            dense_embed, neighbor_weighted, local_distance, mask)

        if self.attn_norm:
            attention_norm = self.layer_norm[self.n_attention -
                                            1](context+dense_embed)
            f_out = self.forward_trans[self.n_attention-1](attention_norm)

            dense_embed = self.forward_norm[self.n_attention -
                                            1](f_out+attention_norm)
        else:
            dense_embed = context

        # Dense layer after Local Attention -> representation for each local structure [B, M, d]
        dense_embed = self.dense_afterLc(dense_embed)
  
        # Using weighted attention score for combining structures representation
        if global_attn:
            attn_global, struc_rep = self.global_attention(
                dense_embed, mask_atom)
        else:
            struc_rep = tf.reduce_sum(dense_embed * mask_atom, axis=1)

        # Shape struct representation [B, d]
        struc_rep = self.dense_bftotal(struc_rep)

        # Shape predict_property [B, 1]
        predict_property = self.predict_property(struc_rep) * self.std + self.mean

        if train:
            return predict_property
        else:
            return predict_property, attn_global


def create_model(config, mean=0.0, std=1.0, mode='train'):

    atomic = tf.keras.layers.Input(name='atomic', shape=(None,), dtype='int32')
    mask_atom = tf.keras.layers.Input(shape=[None, 1], name='mask_atom')

    local = tf.keras.layers.Input(
        name='locals', shape=(None, None), dtype='int32')
    mask_local = tf.keras.layers.Input(
        name='mask_local', shape=(None, None), dtype='float32')

    local_weight = tf.keras.layers.Input(
        name='local_weight', shape=(None, None, 1), dtype='float32')
    local_distance = tf.keras.layers.Input(
        name='local_distance', shape=(None, None, 20), dtype='float32')

    if config['model']['use_ring']:
        ring_info = tf.keras.layers.Input(
            name='ring_aromatic', shape=(None, 2), dtype='float32')

        inputs = [atomic, ring_info,  mask_atom, local,
                  mask_local, local_weight, local_distance]
     
    else:
        inputs = [atomic,  mask_atom, local,
                  mask_local, local_weight, local_distance]

    if mode == 'train':
        gammodel = GAMNet(config['model'], mean, std)

        out_energy = gammodel(inputs)

        model = tf.keras.Model(inputs=inputs, outputs=[out_energy])

        model.summary()
        model.compile(loss=root_mean_squared_error,
                    optimizer=tf.keras.optimizers.Adam(config['hyper']['lr'], clipnorm=10),
                    metrics=['mae', r2_square])
        

    if mode == 'infer':
        out_energy, attn_global = gammodel(inputs, train=False)

        model = tf.keras.Model(inputs=inputs, outputs=[out_energy, attn_global])
        model.summary()

    return model

