import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
import math
import numpy as np


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
            # self.max_lr = self.max_lr / 2
            tf.keras.backend.set_value(self.model.optimizer.lr, self.max_lr)
        else:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.sgdr())


class LocalAttention(tf.keras.layers.Layer):

    """
    Implements a local attention block
    """

    def __init__(self, dim=16, num_head=8, v_proj=True, scale=0.5,
                 name='LA_layer'):
        """_summary_

        Args:
            dim (int, optional): Dimension of projection for query and key attention. Defaults to 16.
            num_head (int, optional): Number of head attention use. Total dim will be dim * num_head. Defaults to 8.
            v_proj (bool, optional): A Boolen for whether using value project or not. Defaults to True.
            scale (float, optional): A scalar for normalization attention value (default to Transformer paper). Defaults to 0.5.
            name (str, optional):  Defaults to 'LA_layer'.
        """
        super(LocalAttention, self).__init__(name)

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
        query_t = tf.multiply(query_t, dk)

        energy = tf.einsum('bchd,bcnhd->bhcn', query_t, key)

        # shape attn [bs, heads, len_atom_centers, num_neighbors] -> softmax over num_neighbors
        mask_scaled = (1.0 - tf.expand_dims(mask, 1)) * -1e9
        energy += mask_scaled

        attn = tf.nn.softmax(energy, -1)

        if self.v_proj:
            v = value
        else:
            v = key

        context = tf.einsum('bcn, bcnhd -> bcnhd', mask,
                            tf.einsum('bhcn, bcnhd -> bcnhd', attn, v))

        context = tf.reshape(
            context, [bs, qlen, nlen, self.num_head * self.dim])

        # Taking sum over weighted neighbor representation and query representation for center representation
        context = tf.reduce_sum(context, 2) + query

        return attn, context


class GlobalAttention(tf.keras.layers.Layer):

    """
    Implements a global attention block
    """

    def __init__(self,  dim=16, num_head=8,
                 v_proj=True, scale=0.5,  norm=False, name='GA_layer'):
        """

        Args:
            dim (int, optional): Dimension of projection for query and key attention. Defaults to 16.
            num_head (int, optional): Number of head attention use. Total dim will be dim * num_head. Defaults to 8.
            v_proj (bool, optional): A Boolen for whether using value project or not. Defaults to True.
            scale (float, optional): A scalar for normalization attention value. Defaults to 0.5.
            norm (bool, optional): A Boolen for whether using normalization for aggreation attention. Defaults to False.
            name (str, optional):  Defaults to 'GA_layer'.
        """
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

        energy = tf.einsum('bqd,bkd->bqk', query, key)
        energy = tf.multiply(mask, energy)

        # Taking the sum of attention from all local structures
        # shape transform_energy [bs, len_atom_centers, 1]
        agg_attention = tf.reduce_sum(energy, -1)
        agg_attention = tf.reshape(
            agg_attention, [tf.shape(atom_query)[0], -1, 1])

        agg_attention = tf.multiply(mask, agg_attention)

        # Normalize the score for better softmax behaviors
        if self.norm:
            # Normalize score
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
        context = tf.multiply(mask, tf.einsum('bqj,bqd -> bqd', attn, v))

        context = tf.reduce_sum(context, 1)

        return attn, context


def root_mean_squared_error(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))


def mean_squared_error(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))


def r2_square(y_true, y_pred):
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))
