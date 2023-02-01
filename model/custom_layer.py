import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
import math
import numpy as np

seed = 2134
tf.random.set_seed(seed)

import tensorflow_probability as tfp


class AutoClipper:
    """
        From paper: AutoClip: Adaptive Gradient Clipping
        https://github.com/pseeth/autoclip
    """
    def __init__(self, clip_percentile, history_size=10000):
        self.clip_percentile = clip_percentile
        self.grad_history = tf.Variable(tf.zeros(history_size), trainable=False)
        self.i = tf.Variable(0, trainable=False)
        self.history_size = history_size

    def __call__(self, grads_and_vars):
        grad_norms = [self._get_grad_norm(g) for g, _ in grads_and_vars]
        total_norm = tf.norm(grad_norms)
        assign_idx = tf.math.mod(self.i, self.history_size)
        self.grad_history = self.grad_history[assign_idx].assign(total_norm)
        self.i = self.i.assign_add(1)
        clip_value = tfp.stats.percentile(self.grad_history[: self.i], q=self.clip_percentile)
        return [(tf.clip_by_norm(g, clip_value), v) for g, v in grads_and_vars]

    def _get_grad_norm(self, t, axes=None, name=None):
        values = tf.convert_to_tensor(t.values if isinstance(t, tf.IndexedSlices) else t, name="t")

        # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
        l2sum = tf.math.reduce_sum(values * values, axes, keepdims=True)
        pred = l2sum > 0
        # Two-tap tf.where trick to bypass NaN gradients
        l2sum_safe = tf.where(pred, l2sum, tf.ones_like(l2sum))
        return tf.squeeze(tf.where(pred, tf.math.sqrt(l2sum_safe), l2sum))

class SGDR(tf.keras.callbacks.Callback):
    """This callback implements the learning rate schedule for
    Stochastic Gradient Descent with warm Restarts (SGDR),
    as proposed by Loshchilov & Hutter (https://arxiv.org/abs/1608.03983).

    The learning rate at each epoch is computed as:
    lr(i) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * i/num_epochs))

    Here, num_epochs is the number of epochs in the current cycle, which starts
    with base_epochs initially and is multiplied by mul_epochs after each cycle.

    # Example
        ```python
            sgdr = CyclicLR(lr_min=0.0, lr_max=0.05,
                                base_epochs=10, mul_epochs=2)
            model.compile(optimizer=keras.optimizers.SGD(decay=1e-4, momentum=0.9),
                          loss=loss)
            model.fit(X_train, Y_train, callbacks=[sgdr])
        ```

    # Arguments
        lr_min: minimum learning rate reached at the end of each cycle.
        lr_max: maximum learning rate used at the beginning of each cycle.
        base_epochs: number of epochs in the first cycle.
        mul_epochs: factor with which the number of epochs is multiplied
                after each cycle.
    """

    def __init__(self, lr_min=0.0, lr_max=0.05, base_epochs=10, mul_epochs=2):
        super(SGDR, self).__init__()

        self.lr_min = lr_min
        self.lr_max = lr_max
        self.base_epochs = base_epochs
        self.mul_epochs = mul_epochs

        self.cycles = 0.
        self.cycle_iterations = 0.
        self.trn_iterations = 0.

        self._reset()

    def _reset(self, new_lr_min=None, new_lr_max=None,
               new_base_epochs=None, new_mul_epochs=None):
        """Resets cycle iterations."""

        if new_lr_min != None:
            self.lr_min = new_lr_min
        if new_lr_max != None:
            self.lr_max = new_lr_max
        if new_base_epochs != None:
            self.base_epochs = new_base_epochs
        if new_mul_epochs != None:
            self.mul_epochs = new_mul_epochs
        self.cycles = 0.
        self.cycle_iterations = 0.

    def sgdr(self):

        cycle_epochs = self.base_epochs * (self.mul_epochs ** self.cycles)
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(np.pi * (self.cycle_iterations + 1) / cycle_epochs))

    def on_train_begin(self, logs=None):

        if self.cycle_iterations == 0:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.lr_max)
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
            tf.keras.backend.set_value(self.model.optimizer.lr, self.lr_max)
        else:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.sgdr())


class SGDRC(tf.keras.callbacks.Callback):
    """This callback implements the learning rate schedule for
    Stochastic Gradient Descent with warm Restarts (SGDR),
    as proposed by Loshchilov & Hutter (https://arxiv.org/abs/1608.03983).

    The learning rate at each epoch is computed as:
    lr(i) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * i/num_epochs))

    Here, num_epochs is the number of epochs in the current cycle, which starts
    with base_epochs initially and is multiplied by mul_epochs after each cycle.

    # Example
        ```python
            sgdr = SGDR(lr_min=0.0, lr_max=0.05,
                                base_epochs=10, mul_epochs=2)
            model.compile(optimizer=keras.optimizers.SGD(decay=1e-4, momentum=0.9),
                          loss=loss)
            model.fit(X_train, Y_train, callbacks=[sgdr])
        ```

    # Arguments
        lr_min: minimum learning rate reached at the end of each cycle.
        lr_max: maximum learning rate used at the beginning of each cycle.
        t0: number of epochs in the first cycle.
        tmult: factor with which the number of epochs is multiplied
                after each cycle.
    """
    def __init__(self, lr_max, lr_min, lr_max_compression=5, t0=10, tmult=1, trigger_val_mae=9999, show_lr=True):
        # Global learning rate max/min
        self.lr_max = lr_max
        self.lr_min = lr_min
        # Max learning rate compression
        self.lr_max_compression = lr_max_compression
        # Warm restarts params
        self.t0 = t0
        self.tmult = tmult
        # Learning rate decay trigger
        self.trigger_val_mae = trigger_val_mae
        # init parameters
        self.show_lr = show_lr
        self._init_params()        
        
    def _init_params(self):
        # Decay triggered
        self.triggered = False
        # Learning rate of next warm up
        self.lr_warmup_next = self.lr_max
        self.lr_warmup_current = self.lr_max
        # Current learning rate
        self.lr = self.lr_max
        # Current warm restart interval
        self.ti = self.t0
        # Warm restart count
        self.tcur = 1
        # Best validation accuracy
        self.best_val_mae = 9999

    def on_train_begin(self, logs):
        self._init_params()

    def on_epoch_end(self, epoch, logs):
        if not self.triggered and logs["val_mae"] <= self.trigger_val_mae:
            self.triggered = True

        if self.triggered:
            # Update next warmup lr when validation acc surpassed
            if logs["val_mae"] < self.best_val_mae:
                self.best_val_mae = logs["val_mae"]
                # Avoid lr_warmup_next too small
                if self.lr_max_compression > 0:
                    self.lr_warmup_next = max(self.lr_warmup_current / self.lr_max_compression, self.lr)
                else:
                    self.lr_warmup_next = self.lr
        if self.show_lr:
            print(f"sgdr_triggered = {self.triggered}, " +
                  f"current_lr = {self.lr:f}, next_warmup_lr = {self.lr_warmup_next:f}, next_warmup = {self.ti-self.tcur}")

    # SGDR
    def lr_scheduler(self, epoch):
        if not self.triggered: return self.lr
        # SGDR
        self.tcur += 1
        if self.tcur > self.ti:
            self.ti = int(self.tmult * self.ti)
            self.tcur = 1
            self.lr_warmup_current = self.lr_warmup_next
        self.lr = float(self.lr_min + (self.lr_warmup_current - self.lr_min) * (1 + np.cos(self.tcur/self.ti*np.pi)) / 2.0)
        return self.lr

class LocalAttention(tf.keras.layers.Layer):

    """
    Implements a local attention block
    """

    def __init__(self, dim=128, num_head=8, v_proj=True, scale=0.5,activation='swish',
                 name='LA_layer'):
        """_summary_

        Args:
            dim (int, optional): Dimension of projection for query and key attention. Defaults to 128.
            num_head (int, optional): Number of head attention use. head dim will be dim // num_head. Defaults to 8.
            v_proj (bool, optional): A Boolen for whether using value project or not. Defaults to True.
            scale (float, optional): A scalar for normalization attention value (default to Transformer paper). Defaults to 0.5.
            name (str, optional):  Defaults to 'LA_layer'.
        """
        super(LocalAttention, self).__init__()

        # Init hyperparameter
        self.hdim = dim // num_head
        self.scale=scale
        self.num_head=num_head

        self.v_proj=v_proj

        # Linear projection before attention
        self.proj_q=tf.keras.layers.Dense(
            dim, name='query',
            kernel_regularizer=regularizers.l2(1e-4))

        self.proj_k=tf.keras.layers.Dense(
            dim,  name='key',
            kernel_regularizer=regularizers.l2(1e-4))

        if self.v_proj:
            self.proj_v=tf.keras.layers.Dense(
                dim, name='value',
                kernel_regularizer=regularizers.l2(1e-4))

        # Filter gaussian distance - Distance embedding
        self.filter_dis=tf.keras.layers.Dense(
            dim , name='filter_dis', activation=activation,
            kernel_regularizer=regularizers.l2(1e-4))

    def call(self, atom_query, atom_neighbor, neighbor_distance, mask):
        """
        Args:
            atom_query:     A tensor of size [batch_size, len_atom_centers, dim]. Center representation
                            for all local structure
            atom_neighbor:  A tensor of size [batch_size,len_atom_centers, num_neighbors, dim].
                            Representation for all neighbor of center atoms
            neighbor_distance: A tensor of size [batch_size, len_atom_centers, num_neighbors, 1]
                            Distance from neighbor to center atoms
            mask:           A Boolen tensor for masking different number of neighbors for each center atoms
        """

        neighbor_distance=self.filter_dis(neighbor_distance)
        atom_neighbor=atom_neighbor * neighbor_distance

        # Query centers atoms shape [bs, len_atom_centers, dim]
        query=self.proj_q(atom_query)

        # Key neighbor atoms shape [bs, len_atom_centers, num_neighbors, dim]
        key=self.proj_k(atom_neighbor)

        if self.v_proj:
            value=self.proj_v(atom_neighbor)

        sh=tf.shape(atom_neighbor)
        bs=sh[0]
        qlen=sh[1]
        nlen=sh[2]
        # shape query_t [bs, len_atom_centers, heads dim]
        query_t=tf.reshape(query, [bs, -1, self.num_head, self.hdim])

        # shape key [bs, len_atom_centers, num_neighbors, heads dim]
        key=tf.reshape(key, [bs, -1, nlen, self.num_head, self.hdim])

        value=tf.reshape(value, [bs, -1, nlen, self.num_head, self.hdim])

        # shape query_t [bs, len_atom_centers, heads, dim] * [bs, len_atom_centers, num_neighbors, heads, dim]
        # shape energy [bs, heads, len_atom_centers, num_neighbors]
        dk=tf.cast(tf.shape(key)[-1], tf.float32)**(-self.scale)
        query_t=tf.multiply(query_t, dk)

        energy=tf.einsum('bchd,bcnhd->bhcn', query_t, key)

        # shape attn [bs, heads, len_atom_centers, num_neighbors] -> softmax over num_neighbors
        mask_scaled=(1.0 - tf.expand_dims(mask, 1)) * -1e9
        energy += mask_scaled

        attn=tf.nn.softmax(energy, -1)

        if self.v_proj:
            v=value
        else:
            v=key

        context=tf.einsum('bcn, bcnhd -> bcnhd', mask,
                            tf.einsum('bhcn, bcnhd -> bcnhd', attn, v))

        context=tf.reshape(
            context, [bs, qlen, nlen, self.num_head * self.hdim])

        # Taking sum over weighted neighbor representation and query representation for center representation
        context=tf.reduce_sum(context, 2) + query

        return attn, context


class GlobalAttention(tf.keras.layers.Layer):

    """
    Implements a global attention block
    """

    def __init__(self,  dim=128, v_proj=True, scale=0.5,  norm=True, name='GA_layer'):
        """

        Args:
            dim (int, optional): Dimension of projection for query and key attention. Defaults to 16.
            num_head (int, optional): Number of head attention use. Total dim will be dim * num_head. Defaults to 8.
            v_proj (bool, optional): A Boolen for whether using value project or not. Defaults to True.
            scale (float, optional): A scalar for normalization attention value. Defaults to 0.5.
            norm (bool, optional): A Boolen for whether using normalization for aggreation attention. Defaults to False.
            name (str, optional):  Defaults to 'GA_layer'.
        """
        super(GlobalAttention, self).__init__()

        # Setup
        self.dim=dim
        self.scale=scale
        self.norm=norm

        self.v_proj=v_proj

        # Linear proj. before attention
        self.proj_q=tf.keras.layers.Dense(
            dim, name='query', kernel_regularizer=regularizers.l2(1e-4))

        self.proj_k=tf.keras.layers.Dense(
            dim,  name='key', kernel_regularizer=regularizers.l2(1e-4))

        if self.v_proj:
            self.proj_v=tf.keras.layers.Dense(
                dim, name='value', kernel_regularizer=regularizers.l2(1e-4))

    def call(self, atom_query, mask):
        # Query centers atoms shape [bs, len_atom_centers, dim]
        query=self.proj_q(atom_query)

        # Key centers atoms shape [bs, len_atom_centers, dim]
        key=self.proj_k(atom_query)

        if self.v_proj:
            value=self.proj_v(atom_query)

        # shape energy [bs, len_atom_centers, len_atom_centers]
        N = tf.cast(tf.shape(query)[1], tf.float32)
        dk=tf.cast(tf.shape(key)[-1], tf.float32)**(-self.scale) / N
        query=tf.multiply(query, dk)

        energy=tf.einsum('bqd,bkd->bqk', query, key)
        energy=tf.multiply(mask, energy)

        identity = tf.eye(tf.shape(energy)[1],batch_shape=[tf.shape(energy)[0]], dtype='bool')
        mask_center = tf.cast(tf.math.logical_not(identity), tf.float32)

        # Calculate attention from other atom except for the center
        energy=tf.multiply(mask_center, energy)

        # Taking the sum of attention from all local structures
        # shape transform_energy [bs, len_atom_centers, 1]
        agg_attention= tf.reduce_sum(energy, -1)
        agg_attention= tf.reshape(
            agg_attention, [tf.shape(atom_query)[0], -1, 1])

        agg_attention=tf.multiply(mask, agg_attention)

        # Normalize the score for better softmax behaviors
        if self.norm:
            # Normalize score
            agg_attention, _=tf.linalg.normalize(
                agg_attention, ord='euclidean', axis=1, name=None
            )

        mask_scale=(1.0 - mask) * -1e9
        agg_attention += mask_scale

        attn=tf.nn.softmax(agg_attention, 1)

        if self.v_proj:
            v=value
        else:
            v=key

        # Multiply the attention score and local structure representation
        context=tf.multiply(mask, tf.einsum('bqj,bqd -> bqd', attn, v))

        context=tf.reduce_sum(context, 1)

        return attn, context


def root_mean_squared_error(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))


def mean_squared_error(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))


def r2_square(y_true, y_pred):
    SS_res=K.sum(K.square(y_true-y_pred))
    SS_tot=K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))
