import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


@tf.custom_gradient
def mrelu(input_tensor):
    """
    A modified version of ReLU with linear gradient.
    """

    def grad(dy):
        return dy  # Linear gradient for the backward pass.

    return tf.maximum(input_tensor, 0), grad


def gather_shape(x):
    import tensorflow as tf

    x_shape = tf.shape(x)
    B, M, N = x_shape[0], x_shape[1], x_shape[2]

    range_B = tf.broadcast_to(tf.reshape(tf.range(B), [B, 1, 1, 1]), [B, M, N, 1])

    x_indices = tf.concat([range_B, tf.expand_dims(x, -1)], -1)

    return x_indices


class GaussianExpansion(tf.keras.layers.Layer):
    """
    Simple Gaussian expansion.
    A vector of distance [d1, d2, d3, ..., dn] is expanded to a
    matrix of shape [n, m], where m is the number of Gaussian basis centers

    """

    def __init__(self, centers, width=0.5, **kwargs):
        """
        Args:
            centers (np.ndarray): Gaussian basis centers
            width (float): width of the Gaussian basis
            **kwargs:
        """
        self.centers = centers

        if width is None:
            self.width = np.diff(self.centers).mean()
        else:
            self.width = width**2

        super().__init__(**kwargs)

    def call(self, inputs, masks=None):
        """
        The core logic function

        Args:
            inputs (tf.Tensor): input distance tensor, with shape [B, M, N]
            masks (tf.Tensor): bool tensor, not used here
        """
        return tf.math.exp(
            -((tf.expand_dims(inputs, -1) - tf.expand_dims(tf.expand_dims(self.centers, 0), 0)) ** 2) / self.width
        )

    def get_config(self):
        config = super(GaussianExpansion, self).get_config()

        config.update(
            {
                "centers": self.centers,
            }
        )
        return config


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
            model.compile(optimizer=keras.optimizers.SGD(decay=1e-4, momentum=0.9), loss=loss)
            model.fit(X_train, Y_train, callbacks=[sgdr])
        ```

    # Arguments
        lr_min: minimum learning rate reached at the end of each cycle.
        lr_max: maximum learning rate used at the beginning of each cycle.
        t0: number of epochs in the first cycle.
        tmult: factor with which the number of epochs is multiplied
                after each cycle.
    """

    def __init__(
        self,
        lr_max,
        lr_min,
        lr_max_compression=5,
        t0=10,
        tmult=1,
        trigger_val_mae=9999,
        show_lr=True,
    ):
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
            print(
                f"sgdr_triggered = {self.triggered}, "
                + f"current_lr = {self.lr:f}, next_warmup_lr = {self.lr_warmup_next:f}, next_warmup = {self.ti-self.tcur}"
            )

    # SGDR
    def lr_scheduler(self, epoch):
        if not self.triggered:
            return self.lr
        # SGDR
        self.tcur += 1
        if self.tcur > self.ti:
            self.ti = int(self.tmult * self.ti)
            self.tcur = 1
            self.lr_warmup_current = self.lr_warmup_next
        self.lr = float(
            self.lr_min + (self.lr_warmup_current - self.lr_min) * (1 + np.cos(self.tcur / self.ti * np.pi)) / 2.0
        )
        return self.lr
