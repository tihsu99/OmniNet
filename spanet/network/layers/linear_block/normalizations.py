from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers

class MaskedBatchNorm(tf.keras.layers.Layer):
    def __init__(self, output_dim: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True):
        super(MaskedBatchNorm, self).__init__()

        self.track_running_stats = track_running_stats
        self.num_features = output_dim
        self.momentum = momentum
        self.affine = affine
        self.eps = eps

        # Register affine transform learnable parameters
        if affine:
            self.weight = self.add_weight(shape=(1, 1, output_dim), initializer="ones", trainable=True)
            self.bias = self.add_weight(shape=(1, 1, output_dim), initializer="zeros", trainable=True)
        else:
            self.weight = None
            self.bias = None

        # Register moving average storable parameters
        if track_running_stats:
            self.running_mean = self.add_weight(shape=(1, 1, output_dim), initializer="zeros", trainable=False)
            self.running_var = self.add_weight(shape=(1, 1, output_dim), initializer="ones", trainable=False)
            self.num_batches_tracked = self.add_weight(shape=(), initializer="zeros", trainable=False, dtype=tf.int64)
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.assign(tf.zeros_like(self.running_mean))
            self.running_var.assign(tf.ones_like(self.running_var))
            self.num_batches_tracked.assign(tf.zeros_like(self.num_batches_tracked))

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.assign(tf.ones_like(self.weight))
            self.bias.assign(tf.zeros_like(self.bias))

    def call(self, features: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        # Calculate the masked mean and variance
        timesteps, batch_size, feature_dim = tf.shape(features)[0], tf.shape(features)[1], tf.shape(features)[2]
        mask = tf.reshape(mask, (timesteps, batch_size))

        if mask is not None:
            masked_features = tf.boolean_mask(features, mask)
        else:
            masked_features = tf.reshape(features, (timesteps * batch_size, feature_dim))

        # Compute masked image statistics
        current_mean = tf.reshape(tf.reduce_mean(masked_features, axis=0), (1, 1, feature_dim))
        current_var = tf.reshape(tf.math.reduce_variance(masked_features, axis=0), (1, 1, feature_dim))

        # Update running statistics
        if self.track_running_stats and self.training:
            if self.num_batches_tracked == 0:
                self.running_mean.assign(current_mean)
                self.running_var.assign(current_var)
            else:
                self.running_mean.assign((1 - self.momentum) * self.running_mean + self.momentum * current_mean)
                self.running_var.assign((1 - self.momentum) * self.running_var + self.momentum * current_var)

            self.num_batches_tracked.assign_add(1)

        # Apply running statistics transform
        if self.track_running_stats and not self.training:
            normed_images = (features - self.running_mean) / tf.sqrt(self.running_var + self.eps)
        else:
            normed_images = (features - current_mean) / tf.sqrt(current_var + self.eps)

        # Apply affine transform from learned parameters
        if self.affine:
            normed_images = normed_images * self.weight + self.bias

        return normed_images * tf.expand_dims(mask, axis=2)


class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, output_dim: int):
        super(BatchNorm, self).__init__()

        self.output_dim = output_dim
        self.normalization = layers.BatchNormalization()

    def call(self, x: tf.Tensor, sequence_mask: tf.Tensor) -> tf.Tensor:
        max_jets, batch_size, output_dim = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        y = tf.reshape(x, (max_jets * batch_size, output_dim))
        y = self.normalization(y)
        return tf.reshape(y, (max_jets, batch_size, output_dim))


class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, output_dim: int):
        super(LayerNorm, self).__init__()

        self.output_dim = output_dim
        self.normalization = layers.LayerNormalization()

    def call(self, x: tf.Tensor, sequence_mask: tf.Tensor) -> tf.Tensor:
        return self.normalization(x)


def create_normalization(normalization: str, output_dim: int) -> tf.keras.layers.Layer:
    normalization = normalization.lower().replace("_", "").replace(" ", "")

    if normalization == "batchnorm":
        return BatchNorm(output_dim)
    elif normalization == "maskedbatchnorm":
        return MaskedBatchNorm(output_dim)
    elif normalization == "layernorm":
        return LayerNorm(output_dim)
    else:
        return layers.Lambda(lambda x, mask: x)

