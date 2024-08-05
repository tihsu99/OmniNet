import tensorflow as tf
from tensorflow.keras import layers

class Normalizer(tf.keras.layers.Layer):
    def __init__(self, mean: tf.Tensor, std: tf.Tensor):
        super(Normalizer, self).__init__()
        self.mean = tf.Variable(mean, trainable=False)
        self.std = tf.Variable(std, trainable=False)

    def call(self, data: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        data = (data - self.mean) / self.std
        return data * tf.expand_dims(mask, axis=-1)


