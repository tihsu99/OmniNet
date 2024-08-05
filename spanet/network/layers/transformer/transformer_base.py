import tensorflow as tf
from tensorflow.keras import layers

from spanet.options import Options

class TransformerBase(tf.keras.layers.Layer):
    def __init__(self, options: Options, num_layers: int):
        super(TransformerBase, self).__init__()

        self.num_layers = num_layers

        self.dropout = options.dropout
        self.hidden_dim = options.hidden_dim
        self.num_heads = options.num_attention_heads
        self.transformer_activation = options.transformer_activation
        self.dim_feedforward = int(round(options.transformer_dim_scale * options.hidden_dim))

    def call(self, x: tf.Tensor, padding_mask: tf.Tensor, sequence_mask: tf.Tensor) -> tf.Tensor:
        return x

