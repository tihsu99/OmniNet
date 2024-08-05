import tensorflow as tf
from tensorflow.keras import layers

from spanet.options import Options
from spanet.network.layers.linear_block.masking import create_masking
from spanet.network.layers.transformer.transformer_base import TransformerBase


class StandardTransformer(TransformerBase):
    def __init__(self, options: Options, num_layers: int):
        super(StandardTransformer, self).__init__(options, num_layers)

        self.masking = create_masking(options.masking)
        self.transformer_layers = [
            layers.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                num_heads=self.num_heads,
                feed_forward_dim=self.dim_feedforward,
                dropout=self.dropout,
                activation=self.transformer_activation
            ) for _ in range(num_layers)
        ]
        self.transformer = layers.TransformerEncoder(self.transformer_layers)

    def call(self, x: tf.Tensor, padding_mask: tf.Tensor, sequence_mask: tf.Tensor) -> tf.Tensor:
        padding_mask = tf.cast(tf.expand_dims(tf.logical_not(padding_mask), axis=1), dtype=tf.int32)
        output = self.transformer(x, mask=padding_mask)
        return self.masking(output, sequence_mask)

