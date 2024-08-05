import tensorflow as tf
from tensorflow.keras import layers

from spanet.options import Options
from spanet.network.layers.linear_block.gru_block import GRUGate, GRUBlock
from spanet.network.layers.transformer.transformer_base import TransformerBase


class GTrXL(tf.keras.layers.Layer):
    def __init__(self, options, hidden_dim: int, num_heads: int, dropout: float):
        super(GTrXL, self).__init__()

        self.attention_norm = layers.LayerNormalization()
        self.attention_gate = GRUGate(hidden_dim)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim,
            dropout=dropout
        )

        self.feed_forward = GRUBlock(options, hidden_dim, hidden_dim, skip_connection=True)

    def call(self, x: tf.Tensor, padding_mask: tf.Tensor, sequence_mask: tf.Tensor) -> tf.Tensor:
        output = self.attention_norm(x)
        output = self.attention(
            query=output,
            value=output,
            key=output,
            attention_mask=tf.cast(tf.expand_dims(tf.logical_not(padding_mask), axis=1), dtype=tf.int32)
        )

        output = self.attention_gate(output, x)

        return self.feed_forward(output, sequence_mask)


class GatedTransformer(TransformerBase):
    def __init__(self, options: Options, num_layers: int):
        super(GatedTransformer, self).__init__(options, num_layers)

        self.layers = [GTrXL(options, self.hidden_dim, self.num_heads, self.dropout) for _ in range(num_layers)]

    def call(self, x: tf.Tensor, padding_mask: tf.Tensor, sequence_mask: tf.Tensor) -> tf.Tensor:
        output = x

        for layer in self.layers:
            output = layer(output, padding_mask, sequence_mask)

        return output

