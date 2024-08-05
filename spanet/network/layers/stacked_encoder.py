from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers

from spanet.options import Options
from spanet.network.layers.transformer import create_transformer
from spanet.network.layers.linear_stack import create_linear_stack


class StackedEncoder(tf.keras.layers.Layer):
    def __init__(
            self,
            options: Options,
            num_linear_layers: int,
            num_encoder_layers: int
    ):
        super(StackedEncoder, self).__init__()

        self.particle_vector = self.add_weight(
            shape=(1, 1, options.hidden_dim),
            initializer='random_normal',
            trainable=True
        )

        self.encoder = create_transformer(options, num_encoder_layers)
        self.embedding = create_linear_stack(options, num_linear_layers, options.hidden_dim, options.skip_connections)

    def call(self, encoded_vectors: tf.Tensor, padding_mask: tf.Tensor, sequence_mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """ Apply time-independent linear layers followed by a transformer encoder.

        This is used during the branches and symmetric attention layers.

        Parameters
        ----------
        encoded_vectors: [T, B, D]
            Input sequence to predict on.
        padding_mask : [B, T]
            Negative mask for transformer input.
        sequence_mask : [T, B, 1]
            Positive mask for zeroing out padded vectors between operations.

        Returns
        -------
        output : [T, B, 1]
            New encoded vectors.
        """
        num_vectors, batch_size, hidden_dim = tf.shape(encoded_vectors)[0], tf.shape(encoded_vectors)[1], tf.shape(encoded_vectors)[2]

        # Embed vectors again into particle space
        encoded_vectors = self.embedding(encoded_vectors, sequence_mask)

        # Add a "particle vector" which will store particle level data.
        particle_vector = tf.tile(self.particle_vector, [1, batch_size, 1])
        combined_vectors = tf.concat((particle_vector, encoded_vectors), axis=0)

        # Modify the padding mask to indicate that the particle vector is real.
        particle_padding_mask = tf.zeros((batch_size, 1), dtype=padding_mask.dtype)
        combined_padding_mask = tf.concat((particle_padding_mask, padding_mask), axis=1)

        # Modify the sequence mask to indicate that the particle vector is real.
        particle_sequence_mask = tf.ones((1, batch_size, 1), dtype=tf.bool)
        combined_sequence_mask = tf.concat((particle_sequence_mask, sequence_mask), axis=0)

        # Run all of the vectors through transformer encoder
        combined_vectors = self.encoder(combined_vectors, combined_padding_mask, combined_sequence_mask)
        particle_vector, encoded_vectors = combined_vectors[0], combined_vectors[1:]

        return encoded_vectors, particle_vector

# Ensure that the create_transformer and create_linear_stack functions are compatible with TensorFlow and have appropriate implementations.

