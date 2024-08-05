from typing import Optional
import tensorflow as tf
from tensorflow.keras import layers

from spanet.options import Options
from spanet.network.layers.linear_block.normalizations import create_normalization
from spanet.network.layers.linear_block.activations import create_activation, create_dropout
from spanet.network.layers.linear_block.masking import create_masking


class NormalizationFirstBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            options: Options,
            input_dim: int,
            output_dim: int,
            skip_connection: bool = False,
            normalization: Optional[str] = None,
            dropout: Optional[float] = None):
        super(NormalizationFirstBlock, self).__init__()

        self.output_dim = output_dim
        self.skip_connection = skip_connection

        # Basic matrix multiplication layer as the base
        self.linear = layers.Dense(output_dim, use_bias=False)

        # Select non-linearity.
        self.activation = create_activation(options.linear_activation, output_dim)

        # Optional activation normalization. Either batch or layer norm.
        normalization = options.normalization if normalization is None else normalization
        self.normalization = create_normalization(normalization, output_dim)

        # Optional dropout for regularization.
        self.dropout = create_dropout(options.dropout if dropout is None else dropout)

        # Mask out padding values
        self.masking = create_masking(options.masking)

    def call(self, x: tf.Tensor, sequence_mask: tf.Tensor, residual: Optional[tf.Tensor] = None) -> tf.Tensor:
        """ Simple robust linear layer with non-linearity, normalization, and dropout.

        Parameters
        ----------
        x: [T, B, D]
            Input data.
        sequence_mask: [T, B, 1]
            Positive mask indicating if the jet is a true jet or not.

        Returns
        -------
        y: [T, B, D]
            Output data.
        """
        num_vectors, batch_size, dimensions = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        flat_shape = num_vectors * batch_size

        # Default residual connection is the input itself. Could be longer than one layer though.
        if residual is None:
            residual = x

        # Apply the basic matrix multiplication and batch norm operations.
        y = self.linear(x)
        y = self.normalization(y, sequence_mask)

        # Optionally add a skip-connection to the network to add residual information.
        if self.skip_connection:
            y = y + residual

        # Finally, apply a non-linearity.
        # Some activation functions require 1D input, so reshape before applying.
        y = tf.reshape(y, (flat_shape, self.output_dim))
        y = self.activation(y)
        y = tf.reshape(y, (num_vectors, batch_size, self.output_dim))

        # Apply a final dropout and mask out any padding vectors for cleanliness.
        return self.masking(self.dropout(y), sequence_mask)


class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, options: Options, input_dim: int, output_dim: int, skip_connection: bool = True):
        super(ResNetBlock, self).__init__()

        self.block_1 = NormalizationFirstBlock(options, input_dim, output_dim, skip_connection=False, dropout=0.0)
        self.block_2 = NormalizationFirstBlock(options, output_dim, output_dim, skip_connection=skip_connection)

    def call(self, x: tf.Tensor, sequence_mask: tf.Tensor) -> tf.Tensor:
        hidden = self.block_1(x, sequence_mask)
        return self.block_2(hidden, sequence_mask, residual=hidden)

