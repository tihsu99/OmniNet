import tensorflow as tf
from tensorflow.keras import layers

from spanet.network.layers.linear_block.activations import create_activation, create_dropout, create_residual_connection
from spanet.network.layers.linear_block.normalizations import create_normalization
from spanet.network.layers.linear_block.masking import create_masking
from spanet.options import Options


class GLU(tf.keras.layers.Layer):
    def __init__(self, hidden_dim: int):
        super(GLU, self).__init__()
        self.linear_1 = layers.Dense(hidden_dim)
        self.linear_2 = layers.Dense(hidden_dim)
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.sigmoid(self.linear_1(x)) * self.linear_2(x)


class GatedBlock(tf.keras.layers.Layer):
    def __init__(self, options: Options, input_dim: int, output_dim: int, skip_connection: bool = False):
        super(GatedBlock, self).__init__()

        self.output_dim = output_dim
        self.skip_connection = skip_connection
        self.hidden_dim = int(round(options.transformer_dim_scale * input_dim))

        # The two fundamental linear layers for the gated network.
        self.linear_1 = layers.Dense(output_dim)
        self.linear_2 = layers.Dense(self.hidden_dim)

        # Select non-linearity.
        self.activation = create_activation(options.linear_activation, self.hidden_dim)

        # Create normalization layer for keeping values in good ranges.
        self.normalization = create_normalization(options.normalization, output_dim)

        # Optional dropout for regularization.
        self.dropout = create_dropout(options.dropout)

        # Possibly need a linear layer to create residual connection.
        self.residual = create_residual_connection(skip_connection, input_dim, output_dim)

        self.gate = GLU(output_dim)

        # Mask out padding values
        self.masking = create_masking(options.masking)

    def call(self, x: tf.Tensor, sequence_mask: tf.Tensor) -> tf.Tensor:
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
        max_jets, batch_size, dimensions = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        # Flatten the data and apply the basic matrix multiplication and non-linearity.
        x = tf.reshape(x, (max_jets * batch_size, dimensions))

        # Apply both linear layers with expansion in the middle.
        eta_2 = self.activation(self.linear_2(x))
        eta_1 = self.linear_1(eta_2)

        # Apply gating mechanism to possibly ignore this layer.
        output = self.dropout(eta_1)
        output = self.gate(output)

        # Optionally add a skip-connection to the network to add residual information.
        if self.skip_connection:
            output = output + self.residual(x)

        # Reshape the data back into the time-series and apply normalization.
        output = tf.reshape(output, (max_jets, batch_size, self.output_dim))
        output = self.normalization(output, sequence_mask)
        return self.masking(output, sequence_mask)

