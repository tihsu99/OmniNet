import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import Tensor

from spanet.network.layers.linear_block.activations import create_activation, create_dropout, create_residual_connection
from spanet.network.layers.linear_block.normalizations import create_normalization
from spanet.network.layers.linear_block.masking import create_masking
from spanet.options import Options


class GRUGate(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, gate_initialization: float = 2.0):
        super(GRUGate, self).__init__()

        self.linear_W_r = layers.Dense(hidden_dim, use_bias=True)
        self.linear_U_r = layers.Dense(hidden_dim, use_bias=False)

        self.linear_W_z = layers.Dense(hidden_dim, use_bias=True)
        self.linear_U_z = layers.Dense(hidden_dim, use_bias=False)

        self.linear_W_g = layers.Dense(hidden_dim, use_bias=True)
        self.linear_U_g = layers.Dense(hidden_dim, use_bias=False)

        self.gate_bias = tf.Variable(tf.ones(hidden_dim) * gate_initialization, trainable=True)

    def call(self, vectors: Tensor, residual: Tensor) -> Tensor:
        r = tf.sigmoid(self.linear_W_r(vectors) + self.linear_U_r(residual))
        z = tf.sigmoid(self.linear_W_z(vectors) + self.linear_U_z(residual) - self.gate_bias)
        h = tf.tanh(self.linear_W_g(vectors) + self.linear_U_g(r * residual))

        return (1 - z) * residual + z * h


class GRUBlock(tf.keras.layers.Layer):
    def __init__(self, options: Options, input_dim: int, output_dim: int, skip_connection: bool = False):
        super(GRUBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.skip_connection = skip_connection
        self.hidden_dim = int(round(options.transformer_dim_scale * input_dim))

        # Create normalization layer for keeping values in good ranges.
        self.normalization = create_normalization(options.normalization, input_dim)

        # The primary linear layers applied before the gate
        self.linear_1 = tf.keras.Sequential([
            layers.Dense(self.hidden_dim),
            create_activation(options.linear_activation, self.hidden_dim),
            create_dropout(options.dropout)
        ])

        self.linear_2 = tf.keras.Sequential([
            layers.Dense(output_dim),
            create_activation(options.linear_activation, output_dim),
            create_dropout(options.dropout)
        ])

        # GRU layer to gate and project back to output.
        self.gru = GRUGate(output_dim)

        # Possibly need a linear layer to create residual connection.
        self.residual = create_residual_connection(skip_connection, input_dim, output_dim)

        # Mask out padding values
        self.masking = create_masking(options.masking)

    def call(self, x: Tensor, sequence_mask: Tensor) -> Tensor:
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
        timesteps, batch_size, input_dim = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        # Apply normalization first for this type of linear block.
        output = self.normalization(x, sequence_mask)

        # Flatten the data and apply the basic matrix multiplication and non-linearity.
        output = tf.reshape(output, (timesteps * batch_size, self.input_dim))

        # Apply linear layer with expansion in the middle.
        output = self.linear_1(output)
        output = self.linear_2(output)

        # Reshape the data back into the time-series and apply normalization.
        output = tf.reshape(output, (timesteps, batch_size, self.output_dim))

        # Apply gating mechanism and skip connection using the GRU mechanism.
        if self.skip_connection:
            output = self.gru(output, self.residual(x))

        return self.masking(output, sequence_mask)

