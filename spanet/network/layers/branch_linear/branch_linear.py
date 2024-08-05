import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from spanet.options import Options
from spanet.network.layers.linear_stack import create_linear_stack

class BranchLinear(tf.keras.Model):
    def __init__(self, options, num_layers: int, num_outputs: int = 1, batch_norm: bool = True):
        super(BranchLinear, self).__init__()
        
        self.hidden_dim = options.hidden_dim
        self.num_layers = num_layers
        self.hidden_layers = self.create_linear_stack(options, num_layers, self.hidden_dim, options.skip_connections)

        if batch_norm:
            self.output_norm = layers.BatchNormalization()
        else:
            self.output_norm = tf.keras.layers.Lambda(lambda x: x)
        
        self.output_layer = layers.Dense(num_outputs)

    def create_linear_stack(self, options, num_layers, hidden_dim, skip_connections):
        stack = []
        for _ in range(num_layers):
            stack.append(layers.Dense(hidden_dim, activation='relu'))
            if skip_connections:
                stack.append(layers.Add())
        return keras.Sequential(stack)

    def call(self, single_vector: tf.Tensor) -> tf.Tensor:
        batch_size, input_dim = tf.shape(single_vector)[0], tf.shape(single_vector)[1]

        # Convert our single vector into a sequence of length 1.
        single_vector = tf.reshape(single_vector, (1, batch_size, input_dim))

        # Run through hidden layer stack first, and then take the first timestep out.
        hidden = self.hidden_layers(single_vector)
        hidden = tf.reshape(hidden, (batch_size, self.hidden_dim))

        # Run through the linear layer stack and output the result
        classification = self.output_layer(self.output_norm(hidden))

        return classification
