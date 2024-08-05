from typing import List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from spanet.options import Options
from spanet.network.layers.branch_linear import BranchLinear

class MultiOutputBranchLinear(tf.keras.Model):
    def __init__(self, options, num_layers: int, num_outputs: tf.Tensor):
        super(MultiOutputBranchLinear, self).__init__()

        self.hidden_dim = options.hidden_dim
        self.num_layers = num_layers

        self.shared_layers = BranchLinear(
            options,
            max(self.num_layers - 1, 1),
            self.hidden_dim,
            batch_norm=False
        )

        self.output_layers = [
            BranchLinear(options, 1, int(output_dim))
            for output_dim in num_outputs
        ]

    def call(self, vector: tf.Tensor) -> List[tf.Tensor]:
        vector = self.shared_layers(vector)

        return [output_layer(vector) for output_layer in self.output_layers]


