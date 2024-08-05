from typing import Type
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from spanet.options import Options
from spanet.network.layers.branch_linear import BranchLinear
from spanet.dataset.regressions import Regression

class NormalizedBranchLinear(tf.keras.Model):
    def __init__(self, options: Options, num_layers: int, regression: Type[Regression], mean: tf.Tensor, std: tf.Tensor):
        super(NormalizedBranchLinear, self).__init__()

        self.hidden_dim = options.hidden_dim
        self.num_outputs = 1
        self.num_layers = num_layers

        self.regression = regression
        self.mean = tf.Variable(mean, trainable=False)
        self.std = tf.Variable(std, trainable=False)
        self.linear = BranchLinear(
            options,
            self.num_layers,
            self.num_outputs
        )

    def call(self, vector: tf.Tensor) -> tf.Tensor:
        """ Produce a single classification output for a sequence of vectors.

        Parameters
        ----------
        vector : [B, D]
            Hidden activations after central encoder.

        Returns
        -------
        classification: [B, O]
            Probability of this particle existing in the data.
        """
        normalized_output = self.linear(vector)
        return self.regression.denormalize(normalized_output, self.mean, self.std)


