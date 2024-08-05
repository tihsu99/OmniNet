from typing import Dict
from collections import OrderedDict

import tensorflow as tf
from tensorflow.keras import layers

from spanet.options import Options
from spanet.dataset.regressions import regression_class
from spanet.network.layers.branch_linear import NormalizedBranchLinear
from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset


class RegressionDecoder(tf.keras.layers.Layer):
    def __init__(self, options: Options, training_dataset: JetReconstructionDataset):
        super(RegressionDecoder, self).__init__()

        # Compute training dataset statistics to fix the final weight and bias.
        means, stds = training_dataset.compute_regression_statistics()

        # A unique linear decoder for each possible regression.
        networks = OrderedDict()
        for name, data in training_dataset.regressions.items():
            if data is None:
                continue

            networks[name] = NormalizedBranchLinear(
                options,
                options.num_regression_layers,
                regression_class(training_dataset.regression_types[name]),
                means[name],
                stds[name]
            )

        self.networks = {name: network for name, network in networks.items()}

    def call(self, vectors: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        # vectors: Dict with mapping name -> [B, D]
        # outputs: Dict with mapping name -> [B, O_name]

        return {
            key: tf.reshape(network(vectors['/'.join(key.split('/')[:-1])]), [-1])
            for key, network in self.networks.items()
        }

# Ensure that the NormalizedBranchLinear class is translated to TensorFlow as well.

