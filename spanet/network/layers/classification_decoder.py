from typing import Dict, List
from collections import OrderedDict

import tensorflow as tf
from tensorflow.keras import layers

from spanet.options import Options
from spanet.network.layers.branch_linear import MultiOutputBranchLinear, BranchLinear
from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset


class ClassificationDecoder(tf.keras.layers.Layer):
    def __init__(self, options: Options, training_dataset: JetReconstructionDataset):
        super(ClassificationDecoder, self).__init__()

        # Compute training dataset statistics to fix the final weight and bias.
        counts = training_dataset.compute_classification_class_counts()

        # A unique linear decoder for each possible regression.
        networks = OrderedDict()
        for name, data in training_dataset.classifications.items():
            if data is None:
                continue

            networks[name] = BranchLinear(
                options,
                options.num_classification_layers,
                counts[name]
            )

            # If you need to use MultiOutputBranchLinear, uncomment the following:
            # networks[name] = MultiOutputBranchLinear(
            #     options,
            #     options.num_classification_layers,
            #     counts[name]
            # )

        self.networks = {name: network for name, network in networks.items()}

    def call(self, vectors: Dict[str, tf.Tensor]) -> Dict[str, List[tf.Tensor]]:
        # vectors: Dict with mapping name -> [B, D]
        # outputs: Dict with mapping name -> [B, O_name]

        return {
            key: network(vectors['/'.join(key.split('/')[:-1])])
            for key, network in self.networks.items()
        }

# Ensure that the BranchLinear and MultiOutputBranchLinear classes are translated to TensorFlow as well.

