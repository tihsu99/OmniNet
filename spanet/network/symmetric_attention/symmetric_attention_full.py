from itertools import islice
from typing import List, Tuple
from opt_einsum import contract_expression

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from spanet.network.symmetric_attention.symmetric_attention_base import SymmetricAttentionBase
from spanet.network.utilities.linear_form import contract_linear_form, symmetric_tensor
from spanet.options import Options


# noinspection SpellCheckingInspection
class SymmetricAttentionFull(SymmetricAttentionBase):
    def __init__(self,
                 options: Options,
                 degree: int,
                 permutation_indices: List[Tuple[int, ...]] = None,
                 attention_dim: int = None) -> None:
        super(SymmetricAttentionFull, self).__init__(
            options,
            degree,
            permutation_indices,
            attention_dim
        )

        self.weights_shape = [self.features] * degree

        self.weights = tf.Variable(
            initial_value=tf.random.normal(self.weights_shape),
            trainable=True,
            dtype=tf.float32,
            name='weights'
        )

        self.output_operation = self.make_contraction()

        self.reset_parameters()

    def make_contraction(self):
        weights_index_names = np.array(list(self.WEIGHTS_INDEX_NAMES))
        input_index_names = np.array(list(self.INPUT_INDEX_NAMES))
        batch_index_name = 'b'

        operations = map(lambda x: batch_index_name + ''.join(x), zip(input_index_names, weights_index_names))
        operations = ','.join(islice(operations, self.degree))

        operand = f",{''.join(weights_index_names[:self.degree])}"
        result = f"->b{''.join(input_index_names[:self.degree])}"

        expression = operations + operand + result
        shapes = [(self.batch_size, self.DEFAULT_JET_COUNT, self.features)] * self.degree
        shapes.append((self.features,) * self.degree)
        return contract_expression(expression, *shapes, optimize='optimal')

    def reset_parameters(self) -> None:
        # Using Xavier uniform initializer to initialize weights
        initializer = tf.keras.initializers.GlorotUniform()
        self.weights.assign(initializer(self.weights.shape))

    def call(self, x: tf.Tensor, padding_mask: tf.Tensor, sequence_mask: tf.Tensor) -> tf.Tensor:
        """ Perform symmetric attention on the hidden vectors and produce the output logits.

        This is the full version which creates the N^D tensor and perfoms a general linear form calculation.

        Parameters
        ----------
        x : [T, B, D]
            Hidden activations after branch encoders.
        padding_mask: [B, T]
            Negative mask indicating that a jet is padding for transformer.
        sequence_mask: [T, B, 1]
            Positive mask indicating jet is real.

        Returns
        -------
        output : [T, T, ...]
            Prediction logits for this particle.
        """

        x = tf.transpose(x, perm=[1, 0, 2])

        # Enforce that symmetries of the particle permutation group
        symmetric_weights = symmetric_tensor(self.weights, self.no_identity_permutations)

        # Perform the generalized matrix multiplication operation.
        output = contract_linear_form(symmetric_weights, x)

        output = symmetric_tensor(output, self.batch_no_identity_permutations)

        return output

