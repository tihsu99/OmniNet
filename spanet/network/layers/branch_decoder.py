from typing import Tuple, List
from opt_einsum import contract_expression

import tensorflow as tf
from tensorflow.keras import layers

from spanet.options import Options
from spanet.dataset.types import Symmetries
from spanet.network.utilities import masked_log_softmax
from spanet.network.layers.stacked_encoder import StackedEncoder
from spanet.network.layers.branch_linear import BranchLinear
from spanet.network.symmetric_attention import SymmetricAttentionSplit, SymmetricAttentionFull


class BranchDecoder(tf.keras.layers.Layer):
    # noinspection SpellCheckingInspection
    WEIGHTS_INDEX_NAMES = "ijklmn"
    DEFAULT_JET_COUNT = 16

    def __init__(
        self,
        options: Options,
        particle_name: str,
        product_names: List[str],
        product_symmetries: Symmetries,
        softmax_output: bool = True
    ):
        super(BranchDecoder, self).__init__()

        self.degree = product_symmetries.degree
        self.particle_name = particle_name
        self.product_names = product_names
        self.softmax_output = softmax_output
        self.combinatorial_scale = options.combinatorial_scale

        # Each branch has a personal encoder stack to extract particle-level data
        self.encoder = StackedEncoder(
            options,
            options.num_branch_embedding_layers,
            options.num_branch_encoder_layers
        )

        # Symmetric attention to create the output distribution
        attention_layer = SymmetricAttentionSplit if options.split_symmetric_attention else SymmetricAttentionFull
        self.attention = attention_layer(options, self.degree, product_symmetries.permutations)

        # Optional output predicting if the particle was present or not
        self.detection_classifier = BranchLinear(options, options.num_detector_layers)

        self.num_targets = len(self.attention.permutation_group)
        self.permutation_indices = self.attention.permutation_indices

        self.padding_mask_operation = self.create_padding_mask_operation(options.batch_size)
        self.diagonal_mask_operation = self.create_diagonal_mask_operation()
        self.diagonal_masks = {}

    def create_padding_mask_operation(self, batch_size: int):
        weights_index_names = self.WEIGHTS_INDEX_NAMES[:self.degree]
        operands = ','.join(map(lambda x: 'b' + x, weights_index_names))
        expression = f"{operands}->b{weights_index_names}"
        return expression

    def create_diagonal_mask_operation(self):
        weights_index_names = self.WEIGHTS_INDEX_NAMES[:self.degree]
        operands = ','.join(map(lambda x: 'b' + x, weights_index_names))
        expression = f"{operands}->{weights_index_names}"
        return expression

    def create_output_mask(self, output: tf.Tensor, sequence_mask: tf.Tensor) -> tf.Tensor:
        num_jets = tf.shape(output)[1]

        # batch_sequence_mask: [B, T, 1] Positive mask indicating jet is real.
        batch_sequence_mask = tf.transpose(sequence_mask, perm=[1, 0, 2])

        # =========================================================================================
        # Padding mask
        # =========================================================================================
        padding_mask_operands = [tf.squeeze(batch_sequence_mask, axis=-1)] * self.degree
        padding_mask = tf.einsum(self.padding_mask_operation, *padding_mask_operands)
        padding_mask = tf.cast(padding_mask, dtype=tf.bool)

        # =========================================================================================
        # Diagonal mask
        # =========================================================================================
        try:
            diagonal_mask = self.diagonal_masks[(num_jets.numpy(), output.device)]
        except KeyError:
            identity = 1 - tf.eye(num_jets, dtype=output.dtype)

            diagonal_mask_operands = [identity] * self.degree
            diagonal_mask = tf.einsum(self.diagonal_mask_operation, *diagonal_mask_operands)
            diagonal_mask = tf.expand_dims(diagonal_mask, axis=0) < (num_jets + 1 - self.degree)
            self.diagonal_masks[(num_jets.numpy(), output.device)] = diagonal_mask

        return tf.logical_and(padding_mask, diagonal_mask)

    def call(
            self,
            event_vectors: tf.Tensor,
            padding_mask: tf.Tensor,
            sequence_mask: tf.Tensor,
            global_mask: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """ Create a distribution over jets for a given particle and a probability of its existence.

        Parameters
        ----------
        event_vectors : [T, B, D]
            Hidden activations after central encoder.
        padding_mask : [B, T]
            Negative mask for transformer input.
        sequence_mask : [T, B, 1]
            Positive mask for zeroing out padded vectors between operations.

        Returns
        -------
        selection : [TS, TS, ...]
            Distribution over sequential vectors for the target vectors.
        classification: [B]
            Probability of this particle existing in the data.
        """

        # ------------------------------------------------------
        # Apply the branch's independent encoder to each vector.
        # particle_vectors : [T, B, D]
        # ------------------------------------------------------
        encoded_vectors, particle_vector = self.encoder(event_vectors, padding_mask, sequence_mask)

        # -----------------------------------------------
        # Run the encoded vectors through the classifier.
        # detection: [B, 1]
        # -----------------------------------------------
        detection = tf.squeeze(self.detection_classifier(particle_vector), axis=-1)

        # --------------------------------------------------------
        # Extract sequential vectors only for the assignment step.
        # sequential_particle_vectors : [TS, B, D]
        # sequential_padding_mask : [B, TS]
        # sequential_sequence_mask : [TS, B, 1]
        # --------------------------------------------------------
        sequential_particle_vectors = tf.boolean_mask(encoded_vectors, global_mask)
        sequential_padding_mask = tf.boolean_mask(padding_mask, global_mask, axis=1)
        sequential_sequence_mask = tf.boolean_mask(sequence_mask, global_mask)

        # --------------------------------------------------------------------
        # Create the vector distribution logits and the correctly shaped mask.
        # assignment : [TS, TS, ...]
        # assignment_mask : [TS, TS, ...]
        # --------------------------------------------------------------------
        assignment, daughter_vectors = self.attention(
            sequential_particle_vectors,
            sequential_padding_mask,
            sequential_sequence_mask
        )

        assignment_mask = self.create_output_mask(assignment, sequential_sequence_mask)

        # ---------------------------------------------------------------------------
        # Need to reshape output to make softmax-calculation easier.
        # We transform the mask and output into a flat representation.
        # Afterwards, we apply a masked log-softmax to create the final distribution.
        # output : [TS, TS, ...]
        # mask : [TS, TS, ...]
        # ---------------------------------------------------------------------------
        if self.softmax_output:
            original_shape = tf.shape(assignment)
            batch_size = original_shape[0]

            assignment = tf.reshape(assignment, (batch_size, -1))
            assignment_mask = tf.reshape(assignment_mask, (batch_size, -1))

            assignment = masked_log_softmax(assignment, assignment_mask)
            assignment = tf.reshape(assignment, original_shape)

        return assignment, detection, assignment_mask, particle_vector, daughter_vectors

