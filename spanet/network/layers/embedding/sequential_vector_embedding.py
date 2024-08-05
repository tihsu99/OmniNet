from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers

from spanet.network.layers.embedding_stack import EmbeddingStack
from spanet.options import Options


class SequentialVectorEmbedding(tf.keras.layers.Layer):
    def __init__(self, options: Options, input_dim: int):
        super(SequentialVectorEmbedding, self).__init__()

        self.input_dim = input_dim
        self.mask_sequence_vectors = options.mask_sequence_vectors
        self.embedding_stack = EmbeddingStack(options, input_dim)

    def call(self, vectors: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """ A stack of linear blocks with each layer doubling the hidden dimension

        Parameters
        ----------
        vectors : [B, T, I]
            Input vector data.
        mask : [B, T]
            Positive mask indicating that the jet is a real jet.

        Returns
        -------
        embeddings: [T, B, D]
            Hidden activations after embedding.
        padding_mask: [B, T]
            Negative mask indicating that a jet is padding for transformer.
        sequence_mask: [T, B, 1]
            Positive mask indicating jet is real.
        global_mask: [T]
            Negative mask for indicating a sequential variable or a global variable.
        """
        batch_size, max_vectors, input_dim = tf.shape(vectors)[0], tf.shape(vectors)[1], tf.shape(vectors)[2]

        # Create an negative mask for transformer layers.
        padding_mask = tf.logical_not(mask)

        # Create a positive mask indicating jet is real. This is for zeroing vectors at intermediate steps.
        # Alternatively, replace it with all ones if we are not masking (basically never).
        sequence_mask = tf.transpose(tf.reshape(mask, (batch_size, max_vectors, 1)), perm=[1, 0, 2])
        if not self.mask_sequence_vectors:
            sequence_mask = tf.ones_like(sequence_mask)

        # Create a negative mask indicating that all of the vectors that we embed will
        # be sequential variables and not global variables.
        global_mask = tf.ones((max_vectors,), dtype=tf.bool)

        # Reshape vector to have time axis first for transformer input.
        embeddings = tf.transpose(vectors, perm=[1, 0, 2])

        # Embed vectors into latent space.
        embeddings = self.embedding_stack([embeddings, sequence_mask])

        return embeddings, padding_mask, sequence_mask, global_mask



