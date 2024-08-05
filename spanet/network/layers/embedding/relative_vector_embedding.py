from typing import Tuple
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

from spanet.network.layers.embedding_stack import EmbeddingStack
from spanet.network.layers.linear_block import create_linear_block
from spanet.options import Options


class RelativeVectorEmbedding(tf.keras.layers.Layer):
    """
    An implementation of a lorentz-invariant embeddings using attention.
    Inspired by Covariant Attention from Qiu Et al.

    A Holistic Approach to Predicting Top Quark Kinematic Properties with the Covariant Particle Transformer
    Shikai Qiu, Shuo Han, Xiangyang Ju, Benjamin Nachman, and Haichen Wang
    https://arxiv.org/pdf/2203.05687.pdf
    """
    def __init__(self, options: Options, input_dim: int):
        super(RelativeVectorEmbedding, self).__init__()

        self.input_dim = input_dim
        self.mask_sequence_vectors = options.mask_sequence_vectors

        self.shared_embedding_stack = EmbeddingStack(options, input_dim)
        self.shared_embedding_norm = layers.LayerNormalization()

        self.query_embedding = layers.Dense(options.hidden_dim, use_bias=False)
        self.key_embedding = layers.Dense(options.hidden_dim, use_bias=False)
        self.value_embedding = layers.Dense(options.hidden_dim, use_bias=False)
        self.attention_scale = np.sqrt(options.hidden_dim)

        self.output = create_linear_block(options, options.hidden_dim, options.hidden_dim, options.skip_connections)

    def call(self, vectors: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """ A stack of linear blocks with each layer doubling the hidden dimension

        Parameters
        ----------
        vectors : [B, T, T, I]
            Relative vector data.
        mask : [B, T, T]
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
        batch_size, max_vectors, _, input_dim = tf.shape(vectors)

        # Get the full pairwise mask
        square_mask = tf.transpose(mask, perm=[1, 2, 0])

        # Construct output linear masks from diagonal of relative mask.
        sequence_mask = tf.linalg.diag_part(square_mask)
        padding_mask = tf.logical_not(sequence_mask)
        sequence_mask = tf.expand_dims(tf.transpose(sequence_mask, perm=[1, 0]), axis=-1)

        # Flatten square mask to allow input to the linear layers
        square_mask = tf.reshape(square_mask, [max_vectors * max_vectors, batch_size])
        square_mask = tf.expand_dims(square_mask, axis=-1)

        # Perform main embedding on vectors to get them up to hidden dim.
        vectors = tf.transpose(tf.reshape(vectors, [batch_size, max_vectors * max_vectors, -1]), perm=[1, 0, 2])
        vectors = self.shared_embedding_stack([vectors, square_mask])
        vectors = self.shared_embedding_norm(vectors)
        vectors = tf.reshape(vectors, [max_vectors, max_vectors, batch_size, -1])

        # Compute attention components
        keys = self.key_embedding(vectors)
        values = self.value_embedding(vectors)
        queries = tf.transpose(tf.linalg.diag_part(vectors), perm=[1, 0, 2])
        queries = self.query_embedding(queries)

        # Attention mask for zeroing out the masked vectors.
        # Ensure the diagonal is True to avoid nans.
        attention_mask = tf.reshape(square_mask, [max_vectors, max_vectors, batch_size])
        attention_mask = tf.logical_or(attention_mask, tf.expand_dims(tf.eye(attention_mask.shape[0], dtype=attention_mask.dtype), axis=-1))

        # Compute attention softmax weights using dot-product
        attention_weights = tf.einsum('rbd,rcbd->rcb', queries, keys) / self.attention_scale
        attention_weights += tf.math.log(tf.cast(attention_mask, tf.float32))
        attention_weights = tf.nn.softmax(attention_weights, axis=1)

        embeddings = tf.reduce_sum(tf.expand_dims(attention_weights, axis=-1) * values, axis=1)
        embeddings = self.output([embeddings, sequence_mask])

        # Create a negative mask indicating that all of the vectors that we embed will
        # be sequential variables and not global variables.
        global_mask = tf.ones((max_vectors,), dtype=tf.bool)

        return embeddings, padding_mask, sequence_mask, global_mask

