from typing import List

import tensorflow as tf
from tensorflow.keras import layers

from spanet.options import Options
from spanet.network.layers.linear_block.basic_block import BasicBlock
from spanet.network.layers.linear_block import create_linear_block


class EmbeddingStack(tf.keras.layers.Layer):
    def __init__(self, options: Options, input_dim: int):
        super(EmbeddingStack, self).__init__()

        self.input_dim = input_dim
        self.embedding_layers = [self.create_embedding_layers(options, input_dim)]

    @staticmethod
    def create_embedding_layers(options: Options, input_dim: int) -> List[layers.Layer]:
        """ Create a stack of linear layers with increasing hidden dimensions.

        Each hidden layer will have double the dimensions as the previous, beginning with the
        size of the feature-space and ending with the hidden_dim specified in options.
        """

        embedding_layers = [create_linear_block(
            options,
            input_dim,
            options.initial_embedding_dim,
            options.initial_embedding_skip_connections
        )]
        current_embedding_dim = options.initial_embedding_dim

        # Keep doubling dimensions until we reach the desired latent dimension.
        for i in range(options.num_embedding_layers):
            next_embedding_dim = 2 * current_embedding_dim
            if next_embedding_dim >= options.hidden_dim:
                break

            embedding_layers.append(create_linear_block(
                options,
                current_embedding_dim,
                next_embedding_dim,
                options.initial_embedding_skip_connections
            ))
            current_embedding_dim = next_embedding_dim

        # Final embedding layer to ensure proper size on output.
        embedding_layers.append(create_linear_block(
            options,
            current_embedding_dim,
            options.hidden_dim,
            options.initial_embedding_skip_connections
        ))

        return embedding_layers

    def call(self, vectors: tf.Tensor, sequence_mask: tf.Tensor) -> tf.Tensor:
        """ Embed a sequence of vectors through a series of doubling linear layers.

        Parameters
        ----------
        vectors: [T, B, I]
            Original vectors to embed.
        sequence_mask: [T, B, 1]
            Positive mask indicating if the vector is real or padding.

        Returns
        -------
        embeddings: [T, B, D]
            Output embeddings.
        """

        embeddings = vectors

        for layer in self.embedding_layers:
            embeddings = layer(embeddings, sequence_mask)

        return embeddings

# Ensure that the create_linear_block function is compatible with TensorFlow and has appropriate implementations.

