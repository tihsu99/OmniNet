import tensorflow as tf

class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim: int):
        super(PositionEmbedding, self).__init__()
        self.position_embedding = tf.Variable(tf.random.normal([1, 1, embedding_dim]), trainable=True)

    def call(self, current_embeddings: tf.Tensor) -> tf.Tensor:
        num_vectors, batch_size, input_dim = tf.shape(current_embeddings)

        position_embedding = tf.tile(self.position_embedding, [num_vectors, batch_size, 1])
        return tf.concat([current_embeddings, position_embedding], axis=2)

