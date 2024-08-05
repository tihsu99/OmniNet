import tensorflow as tf

def masked_log_softmax(
        vector: tf.Tensor,
        mask: tf.Tensor,
        dim: int = -1
) -> tf.Tensor:
    """
    Another alternative implementation of the masked log-softmax, this time doing a pure
    mask (setting invalid values to -inf) but also preventing any gradient from flowing
    at all to masked values!
    """

    if mask is not None:
        # Create a -inf with the correct device and datatype
        fill_value = tf.math.log(tf.zeros_like(vector))

        # Replace all masked entries in the output with the gradient-less -inf
        vector = tf.where(mask, vector, fill_value)

    return tf.nn.log_softmax(vector, axis=dim)


def masked_softmax(
        vector: tf.Tensor,
        mask: tf.Tensor,
        dim: int = -1,
        memory_efficient: bool = False,
) -> tf.Tensor:
    return tf.exp(masked_log_softmax(vector, mask, dim))

