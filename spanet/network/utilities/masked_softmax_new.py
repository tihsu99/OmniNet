"""
Adapted from AllenNLP
https://github.com/allenai/allennlp

"""
import tensorflow as tf

def tiny_value_of_dtype(dtype: tf.DType) -> float:
    """
    Returns a moderately tiny value for a given TensorFlow data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not tf.as_dtype(dtype).is_floating:
        raise TypeError("Only supports floating point dtypes.")
    if dtype in [tf.float32, tf.float64]:
        return 1e-13
    elif dtype == tf.float16:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))

def info_value_of_dtype(dtype: tf.DType):
    """
    Returns the `finfo` or `iinfo` object of a given TensorFlow data type. Does not allow tf.bool.
    """
    if dtype == tf.bool:
        raise TypeError("Does not support tf.bool")
    elif tf.as_dtype(dtype).is_floating:
        return tf.experimental.numpy.finfo(dtype)
    else:
        return tf.experimental.numpy.iinfo(dtype)

def min_value_of_dtype(dtype: tf.DType) -> float:
    """
    Returns the minimum value of a given TensorFlow data type. Does not allow tf.bool.
    """
    return info_value_of_dtype(dtype).min

def max_value_of_dtype(dtype: tf.DType) -> float:
    """
    Returns the maximum value of a given TensorFlow data type. Does not allow tf.bool.
    """
    return info_value_of_dtype(dtype).max

def masked_softmax(
        vector: tf.Tensor,
        mask: tf.Tensor,
        dim: int = -1,
        memory_efficient: bool = False,
) -> tf.Tensor:
    """
    `tf.nn.softmax(vector)` does not work if some elements of `vector` should be
    masked. This performs a softmax on just the non-masked portions of `vector`. Passing
    `None` in for the mask is also acceptable; you'll just get a regular softmax.
    `vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
    broadcastable to `vector's` shape. If `mask` has fewer dimensions than `vector`, we will
    unsqueeze on dimension 1 until they match. If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If `memory_efficient` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and `memory_efficient` is false, this function
    returns an array of `0.0`. This behavior may cause `NaN` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if `memory_efficient` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = tf.nn.softmax(vector, axis=dim)
    else:
        while mask.shape.ndims < vector.shape.ndims:
            mask = tf.expand_dims(mask, axis=1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = tf.nn.softmax(vector * tf.cast(mask, vector.dtype), axis=dim)
            result = result * tf.cast(mask, vector.dtype)
            result = result / (
                    tf.reduce_sum(result, axis=dim, keepdims=True) + tiny_value_of_dtype(result.dtype)
            )
        else:
            masked_vector = tf.where(mask, vector, min_value_of_dtype(vector.dtype) * tf.ones_like(vector))
            result = tf.nn.softmax(masked_vector, axis=dim)
    return result

def masked_log_softmax(vector: tf.Tensor, mask: tf.Tensor, dim: int = -1) -> tf.Tensor:
    """
    `tf.nn.log_softmax(vector)` does not work if some elements of `vector` should be
    masked. This performs a log_softmax on just the non-masked portions of `vector`. Passing
    `None` in for the mask is also acceptable; you'll just get a regular log_softmax.
    `vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
    broadcastable to `vector's` shape. If `mask` has fewer dimensions than `vector`, we will
    unsqueeze on dimension 1 until they match. If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not `nan`. You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter. Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you `nans`.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up. But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        while mask.shape.ndims < vector.shape.ndims:
            mask = tf.expand_dims(mask, axis=1)

        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked. We need a very small value instead of a
        # zero in the mask for these cases.
        vector = vector + tf.math.log(tf.cast(mask, vector.dtype) + tiny_value_of_dtype(vector.dtype))
    return tf.nn.log_softmax(vector, axis=dim)

