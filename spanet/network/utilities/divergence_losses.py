import tensorflow as tf

@tf.function
def assignment_cross_entropy_loss(prediction: tf.Tensor, target_data: tf.Tensor, target_mask: tf.Tensor, gamma: float) -> tf.Tensor:
    batch_size = tf.shape(prediction)[0]
    prediction_shape = tf.shape(prediction)[1:]

    # Remove missing jets
    target_data = tf.clip_by_value(target_data, 0, tf.int32.max)

    # Find the unravelling shape required to flatten the target indices
    ravel_sizes = tf.math.cumprod(tf.reverse(prediction_shape, axis=[0]))
    ravel_sizes = ravel_sizes // ravel_sizes[0]
    ravel_sizes = tf.reverse(ravel_sizes, axis=[0])
    ravel_sizes = tf.cast(tf.expand_dims(ravel_sizes, 0), tf.int32)

    # Flatten the target and predicted data to be one dimensional
    ravel_target = tf.reduce_sum(target_data * ravel_sizes, axis=1)
    ravel_prediction = tf.reshape(prediction, (batch_size, -1))

    log_probability = tf.gather(ravel_prediction, tf.expand_dims(ravel_target, axis=-1), batch_dims=1)
    log_probability = tf.squeeze(log_probability, axis=-1)
    log_probability = tf.where(target_mask, log_probability, tf.zeros_like(log_probability))

    focal_scale = tf.pow(1 - tf.exp(log_probability), gamma)

    return -log_probability * focal_scale


@tf.function
def kl_divergence_old(p: tf.Tensor, log_p: tf.Tensor, log_q: tf.Tensor) -> tf.Tensor:
    sum_dim = list(range(1, p.ndim))
    return tf.reduce_sum(p * log_p - p * log_q, axis=sum_dim)


@tf.function
def kl_divergence(log_prediction: tf.Tensor, log_target: tf.Tensor) -> tf.Tensor:
    sum_dim = list(range(1, log_prediction.ndim))
    return tf.reduce_sum(tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)(log_target, log_prediction), axis=sum_dim)


@tf.function
def jensen_shannon_divergence(log_p: tf.Tensor, log_q: tf.Tensor) -> tf.Tensor:
    sum_dim = list(range(1, log_p.ndim))

    # log_m = log( (exp(log_p) + exp(log_q)) / 2 )
    log_m = tf.reduce_logsumexp(tf.stack([log_p, log_q], axis=0), axis=0) - tf.math.log(2.0)

    # TODO play around with gradient
    # log_m = tf.stop_gradient(log_m)
    log_p = tf.stop_gradient(log_p)
    log_q = tf.stop_gradient(log_q)

    kl_p = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)(log_m, log_p)
    kl_q = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)(log_m, log_q)

    return tf.reduce_sum(kl_p + kl_q, axis=sum_dim) / 2.0

