import tensorflow as tf

class Statistics:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

class LogGaussianRegression:
    @staticmethod
    def name():
        return "log_gaussian"

    @staticmethod
    def signed_log(x: tf.Tensor) -> tf.Tensor:
        return tf.asinh(x / 2.0)

    @staticmethod
    def inverse_signed_log(x: tf.Tensor) -> tf.Tensor:
        return 2.0 * tf.sinh(x)

    @staticmethod
    def statistics(data: tf.Tensor) -> Statistics:
        data = LogGaussianRegression.signed_log(data)

        mean = tf.reduce_mean(data, axis=None, keepdims=False)
        std = tf.sqrt(tf.reduce_mean(tf.square(data), axis=None, keepdims=False) - tf.square(mean))

        return Statistics(mean, std)

    @staticmethod
    def loss(predictions: tf.Tensor, targets: tf.Tensor, mean: tf.Tensor, std: tf.Tensor) -> tf.Tensor:
        return tf.square(predictions - targets)

    @staticmethod
    def normalize(data: tf.Tensor, mean: tf.Tensor, std: tf.Tensor) -> tf.Tensor:
        data = LogGaussianRegression.signed_log(data)
        return (data - mean) / std

    @staticmethod
    def denormalize(data: tf.Tensor, mean: tf.Tensor, std: tf.Tensor) -> tf.Tensor:
        data = std * data + mean
        return LogGaussianRegression.inverse_signed_log(data)

