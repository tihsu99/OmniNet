import tensorflow as tf

from spanet.dataset.regressions.base_regression import Regression, Statistics


class GaussianRegression(Regression):
    @staticmethod
    def name():
        return "gaussian"

    @staticmethod
    def statistics(data: tf.Tensor) -> Statistics:
        mean = tf.reduce_mean(tf.boolean_mask(data, ~tf.math.is_nan(data)))
        std = tf.sqrt(tf.reduce_mean(tf.square(tf.boolean_mask(data, ~tf.math.is_nan(data)))) - tf.square(mean))

        return Statistics(mean, std)

    @staticmethod
    def loss(predictions: tf.Tensor, targets: tf.Tensor, mean: tf.Tensor, std: tf.Tensor) -> tf.Tensor:
        return tf.square((predictions - targets) / std)

