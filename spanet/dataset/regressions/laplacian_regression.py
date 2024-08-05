import tensorflow as tf

class Statistics:
    def __init__(self, median, deviation):
        self.median = median
        self.deviation = deviation

class LaplacianRegression:
    @staticmethod
    def name():
        # Returns the name of the regression type
        return "laplacian"

    @staticmethod
    def statistics(data: tf.Tensor) -> Statistics:
        # Remove NaN values from the data
        valid_data = tf.boolean_mask(data, ~tf.math.is_nan(data))

        # Calculate the median of the valid data
        median = tfp.stats.percentile(valid_data, 50.0)
        
        # Calculate the mean absolute deviation from the median
        deviation = tf.reduce_mean(tf.abs(valid_data - median))

        # Return a Statistics object with the median and deviation
        return Statistics(median, deviation)

    @staticmethod
    def loss(predictions: tf.Tensor, targets: tf.Tensor, mean: tf.Tensor, std: tf.Tensor) -> tf.Tensor:
        # Calculate the Laplacian loss
        return tf.abs(predictions - targets) / std

