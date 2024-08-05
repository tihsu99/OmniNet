import tensorflow as tf
from tensorflow.keras import layers


class IdentityMasking(tf.keras.layers.Layer):
    def call(self, values, sequence_mask):
        return values


class MultiplicativeMasking(tf.keras.layers.Layer):
    def call(self, values, sequence_mask):
        return values * tf.cast(sequence_mask, values.dtype)


class FillingMasking(tf.keras.layers.Layer):
    def call(self, values, sequence_mask):
        return tf.where(tf.logical_not(sequence_mask), tf.zeros_like(values), values)


def create_masking(masking: str) -> tf.keras.layers.Layer:
    masking = masking.lower().replace("_", "").replace(" ", "")

    if masking == "multiplicative":
        return MultiplicativeMasking()
    elif masking == "filling":
        return FillingMasking()
    else:
        return IdentityMasking()

