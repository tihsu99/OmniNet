import tensorflow as tf
from tensorflow.keras import layers

def create_activation(activation: str, input_dim: int) -> layers.Layer:
    activation = activation.lower().replace("_", "").replace(" ", "")

    if activation == "relu":
        return layers.ReLU()
    elif activation == "prelu":
        return layers.PReLU(shared_axes=[-1])  # PReLU doesn't use input_dim in TensorFlow
    elif activation == "elu":
        return layers.ELU()
    elif activation == "celu":
        return layers.CELU()
    elif activation == "gelu":
        return layers.GELU()
    else:
        return layers.Lambda(lambda x: x)  # Identity

def create_dropout(dropout: float) -> layers.Layer:
    if dropout > 0:
        return layers.Dropout(dropout)
    else:
        return layers.Lambda(lambda x: x)  # Identity

class ZeroModule(tf.keras.layers.Layer):
    def call(self, vectors):
        return tf.zeros_like(vectors)

def create_residual_connection(skip_connection: bool, input_dim: int, output_dim: int) -> layers.Layer:
    if input_dim == output_dim or not skip_connection:
        return layers.Lambda(lambda x: x)  # Identity

    return layers.Dense(output_dim)

