from typing import List
import tensorflow as tf


@tf.function
def contract_4d(weights: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    factor = tf.sqrt(tf.cast(tf.shape(weights)[0], tf.float32))
    y = tf.einsum('ijkl,bxi->jklbx', weights, x) / factor
    y = tf.einsum('jklbx,byj->klbxy', y, x) / factor
    y = tf.einsum('klbxy,bzk->lbxyz', y, x) / factor
    y = tf.einsum('lbxyz,bwl->bxyzw', y, x) / factor

    return y


@tf.function
def contract_3d(weights: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    factor = tf.sqrt(tf.cast(tf.shape(weights)[0], tf.float32))
    y = tf.einsum('ijk,bxi->jkbx', weights, x) / factor
    y = tf.einsum('jkbx,byj->kbxy', y, x) / factor
    y = tf.einsum('kbxy,bzk->bxyz', y, x) / factor
    return y


@tf.function
def contract_2d(weights: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    factor = tf.sqrt(tf.cast(tf.shape(weights)[0], tf.float32))
    y = tf.einsum('ij,bxi->jbx', weights, x) / factor
    y = tf.einsum('jbx,byj->bxy', y, x) / factor
    return y


@tf.function
def contract_1d(weights: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    factor = tf.sqrt(tf.cast(tf.shape(weights)[0], tf.float32))
    y = tf.einsum('i,bxi->bx', weights, x) / factor
    return y


@tf.function
def contract_linear_form(weights: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
    if tf.rank(weights) == 4:
        return contract_4d(weights, x)
    elif tf.rank(weights) == 3:
        return contract_3d(weights, x)
    elif tf.rank(weights) == 2:
        return contract_2d(weights, x)
    else:
        return contract_1d(weights, x)


@tf.function
def symmetric_tensor(weights: tf.Tensor, permutation_group: List[List[int]]) -> tf.Tensor:
    symmetric_weights = weights
    for sigma in permutation_group:
        symmetric_weights = symmetric_weights + tf.transpose(weights, perm=sigma)
    return symmetric_weights / (len(permutation_group) + 1)


def create_symmetric_function(permutation_group: List[List[int]]):
    code = [
        "def symmetrize_tensor(weights):",
        "    symmetric_weights = weights"
        "    "
    ]

    for sigma in permutation_group:
        code.append(f"    symmetric_weights = symmetric_weights + tf.transpose(weights, perm={sigma})")

    code.append(f"    return symmetric_weights / {len(permutation_group) + 1}")
    code = "\n".join(code)

    environment = globals().copy()
    exec(code, environment)

    return environment["symmetrize_tensor"]


@tf.function
def batch_symmetric_tensor(inputs: tf.Tensor, permutation_group: List[List[int]]) -> tf.Tensor:
    symmetric_outputs = inputs
    for sigma in permutation_group:
        for i in range(inputs.shape[0]):
            symmetric_outputs = tf.tensor_scatter_nd_add(symmetric_outputs, [[i]], [tf.transpose(inputs[i], perm=sigma)])
    return symmetric_outputs / (len(permutation_group) + 1)

