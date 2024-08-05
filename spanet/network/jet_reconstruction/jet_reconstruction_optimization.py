import tensorflow as tf
import numpy as np
from typing import Tuple, List


from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork
from spanet.options import Options


class JetReconstructionOptimization(JetReconstructionNetwork):
    def __init__(self, options: Options):
        super(JetReconstructionOptimization, self).__init__(options)

        self.num_losses = (
            (self.options.assignment_loss_scale > 0) * len(self.training_dataset.assignments) +
            (self.options.detection_loss_scale > 0) * len(self.training_dataset.assignments) +
            (self.options.regression_loss_scale > 0) * len(self.training_dataset.regressions) +
            (self.options.classification_loss_scale > 0) * len(self.training_dataset.classifications) +
            (self.options.kl_loss_scale > 0)
        )

        self.loss_weight_logits = tf.Variable(tf.zeros(self.num_losses), trainable=True)
        self.loss_weight_alpha = 0.0

    def jacobian(self, outputs: tf.Tensor, inputs: List[tf.Tensor]) -> List[tf.Tensor]:
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            output_vals = tf.reshape(outputs, (-1,))

            jacobians = []
            for i in range(len(output_vals)):
                jacobians.append(tape.gradient(output_vals[i], inputs))
        return jacobians

    def optimizer_zero_grad(self, optimizer: tf.optimizers.Optimizer):
        optimizer.zero_grad()

    def backward(self, loss: tf.Tensor, tape: tf.GradientTape, optimizer: tf.optimizers.Optimizer) -> None:
        if not self.options.balance_losses:
            grads = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))
            return

        loss_weights = tf.nn.softmax(self.loss_weight_logits)
        free_weights = loss_weights.numpy()

        parameters = [v for v in self.trainable_variables if v.trainable]
        jacobians = self.jacobian(loss, parameters)
        GW = []

        for parameter, jacobian in zip(parameters, jacobians):
            weights = tf.reshape(free_weights, (-1,) + (1,) * (len(jacobian.shape) - 1))
            parameter_grad = tf.reduce_sum(weights * jacobian, axis=0)
            optimizer.apply_gradients([(parameter_grad, parameter)])
            GW.append(tf.reshape(jacobian, (self.num_losses, -1)))

        GW = tf.concat(GW, axis=-1)
        GW = loss_weights[:, tf.newaxis] * GW
        GW = tf.sqrt(tf.reduce_sum(tf.square(GW), axis=-1))

        GW_bar = tf.reduce_mean(GW)

        r = (loss / tf.reduce_mean(loss)) ** self.loss_weight_alpha
        L_grad = tf.reduce_sum(tf.abs(GW - GW_bar * r))
        loss_weight_grad = tape.gradient(L_grad, self.loss_weight_logits)
        optimizer.apply_gradients([(loss_weight_grad, self.loss_weight_logits)])

        for i, a in enumerate(free_weights):
            tf.summary.scalar(f"weights/{i}", a)

