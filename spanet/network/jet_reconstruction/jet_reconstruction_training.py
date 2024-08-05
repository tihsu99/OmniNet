from typing import Tuple, Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, metrics

from spanet.options import Options
from spanet.dataset.types import Batch, Source, AssignmentTargets
from spanet.dataset.regressions import regression_loss
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork
from spanet.network.utilities.divergence_losses import assignment_cross_entropy_loss, jensen_shannon_divergence


def numpy_tensor_array(tensor_list):
    output = np.empty(len(tensor_list), dtype=object)
    output[:] = tensor_list

    return output


class JetReconstructionTraining(JetReconstructionNetwork):
    def __init__(self, options: Options, torch_script: bool = False):
        super(JetReconstructionTraining, self).__init__(options, torch_script)

        self.log_clip = tf.math.log(10 * tf.keras.backend.epsilon())

        self.event_particle_names = list(self.training_dataset.event_info.product_particles.keys())
        self.product_particle_names = {
            particle: self.training_dataset.event_info.product_particles[particle][0]
            for particle in self.event_particle_names
        }

    def particle_symmetric_loss(self, assignment: tf.Tensor, detection: tf.Tensor, target: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        assignment_loss = assignment_cross_entropy_loss(assignment, target, mask, self.options.focal_gamma)
        detection_loss = losses.binary_crossentropy(mask, detection, from_logits=True)

        return tf.stack((
            self.options.assignment_loss_scale * assignment_loss,
            self.options.detection_loss_scale * detection_loss
        ))

    def compute_symmetric_losses(self, assignments: List[tf.Tensor], detections: List[tf.Tensor], targets):
        symmetric_losses = []

        for permutation in self.event_permutation_tensor.numpy():
            current_permutation_loss = tuple(
                self.particle_symmetric_loss(assignment, detection, target, mask)
                for assignment, detection, (target, mask)
                in zip(assignments, detections, targets[permutation])
            )
            symmetric_losses.append(tf.stack(current_permutation_loss))

        return tf.stack(symmetric_losses)

    def combine_symmetric_losses(self, symmetric_losses: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        total_symmetric_loss = tf.reduce_sum(symmetric_losses, axis=(1, 2))
        index = tf.argmin(total_symmetric_loss, axis=0)

        combined_loss = tf.gather(symmetric_losses, index, axis=0)

        if self.options.combine_pair_loss.lower() == "mean":
            combined_loss = tf.reduce_mean(symmetric_losses, axis=0)

        if self.options.combine_pair_loss.lower() == "softmin":
            weights = tf.nn.softmax(-total_symmetric_loss)
            weights = tf.expand_dims(tf.expand_dims(weights, axis=1), axis=1)
            combined_loss = tf.reduce_sum(weights * symmetric_losses, axis=0)

        return combined_loss, index

    def symmetric_losses(
        self,
        assignments: List[tf.Tensor],
        detections: List[tf.Tensor],
        targets: Tuple[Tuple[tf.Tensor, tf.Tensor], ...]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        assignments = [prediction + tf.math.log(tf.cast(decoder.num_targets, tf.float32))
                       for prediction, decoder in zip(assignments, self.branch_decoders)]

        targets = numpy_tensor_array(targets)

        symmetric_losses = self.compute_symmetric_losses(assignments, detections, targets)

        return self.combine_symmetric_losses(symmetric_losses)

    def symmetric_divergence_loss(self, predictions: List[tf.Tensor], masks: tf.Tensor) -> tf.Tensor:
        divergence_loss = []

        for i, j in self.event_info.event_transpositions:
            div = jensen_shannon_divergence(predictions[i], predictions[j])

            loss = tf.exp(-(div ** 2))
            loss = tf.where(tf.logical_not(masks[i]), 0.0, loss)
            loss = tf.where(tf.logical_not(masks[j]), 0.0, loss)

            divergence_loss.append(loss)

        return tf.reduce_mean(tf.stack(divergence_loss), axis=0)

    def add_kl_loss(
            self,
            total_loss: List[tf.Tensor],
            assignments: List[tf.Tensor],
            masks: tf.Tensor,
            weights: tf.Tensor
    ) -> List[tf.Tensor]:
        if len(self.event_info.event_transpositions) == 0:
            return total_loss

        kl_loss = self.symmetric_divergence_loss(assignments, masks)
        kl_loss = tf.reduce_sum(weights * kl_loss) / tf.reduce_sum(masks)

        self.add_metric(kl_loss, name="loss/symmetric_loss")
        if tf.math.is_nan(kl_loss):
            raise ValueError("Symmetric KL Loss has diverged.")

        return total_loss + [self.options.kl_loss_scale * kl_loss]

    def add_regression_loss(
            self,
            total_loss: List[tf.Tensor],
            predictions: Dict[str, tf.Tensor],
            targets:  Dict[str, tf.Tensor]
    ) -> List[tf.Tensor]:
        regression_terms = []

        for key in targets:
            current_target_type = self.training_dataset.regression_types[key]
            current_prediction = predictions[key]
            current_target = targets[key]

            current_mean = self.regression_decoder.networks[key].mean
            current_std = self.regression_decoder.networks[key].std

            current_mask = tf.logical_not(tf.math.is_nan(current_target))

            current_loss = regression_loss(current_target_type)(
                tf.boolean_mask(current_prediction, current_mask),
                tf.boolean_mask(current_target, current_mask),
                current_mean,
                current_std
            )
            current_loss = tf.reduce_mean(current_loss)

            self.add_metric(current_loss, name=f"loss/regression/{key}")

            regression_terms.append(self.options.regression_loss_scale * current_loss)

        return total_loss + regression_terms

    def add_classification_loss(
            self,
            total_loss: List[tf.Tensor],
            predictions: Dict[str, tf.Tensor],
            targets: Dict[str, tf.Tensor]
    ) -> List[tf.Tensor]:
        classification_terms = []

        for key in targets:
            current_prediction = predictions[key]
            current_target = targets[key]

            weight = None if not self.balance_classifications else self.classification_weights[key]
            current_loss = losses.sparse_categorical_crossentropy(
                current_target,
                current_prediction,
                from_logits=True,
                sample_weight=weight
            )

            classification_terms.append(self.options.classification_loss_scale * current_loss)

            self.add_metric(current_loss, name=f"loss/classification/{key}")

        return total_loss + classification_terms

    def train_step(self, batch: Batch) -> Dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            outputs = self(batch.sources)

            symmetric_losses, best_indices = self.symmetric_losses(
                outputs.assignments,
                outputs.detections,
                batch.assignment_targets
            )

            permutations = tf.gather(self.event_permutation_tensor, best_indices, axis=0)
            masks = tf.stack([target.mask for target in batch.assignment_targets])
            masks = tf.gather(masks, permutations, axis=0)

            weights = tf.ones_like(symmetric_losses)

            if self.balance_particles:
                class_indices = tf.reduce_sum(masks * self.particle_index_tensor[tf.newaxis, :], axis=0)
                weights *= tf.gather(self.particle_weights_tensor, class_indices)

            if self.balance_jets:
                weights *= tf.gather(self.jet_weights_tensor, batch.num_vectors)

            masks = tf.expand_dims(masks, axis=1)
            symmetric_losses = tf.reduce_sum(weights * symmetric_losses, axis=-1) / tf.clip_by_value(tf.reduce_sum(masks, axis=-1), 1, np.inf)
            assignment_loss, detection_loss = tf.unstack(symmetric_losses, axis=1)

            self.add_metric(assignment_loss, name="loss/assignment_loss")
            self.add_metric(detection_loss, name="loss/detection_loss")

            if tf.math.reduce_any(tf.math.is_nan(assignment_loss)):
                raise ValueError("Assignment loss has diverged!")

            if tf.math.reduce_any(tf.math.is_inf(assignment_loss)):
                raise ValueError("Assignment targets contain a collision.")

            total_loss = []

            if self.options.assignment_loss_scale > 0:
                total_loss.append(assignment_loss)

            if self.options.detection_loss_scale > 0:
                total_loss.append(detection_loss)

            if self.options.kl_loss_scale > 0:
                total_loss = self.add_kl_loss(total_loss, outputs.assignments, masks, weights)

            if self.options.regression_loss_scale > 0:
                total_loss = self.add_regression_loss(total_loss, outputs.regressions, batch.regression_targets)

            if self.options.classification_loss_scale > 0:
                total_loss = self.add_classification_loss(total_loss, outputs.classifications, batch.classification_targets)

            total_loss = tf.concat([tf.reshape(loss, [-1]) for loss in total_loss], axis=0)

            self.add_metric(tf.reduce_sum(total_loss), name="loss/total_loss")

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": tf.reduce_mean(total_loss)}

