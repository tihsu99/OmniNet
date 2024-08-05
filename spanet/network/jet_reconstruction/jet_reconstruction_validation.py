from typing import Dict, Callable
import warnings

import numpy as np
import tensorflow as tf

from sklearn import metrics as sk_metrics

from spanet.options import Options
from spanet.dataset.evaluator import SymmetricEvaluator
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork


class JetReconstructionValidation(JetReconstructionNetwork):
    def __init__(self, options: Options, torch_script: bool = False):
        super(JetReconstructionValidation, self).__init__(options, torch_script)
        self.evaluator = SymmetricEvaluator(self.training_dataset.event_info)

    @property
    def particle_metrics(self) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        return {
            "accuracy": sk_metrics.accuracy_score,
            "sensitivity": sk_metrics.recall_score,
            "specificity": lambda t, p: sk_metrics.recall_score(~t, ~p),
            "f_score": sk_metrics.f1_score
        }

    @property
    def particle_score_metrics(self) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        return {
            # "roc_auc": sk_metrics.roc_auc_score,
            # "average_precision": sk_metrics.average_precision_score
        }

    def compute_metrics(self, jet_predictions, particle_scores, stacked_targets, stacked_masks):
        event_permutation_group = self.event_permutation_tensor.numpy()
        num_permutations = len(event_permutation_group)
        num_targets, batch_size = stacked_masks.shape
        particle_predictions = particle_scores >= 0.5

        jet_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=bool)
        particle_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=bool)
        for i, permutation in enumerate(event_permutation_group):
            for j, (prediction, target) in enumerate(zip(jet_predictions, stacked_targets[permutation])):
                jet_accuracies[i, j] = np.all(prediction == target, axis=1)

            particle_accuracies[i] = stacked_masks[permutation] == particle_predictions

        jet_accuracies = jet_accuracies.sum(1)
        particle_accuracies = particle_accuracies.sum(1)

        chosen_permutations = tf.gather(self.event_permutation_tensor, tf.argmax(jet_accuracies, axis=0), axis=0)
        chosen_permutations = chosen_permutations.numpy().T
        permuted_masks = tf.gather(tf.convert_to_tensor(stacked_masks), chosen_permutations).numpy()

        num_particles = stacked_masks.sum(0)
        jet_accuracies = jet_accuracies.max(0)
        particle_accuracies = particle_accuracies.max(0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
    
            metrics = {f"jet/accuracy_{i}_of_{j}": (jet_accuracies[num_particles == j] >= i).mean()
                    for j in range(1, num_targets + 1)
                    for i in range(1, j + 1)}

            metrics.update({f"particle/accuracy_{i}_of_{j}": (particle_accuracies[num_particles == j] >= i).mean()
                            for j in range(1, num_targets + 1)
                            for i in range(1, j + 1)})

        particle_scores = particle_scores.ravel()
        particle_targets = permuted_masks.ravel()
        particle_predictions = particle_predictions.ravel()

        for name, metric in self.particle_metrics.items():
            metrics[f"particle/{name}"] = metric(particle_targets, particle_predictions)

        for name, metric in self.particle_score_metrics.items():
            metrics[f"particle/{name}"] = metric(particle_targets, particle_scores)

        metrics["validation_accuracy"] = metrics[f"jet/accuracy_{num_targets}_of_{num_targets}"]

        return metrics

    def validation_step(self, batch, batch_idx) -> Dict[str, np.float32]:
        sources, num_jets, targets, regression_targets, classification_targets = batch
        jet_predictions, particle_scores, regressions, classifications = self.predict(sources)

        batch_size = num_jets.shape[0]
        num_targets = len(targets)

        stacked_targets = np.zeros(num_targets, dtype=object)
        stacked_masks = np.zeros((num_targets, batch_size), dtype=bool)
        for i, (target, mask) in enumerate(targets):
            stacked_targets[i] = target.numpy()
            stacked_masks[i] = mask.numpy()

        regression_targets = {
            key: value.numpy()
            for key, value in regression_targets.items()
        }

        classification_targets = {
            key: value.numpy()
            for key, value in classification_targets.items()
        }

        metrics = self.evaluator.full_report_string(jet_predictions, stacked_targets, stacked_masks, prefix="Purity/")

        for target, prediction, decoder in zip(stacked_targets, jet_predictions, self.branch_decoders):
            for indices in decoder.permutation_indices:
                if len(indices) > 1:
                    prediction[:, indices] = np.sort(prediction[:, indices])
                    target[:, indices] = np.sort(target[:, indices])

        metrics.update(self.compute_metrics(jet_predictions, particle_scores, stacked_targets, stacked_masks))

        for key in regressions:
            delta = regressions[key] - regression_targets[key]
            
            percent_error = np.abs(delta / regression_targets[key])
            self.log(f"REGRESSION/{key}_percent_error", percent_error.mean())

            absolute_error = np.abs(delta)
            self.log(f"REGRESSION/{key}_absolute_error", absolute_error.mean())

            percent_deviation = delta / regression_targets[key]
            tf.summary.histogram(f"REGRESSION/{key}_percent_deviation", percent_deviation, step=self.optimizer.iterations)

            absolute_deviation = delta
            tf.summary.histogram(f"REGRESSION/{key}_absolute_deviation", absolute_deviation, step=self.optimizer.iterations)

        for key in classifications:
            accuracy = (classifications[key] == classification_targets[key])
            self.log(f"CLASSIFICATION/{key}_accuracy", accuracy.mean())

        for name, value in metrics.items():
            if not np.isnan(value):
                self.log(name, value)

        return metrics

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

