from functools import reduce
from itertools import permutations, product
import warnings

import numpy as np
import tensorflow as tf

from spanet.dataset.event_info import EventInfo


class SymmetricEvaluator:
    def __init__(self, event_info: EventInfo):
        self.event_info = event_info
        self.event_group = event_info.event_symbolic_group
        self.target_groups = event_info.product_symbolic_groups

        clusters = []
        cluster_groups = []

        for orbit in self.event_group.orbits():
            orbit = tuple(sorted(orbit))
            names = [event_info.event_particles[i] for i in orbit]

            cluster_name = map(dict.fromkeys, names)
            cluster_name = map(lambda x: x.keys(), cluster_name)
            cluster_name = ''.join(reduce(lambda x, y: x & y, cluster_name))
            clusters.append((cluster_name, names, orbit))

            cluster_group = self.target_groups[names[0]]
            for name in names:
                assert (
                    self.target_groups[name] == cluster_group,
                    "Invalid Symmetry Group. Invariant targets have different structures."
                )

            cluster_groups.append((cluster_name, names, cluster_group))

        self.clusters = clusters
        self.cluster_groups = cluster_groups

    @staticmethod
    def permute_arrays(array_list, permutation):
        return [array_list[index] for index in permutation]

    def sort_outputs(self, predictions, target_jets, target_masks):
        predictions = [tf.identity(p) for p in predictions]
        target_jets = [tf.identity(p) for p in target_jets]

        for i, (_, particle_group) in enumerate(self.target_groups.items()):
            for orbit in particle_group.orbits():
                orbit = tuple(sorted(orbit))

                target_jets[i][:, orbit] = tf.sort(target_jets[i][:, orbit], axis=1)
                predictions[i][:, orbit] = tf.sort(predictions[i][:, orbit], axis=1)

        return predictions, target_jets, target_masks

    def particle_count_info(self, target_masks):
        target_masks = tf.convert_to_tensor(target_masks)

        total_particle_counts = tf.reduce_sum(target_masks, axis=0)

        particle_counts = [tf.reduce_sum(target_masks[list(cluster_indices)], axis=0)
                           for _, _, cluster_indices in self.clusters]

        particle_max = [len(cluster_indices) for _, _, cluster_indices in self.clusters]

        return total_particle_counts, particle_counts, particle_max

    def cluster_purity(self, predictions, target_jets, target_masks):
        results = []

        for cluster_name, cluster_particles, cluster_indices in self.clusters:
            cluster_target_masks = tf.stack([target_masks[i] for i in cluster_indices])
            cluster_target_jets = tf.stack([target_jets[i] for i in cluster_indices])
            cluster_predictions = tf.stack([predictions[i] for i in cluster_indices])

            best_accuracy = tf.zeros(cluster_target_masks.shape[1], dtype=tf.int64)

            for target_permutation in permutations(range(len(cluster_indices))):
                target_permutation = list(target_permutation)

                accuracy = tf.equal(cluster_predictions, tf.gather(cluster_target_jets, target_permutation, axis=0))
                accuracy = tf.reduce_all(accuracy, axis=-1) * tf.gather(cluster_target_masks, target_permutation, axis=0)
                accuracy = tf.reduce_sum(tf.cast(accuracy, tf.int64), axis=0)

                best_accuracy = tf.maximum(accuracy, best_accuracy)

            total_particles = tf.reduce_sum(tf.cast(cluster_target_masks, tf.float32))
            if total_particles > 0:
                cluster_accuracy = tf.reduce_sum(tf.cast(best_accuracy, tf.float32)) / total_particles
            else:
                cluster_accuracy = np.nan

            results.append((cluster_name, cluster_particles, cluster_accuracy))

        return results

    def event_purity(self, predictions, target_jets, target_masks):
        target_masks = tf.stack(target_masks)

        best_accuracy = tf.zeros(target_masks.shape[1], dtype=tf.int64)

        for target_permutation in self.event_info.event_permutation_group:
            permuted_targets = self.permute_arrays(target_jets, target_permutation)
            permuted_mask = self.permute_arrays(target_masks, target_permutation)
            accuracy = np.array([(p == t).all(-1) * m
                                 for p, t, m
                                 in zip(predictions, permuted_targets, permuted_mask)])
            accuracy = tf.reduce_sum(tf.cast(accuracy, tf.int64), axis=0)

            best_accuracy = tf.maximum(accuracy, best_accuracy)

        num_particles_in_event = tf.reduce_sum(tf.cast(target_masks, tf.int64), axis=0)
        accurate_event = tf.equal(best_accuracy, num_particles_in_event)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return tf.reduce_mean(tf.cast(accurate_event, tf.float32))

    def full_report(self, predictions, target_jets, target_masks):
        predictions, target_jets, target_masks = self.sort_outputs(predictions, target_jets, target_masks)

        total_particle_counts, particle_counts, particle_max = self.particle_count_info(target_masks)
        particle_ranges = [list(range(-1, pmax + 1)) for pmax in particle_max]

        full_results = []

        for event_counts in product(*particle_ranges):
            event_mask = total_particle_counts >= 0

            for particle_count, event_count in zip(particle_counts, event_counts):
                if event_count >= 0:
                    event_mask = event_mask & (particle_count == event_count)

                if event_count < 0:
                    event_mask = event_mask & (total_particle_counts > 0)

            event_mask = tf.cast(event_mask, tf.bool)

            masked_predictions = [tf.boolean_mask(p, event_mask) for p in predictions]
            masked_target_jets = [tf.boolean_mask(p, event_mask) for p in target_jets]
            masked_target_masks = [tf.boolean_mask(p, event_mask) for p in target_masks]

            masked_event_purity = self.event_purity(masked_predictions, masked_target_jets, masked_target_masks)
            masked_cluster_purity = self.cluster_purity(masked_predictions, masked_target_jets, masked_target_masks)

            mask_proportion = tf.reduce_mean(tf.cast(event_mask, tf.float32))

            full_results.append((event_counts, mask_proportion, masked_event_purity, masked_cluster_purity))

        return full_results

    def full_report_string(self, predictions, target_jets, target_masks, prefix: str = ""):
        full_purities = {}

        report = self.full_report(predictions, target_jets, target_masks)
        for event_mask, mask_proportion, event_purity, particle_purity in report:

            event_mask_name = ""
            purity = {
                "{}{}/event_purity": event_purity,
                "{}{}/event_proportion": mask_proportion
            }

            for mask_count, (cluster_name, _, cluster_purity) in zip(event_mask, particle_purity):
                mask_count = "*" if mask_count < 0 else str(mask_count)
                event_mask_name = event_mask_name + mask_count + cluster_name
                purity["{}{}/{}_purity".format("{}", "{}", cluster_name)] = cluster_purity

            purity = {
                key.format(prefix, event_mask_name): val for key, val in purity.items()
            }

            full_purities.update(purity)

        return full_purities

