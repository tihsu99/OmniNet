import h5py
import numpy as np

import tensorflow as tf

from spanet.dataset.types import SpecialKey, Statistics, Source
from spanet.dataset.inputs.BaseInput import BaseInput


class RelativeInput(BaseInput):
    def load(self, hdf5_file: h5py.File, limit_index: np.ndarray):
        input_group = [SpecialKey.Inputs, self.input_name]

        # Load in the mask for this vector input
        invariant_mask = tf.convert_to_tensor(self.dataset(hdf5_file, input_group, SpecialKey.Mask)[:], dtype=tf.bool)
        covariant_mask = tf.expand_dims(invariant_mask, axis=1) * tf.expand_dims(invariant_mask, axis=2)

        # Get all the features in this group, we need to figure out if each one is invariant or covariant
        feature_names = list(self.dataset(hdf5_file, [SpecialKey.Inputs], self.input_name).keys())
        feature_names.remove(SpecialKey.Mask)

        # Separate the two types of features by their shape
        self.invariant_features = []
        self.covariant_features = []

        for feature in self.event_info.input_features[self.input_name]:
            if len(self.dataset(hdf5_file, input_group, feature[0]).shape) == 2:
                self.invariant_features.append(feature)
            else:
                self.covariant_features.append(feature)

        # Load in vector features into a pre-made buffer
        num_jets = invariant_mask.shape[1]
        invariant_data = tf.TensorArray(dtype=tf.float32, size=len(self.invariant_features), dynamic_size=False)
        covariant_data = tf.TensorArray(dtype=tf.float32, size=len(self.covariant_features), dynamic_size=False)

        invariant_index = 0
        covariant_index = 0

        for (feature, _, log_transform) in self.event_info.input_features[self.input_name]:
            current_dataset = self.dataset(hdf5_file, input_group, feature)[:]
            if len(current_dataset.shape) == 2:
                current_data = invariant_data
                current_mask = invariant_mask
                current_index = invariant_index
                invariant_index += 1
            else:
                current_data = covariant_data
                current_mask = covariant_mask
                current_index = covariant_index
                covariant_index += 1

            current_data = current_data.write(current_index, current_dataset)
            if log_transform:
                current_data = current_data.read(current_index)
                current_data += 1
                current_data = tf.math.log(current_data)
                current_data = tf.where(current_mask, current_data, tf.zeros_like(current_data))
                current_data = current_data.write(current_index, current_data)

        invariant_data = invariant_data.stack()
        covariant_data = covariant_data.stack()

        invariant_data = tf.transpose(invariant_data, perm=[1, 2, 0])
        covariant_data = tf.transpose(covariant_data, perm=[1, 2, 3, 0])

        self.invariant_data = tf.gather(invariant_data, limit_index)
        self.covariant_data = tf.gather(covariant_data, limit_index)

        self.invariant_mask = tf.gather(invariant_mask, limit_index)
        self.covariant_mask = tf.gather(covariant_mask, limit_index)

    @property
    def reconstructable(self) -> bool:
        return True

    def limit(self, event_mask):
        self.invariant_data = tf.boolean_mask(self.invariant_data, event_mask)
        self.covariant_data = tf.boolean_mask(self.covariant_data, event_mask)

        self.invariant_mask = tf.boolean_mask(self.invariant_mask, event_mask)
        self.covariant_mask = tf.boolean_mask(self.covariant_mask, event_mask)

    def compute_statistics(self) -> Statistics:
        masked_invariant_data = tf.boolean_mask(self.invariant_data, self.invariant_mask)
        masked_covariant_data = tf.boolean_mask(self.covariant_data, self.covariant_mask)

        masked_invariant_mean = tf.reduce_mean(masked_invariant_data, axis=0)
        masked_invariant_std = tf.math.reduce_std(masked_invariant_data, axis=0)
        masked_invariant_std = tf.where(masked_invariant_std < 1e-5, tf.ones_like(masked_invariant_std), masked_invariant_std)

        masked_covariant_mean = tf.reduce_mean(masked_covariant_data, axis=0)
        masked_covariant_std = tf.math.reduce_std(masked_covariant_data, axis=0)
        masked_covariant_std = tf.where(masked_covariant_std < 1e-5, tf.ones_like(masked_covariant_std), masked_covariant_std)

        unnormalized_invariant_features = ~np.array([feature[1] for feature in self.invariant_features])
        masked_invariant_mean = tf.where(unnormalized_invariant_features, tf.zeros_like(masked_invariant_mean), masked_invariant_mean)
        masked_invariant_std = tf.where(unnormalized_invariant_features, tf.ones_like(masked_invariant_std), masked_invariant_std)

        unnormalized_covariant_features = ~np.array([feature[1] for feature in self.covariant_features])
        masked_covariant_mean = tf.where(unnormalized_covariant_features, tf.zeros_like(masked_covariant_mean), masked_covariant_mean)
        masked_covariant_std = tf.where(unnormalized_covariant_features, tf.ones_like(masked_covariant_std), masked_covariant_std)

        return Statistics(
            tf.concat((masked_invariant_mean, masked_covariant_mean), axis=0),
            tf.concat((masked_invariant_std, masked_covariant_std), axis=0)
        )

    def num_vectors(self) -> int:
        return tf.reduce_sum(tf.cast(self.invariant_mask, tf.int32), axis=1)

    def max_vectors(self) -> int:
        return tf.shape(self.invariant_mask)[1]

    def __getitem__(self, item) -> Source:
        invariant_data = self.invariant_data[item]
        covariant_data = self.covariant_data[item]

        invariant_data = tf.expand_dims(invariant_data, axis=-3)

        invariant_data_shape = tf.shape(invariant_data)
        invariant_data_shape = tf.concat([invariant_data_shape[:-3], [invariant_data_shape[-2]], invariant_data_shape[-3:-2]], axis=0)
        invariant_data = tf.broadcast_to(invariant_data, invariant_data_shape)

        return Source(
            data=tf.concat((invariant_data, covariant_data), axis=-1),
            mask=self.covariant_mask[item]
        )

