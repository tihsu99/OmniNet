from typing import Tuple

import h5py
import numpy as np

import tensorflow as tf
from spanet.dataset.types import SpecialKey, Statistics, Source
from spanet.dataset.inputs.BaseInput import BaseInput

class GlobalInput(BaseInput):

    def load(self, hdf5_file: h5py.File, limit_index: np.ndarray):
        input_group = [SpecialKey.Inputs, self.input_name]

        # Try and load a mask for this global features. If none is present, assume all vectors are valid.
        try:
            source_mask = tf.convert_to_tensor(self.dataset(hdf5_file, input_group, SpecialKey.Mask)[:], dtype=tf.bool)
        except KeyError:
            source_mask = tf.ones(self.num_events, dtype=tf.bool)

        # Load in vector features.
        num_features = self.event_info.num_features(self.input_name)
        source_data = tf.TensorArray(dtype=tf.float32, size=num_features, dynamic_size=False)

        for index, (feature, _, log_transform) in enumerate(self.event_info.input_features[self.input_name]):
            data = self.dataset(hdf5_file, input_group, feature)[:]
            if log_transform:
                data += 1
                data = tf.math.log(data)
                data = tf.where(source_mask, data, tf.zeros_like(data))
            source_data = source_data.write(index, data)

        source_data = source_data.stack()
        source_data = tf.transpose(source_data, perm=[1, 0])
        source_data = tf.gather(source_data, limit_index)
        source_mask = tf.gather(source_mask, limit_index)

        # Add a fake timestep dimension to global vectors.
        self.source_data = tf.expand_dims(source_data, axis=1)
        self.source_mask = tf.expand_dims(source_mask, axis=1)

    @property
    def reconstructable(self) -> bool:
        return False

    def limit(self, event_mask):
        self.source_data = tf.boolean_mask(self.source_data, event_mask)
        self.source_mask = tf.boolean_mask(self.source_mask, event_mask)

    def compute_statistics(self) -> Tuple[tf.Tensor, tf.Tensor]:
        masked_data = tf.boolean_mask(self.source_data, self.source_mask)
        masked_mean = tf.reduce_mean(masked_data, axis=0)
        masked_std = tf.math.reduce_std(masked_data, axis=0)

        masked_std = tf.where(masked_std < 1e-5, tf.ones_like(masked_std), masked_std)

        normalized_features = self.event_info.normalized_features(self.input_name)
        masked_mean = tf.where(normalized_features, masked_mean, tf.zeros_like(masked_mean))
        masked_std = tf.where(normalized_features, masked_std, tf.ones_like(masked_std))

        return Statistics(masked_mean, masked_std)

    def num_vectors(self) -> int:
        return tf.reduce_sum(tf.cast(self.source_mask, tf.int32), axis=1)

    def max_vectors(self) -> int:
        return tf.shape(self.source_mask)[1]

    def __getitem__(self, item):
        return Source(self.source_data[item], self.source_mask[item])

