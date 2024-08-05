import os
from glob import glob
from typing import Optional, Union, Tuple
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar

from spanet import JetReconstructionModel, Options
from spanet.dataset.types import Evaluation, Outputs, Source
from spanet.network.jet_reconstruction.jet_reconstruction_network import extract_predictions


def dict_concatenate(tree):
    output = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            output[key] = dict_concatenate(value)
        else:
            output[key] = np.concatenate(value)

    return output


def tree_concatenate(trees):
    leaves = []
    for tree in trees:
        data, tree_spec = tf.nest.flatten(tree)
        leaves.append(data)

    results = [np.concatenate(l) for l in zip(*leaves)]
    return tf.nest.pack_sequence_as(tree_spec, results)


def load_model(
    log_directory: str,
    testing_file: Optional[str] = None,
    event_info_file: Optional[str] = None,
    batch_size: Optional[int] = None,
    cuda: bool = False,
    fp16: bool = False,
    checkpoint: Optional[str] = None
) -> JetReconstructionModel:
    if checkpoint is None:
        checkpoint = sorted(glob(os.path.join(log_directory, "checkpoints/epoch*")))[-1]
        print(f"Loading: {checkpoint}")

    model = JetReconstructionModel.load_from_checkpoint(checkpoint)

    if fp16:
        model = model.to_fp16()

    if testing_file is not None:
        model.options.testing_file = testing_file

    if event_info_file is not None:
        model.options.event_info_file = event_info_file

    if batch_size is not None:
        model.options.batch_size = batch_size

    if cuda:
        model = model.cuda()

    return model


def evaluate_on_test_dataset(
        model: JetReconstructionModel,
        progress=Progbar,
        return_full_output: bool = False,
        fp16: bool = False
) -> Union[Evaluation, Tuple[Evaluation, Outputs]]:
    full_assignments = defaultdict(list)
    full_assignment_probabilities = defaultdict(list)
    full_detection_probabilities = defaultdict(list)

    full_classifications = defaultdict(list)
    full_regressions = defaultdict(list)

    full_outputs = []

    dataloader = model.test_dataloader()
    if progress:
        progress_bar = progress(len(dataloader), verbose=1)

    for batch in dataloader:
        sources = tuple(Source(x[0], x[1]) for x in batch.sources)

        if fp16:
            with tf.keras.mixed_precision.experimental.Policy('mixed_float16'):
                outputs = model(sources)
        else:
            outputs = model(sources)

        assignment_indices = extract_predictions([
            np.nan_to_num(assignment.numpy(), -np.inf)
            for assignment in outputs.assignments
        ])

        detection_probabilities = np.stack([
            tf.sigmoid(detection).numpy()
            for detection in outputs.detections
        ])

        classifications = {
            key: tf.nn.softmax(classification, axis=1).numpy()
            for key, classification in outputs.classifications.items()
        }

        regressions = {
            key: value.numpy()
            for key, value in outputs.regressions.items()
        }

        assignment_probabilities = []
        dummy_index = np.arange(assignment_indices[0].shape[0])
        for assignment_probability, assignment, symmetries in zip(
            outputs.assignments,
            assignment_indices,
            model.event_info.product_symbolic_groups.values()
        ):
            assignment_probability = tf.gather_nd(assignment_probability, np.stack((dummy_index, *assignment.T), axis=-1))
            assignment_probability = tf.exp(assignment_probability)
            assignment_probability = symmetries.order() * assignment_probability
            assignment_probabilities.append(assignment_probability.numpy())

        for i, name in enumerate(model.event_info.product_particles):
            full_assignments[name].append(assignment_indices[i])
            full_assignment_probabilities[name].append(assignment_probabilities[i])
            full_detection_probabilities[name].append(detection_probabilities[i])

        for key, regression in regressions.items():
            full_regressions[key].append(regression)

        for key, classification in classifications.items():
            full_classifications[key].append(classification)

        if return_full_output:
            full_outputs.append(tf.nest.map_structure(lambda x: x.numpy(), outputs))

        if progress:
            progress_bar.update(batch_idx)

    evaluation = Evaluation(
        dict_concatenate(full_assignments),
        dict_concatenate(full_assignment_probabilities),
        dict_concatenate(full_detection_probabilities),
        dict_concatenate(full_regressions),
        dict_concatenate(full_classifications)
    )

    if return_full_output:
        return evaluation, tree_concatenate(full_outputs)

    return evaluation

