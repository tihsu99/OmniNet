from argparse import ArgumentParser
from typing import Optional
from os import getcwd, makedirs, environ
import shutil
import json

import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.mixed_precision import set_global_policy, Policy

from spanet import JetReconstructionModel, Options


def main(
        event_file: str,
        training_file: str,
        validation_file: str,
        options_file: Optional[str],
        checkpoint: Optional[str],
        state_dict: Optional[str],
        freeze_state_dict: bool,

        log_dir: str,
        name: str,

        torch_script: bool,
        fp16: bool,
        verbose: bool,
        full_events: bool,

        profile: bool,
        gpus: Optional[int],
        epochs: Optional[int],
        time_limit: Optional[str],
        batch_size: Optional[int],
        limit_dataset: Optional[float],
        random_seed: int,
    ):

    master = True
    if "NODE_RANK" in environ:
        master = False

    options = Options(event_file, training_file, validation_file)

    if options_file is not None:
        with open(options_file, 'r') as json_file:
            options.update_options(json.load(json_file))

    options.verbose_output = verbose
    if master and verbose:
        print(f"Verbose output activated.")

    if full_events:
        if master:
            print(f"Overriding: Only using full events")
        options.partial_events = False
        options.balance_particles = False

    if gpus is not None:
        if master:
            print(f"Overriding GPU count: {gpus}")
        options.num_gpu = gpus

    if batch_size is not None:
        if master:
            print(f"Overriding Batch Size: {batch_size}")
        options.batch_size = batch_size

    if limit_dataset is not None:
        if master:
            print(f"Overriding Dataset Limit: {limit_dataset}%")
        options.dataset_limit = limit_dataset / 100

    if epochs is not None:
        if master:
            print(f"Overriding Number of Epochs: {epochs}")
        options.epochs = epochs

    if random_seed > 0:
        options.dataset_randomization = random_seed

    if master:
        options.display()

    model = JetReconstructionModel(options)

    if state_dict is not None:
        if master:
            print(f"Loading state dict from: {state_dict}")

        state_dict = tf.saved_model.load(state_dict)
        model.load_weights(state_dict)

        if freeze_state_dict:
            for layer in model.layers:
                layer.trainable = False

    log_dir = getcwd() if log_dir is None else log_dir
    logger = wandb.init(project=name, dir=log_dir) if wandb else TensorBoard(log_dir=log_dir, name=name)

    callbacks = [
        ModelCheckpoint(
            filepath=log_dir,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        LearningRateScheduler(schedule=lambda epoch: 1e-4 * (0.1 ** int(epoch / 10))),
        TensorBoard(log_dir=log_dir, update_freq='epoch')
    ]

    if profile:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch='500,520'))

    if fp16:
        set_global_policy(Policy('mixed_float16'))

    strategy = tf.distribute.MirroredStrategy(devices=[f"/gpu:{i}" for i in range(gpus)]) if gpus else tf.distribute.get_strategy()

    with strategy.scope():
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        if checkpoint:
            model.load_weights(checkpoint)

        makedirs(logger.log_dir, exist_ok=True)
        with open(f"{logger.log_dir}/options.json", 'w') as json_file:
            json.dump(options.__dict__, json_file, indent=4)
        shutil.copy2(options.event_info_file, f"{logger.log_dir}/event.yaml")

        model.fit(
            model.train_dataloader(),
            validation_data=model.val_dataloader(),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-ef", "--event_file", type=str, default="",
                        help="Input file containing event symmetry information.")

    parser.add_argument("-tf", "--training_file", type=str, default="",
                        help="Input file containing training data.")

    parser.add_argument("-vf", "--validation_file", type=str, default="",
                        help="Input file containing Validation data. If not provided, will use training data split.")

    parser.add_argument("-of", "--options_file", type=str, default=None,
                        help="JSON file with option overloads.")

    parser.add_argument("-cf", "--checkpoint", type=str, default=None,
                        help="Optional checkpoint to load the training state from. "
                             "Fully restores model weights and optimizer state.")

    parser.add_argument("-sf", "--state_dict", type=str, default=None,
                        help="Load from checkpoint but only the model weights. "
                             "Can be partial as the weights don't have to match one-to-one.")

    parser.add_argument("-fsf", "--freeze_state_dict", action='store_true',
                        help="Freeze any weights that were loaded from the state dict. "
                             "Used for finetuning new layers.")

    parser.add_argument("-l", "--log_dir", type=str, default=None,
                        help="Output directory for the checkpoints and tensorboard logs. Default to current directory.")

    parser.add_argument("-n", "--name", type=str, default="spanet_output",
                        help="The sub-directory to create for this run and an identifier for WANDB.")

    parser.add_argument("-e", "--epochs", type=int, default=None,
                        help="Override number of epochs to train for")
    
    parser.add_argument("-t", "--time_limit", type=str, default=None,
                        help="Time limit for training, in the format DD:HH:MM:SS.")

    parser.add_argument("-g", "--gpus", type=int, default=None,
                        help="Override GPU count in hyperparameters.")
    
    parser.add_argument("-b", "--batch_size", type=int, default=None,
                        help="Override batch size in hyperparameters.")

    parser.add_argument("-f", "--full_events", action='store_true',
                        help="Limit training to only full events.")

    parser.add_argument("-p", "--limit_dataset", type=float, default=None,
                        help="Limit dataset to only the first L percent of the data (0 - 100).")

    parser.add_argument("-fp16", "--fp16", action="store_true",
                        help="Use TensorFlow mixed precision for training.")

    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Output additional information to console and log.")

    parser.add_argument("-r", "--random_seed", type=int, default=0,
                        help="Set random seed for cross-validation.")

    parser.add_argument("-ts", "--torch_script", action='store_true',
                        help="Compile the neural network using TensorFlow function compilation.")

    parser.add_argument("--profile", action='store_true',
                        help="Profile network for a single training epoch.")

    main(**parser.parse_args().__dict__)

