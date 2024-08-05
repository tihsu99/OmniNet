import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from spanet.options import Options
from spanet.dataset.types import Tuple, Outputs, Source, Predictions

from spanet.network.layers.vector_encoder import JetEncoder
from spanet.network.layers.branch_decoder import BranchDecoder
from spanet.network.layers.embedding import MultiInputVectorEmbedding
from spanet.network.layers.regression_decoder import RegressionDecoder
from spanet.network.layers.classification_decoder import ClassificationDecoder

from spanet.network.prediction_selection import extract_predictions
from spanet.network.jet_reconstruction.jet_reconstruction_base import JetReconstructionBase

TArray = np.ndarray


class JetReconstructionNetwork(JetReconstructionBase):
    def __init__(self, options: Options, torch_script: bool = False):
        """ Base class defining the SPANet architecture.

        Parameters
        ----------
        options: Options
            Global options for the entire network.
            See network.options.Options
        """
        super(JetReconstructionNetwork, self).__init__(options)

        self.hidden_dim = options.hidden_dim

        self.embedding = MultiInputVectorEmbedding(
            options,
            self.training_dataset
        )

        self.encoder = JetEncoder(
            options,
        )

        self.branch_decoders = [
            BranchDecoder(
                options,
                event_particle_name,
                self.event_info.product_particles[event_particle_name].names,
                product_symmetry,
                self.enable_softmax
            )
            for event_particle_name, product_symmetry
            in self.event_info.product_symmetries.items()
        ]

        self.regression_decoder = RegressionDecoder(
            options,
            self.training_dataset
        )

        self.classification_decoder = ClassificationDecoder(
            options,
            self.training_dataset
        )

    @property
    def enable_softmax(self):
        return True

    def call(self, sources: Tuple[Source, ...]) -> Outputs:
        # Embed all of the different input regression_vectors into the same latent space.
        embeddings, padding_masks, sequence_masks, global_masks = self.embedding(sources)

        # Extract features from data using transformer
        hidden, event_vector = self.encoder(embeddings, padding_masks, sequence_masks)

        # Create output lists for each particle in event.
        assignments = []
        detections = []

        encoded_vectors = {
            "EVENT": event_vector
        }

        # Pass the shared hidden state to every decoder branch
        for decoder in self.branch_decoders:
            (
                assignment,
                detection,
                assignment_mask,
                event_particle_vector,
                product_particle_vectors
            ) = decoder(hidden, padding_masks, sequence_masks, global_masks)

            assignments.append(assignment)
            detections.append(detection)

            # Assign the summarizing vectors to their correct structure.
            encoded_vectors["/".join([decoder.particle_name, "PARTICLE"])] = event_particle_vector
            for product_name, product_vector in zip(decoder.product_names, product_particle_vectors):
                encoded_vectors["/".join([decoder.particle_name, product_name])] = product_vector

        # Predict the valid regressions for any real values associated with the event.
        regressions = self.regression_decoder(encoded_vectors)

        # Predict additional classification targets for any branch of the event.
        classifications = self.classification_decoder(encoded_vectors)

        return Outputs(
            assignments,
            detections,
            encoded_vectors,
            regressions,
            classifications
        )

    def predict(self, sources: Tuple[Source, ...]) -> Predictions:
        outputs = self(sources)

        # Extract assignment probabilities and find the least conflicting assignment.
        assignments = extract_predictions([
            np.nan_to_num(assignment.numpy(), -np.inf)
            for assignment in outputs.assignments
        ])

        # Convert detection logits into probabilities and move to CPU.
        detections = np.stack([
            tf.sigmoid(detection).numpy()
            for detection in outputs.detections
        ])

        # Move regressions to CPU and away from torch.
        regressions = {
            key: value.numpy()
            for key, value in outputs.regressions.items()
        }

        classifications = {
            key: tf.argmax(value, axis=1).numpy()
            for key, value in outputs.classifications.items()
        }

        return Predictions(
            assignments,
            detections,
            regressions,
            classifications
        )

    def predict_assignments(self, sources: Tuple[Source, ...]) -> np.ndarray:
        # Run the base prediction step
        assignments = [
            np.nan_to_num(assignment.numpy(), -np.inf)
            for assignment in self(sources).assignments
        ]

        # Find the optimal selection of jets from the output distributions.
        return extract_predictions(assignments)

    def predict_assignments_and_detections(self, sources: Tuple[Source, ...]) -> Tuple[TArray, TArray]:
        assignments, detections, regressions, classifications = self.predict(sources)

        # Always predict the particle exists if we didn't train on it
        if self.options.detection_loss_scale == 0:
            detections += 1

        return assignments, detections >= 0.5

