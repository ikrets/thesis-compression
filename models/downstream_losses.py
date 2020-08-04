import tensorflow as tf
from typing import Callable, Sequence, Any


class PerceptualLoss:
    '''
    Generalizes activation difference.
    '''

    def __init__(self,
                 model: tf.keras.Model,
                 metric_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                 preprocess_fn: Callable[[tf.Tensor], tf.Tensor],
                 readout_layers: Sequence[str],
                 normalize_activations: bool) -> None:
        self.model = model
        self.metric_fn = metric_fn
        self.preprocess_fn = preprocess_fn
        self.readout_layers = readout_layers
        self.normalize_activations = normalize_activations

        self.model_readouts = tf.keras.Model(inputs=self.model.input,
                                             outputs=[self.model.get_layer(L).output for L in self.readout_layers])

    def loss(self,
             X: tf.Tensor,
             X_reconstruction: tf.Tensor) -> tf.Tensor:

        original_readouts = self.model_readouts(self.preprocess_fn(X))
        reconstruction_readouts = self.model_readouts(self.preprocess_fn(X_reconstruction))

        # tf.keras.Model squeezes a list of one output
        if len(self.readout_layers) == 1:
            original_readouts = [original_readouts]
            reconstruction_readouts = [reconstruction_readouts]

        reduced_channels = []
        for i in range(len(self.readout_layers)):
            if self.normalize_activations:
                scaled_original_readouts, _ = tf.linalg.normalize(original_readouts[i], axis=-1)
                scaled_reconstruction_readouts, _ = tf.linalg.normalize(reconstruction_readouts[i], axis=-1)
                diff = tf.math.squared_difference(scaled_original_readouts, scaled_reconstruction_readouts)
            else:
                diff = tf.math.squared_difference(original_readouts[i], reconstruction_readouts[i])

            reduced_channels.append(tf.reduce_mean(diff, axis=(1, 2)))

        return tf.reduce_mean(tf.concat(reduced_channels, axis=1), axis=1)

    def metric(self,
               X_reconstruction: tf.Tensor,
               label: tf.Tensor) -> tf.Tensor:
        return self.metric_fn(label, self.model(self.preprocess_fn(X_reconstruction)))
