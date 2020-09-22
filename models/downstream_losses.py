import tensorflow as tf
import tensorflow_probability as tfp
from typing import Callable, Sequence, Optional, Dict

tfd = tfp.distributions


class PerceptualLoss:
    '''
    Generalizes activation difference.
    '''

    def __init__(self,
                 model: tf.keras.Model,
                 metric_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                 preprocess_fn: Callable[[tf.Tensor], tf.Tensor],
                 readout_layers: Sequence[str],
                 normalize_activations: bool,
                 backbone_layer: Optional[str] = None) -> None:
        if backbone_layer:
            self.backbone_model = model.get_layer(backbone_layer)
        else:
            self.backbone_model = model
        self.model = model

        self.metric_fn = metric_fn
        self.preprocess_fn = preprocess_fn
        self.readout_layers = readout_layers
        self.normalize_activations = normalize_activations

        self.model_readouts = tf.keras.Model(inputs=self.backbone_model.input,
                                             outputs=[self.backbone_model.get_layer(L).output for L in
                                                      self.readout_layers])

    def loss(self, item: Dict[str, tf.Tensor]) -> tf.Tensor:
        original_readouts = self.model_readouts(self.preprocess_fn(item['X']))
        reconstruction_readouts = self.model_readouts(self.preprocess_fn(item['X_reconstruction']))

        # tf.keras.Model squeezes a list of one output
        if len(self.readout_layers) == 1:
            original_readouts = [original_readouts]
            reconstruction_readouts = [reconstruction_readouts]

        reduced_channels = []
        for i in range(len(self.readout_layers)):
            if self.normalize_activations:
                scaled_original_readouts = tf.math.l2_normalize(original_readouts[i], axis=-1)
                scaled_reconstruction_readouts = tf.math.l2_normalize(reconstruction_readouts[i], axis=-1)
                diff = tf.math.squared_difference(scaled_original_readouts, scaled_reconstruction_readouts)
            else:
                diff = tf.math.squared_difference(original_readouts[i], reconstruction_readouts[i])

            reduced_channels.append(tf.reduce_mean(diff, axis=(1, 2)))

        return tf.reduce_mean(tf.concat(reduced_channels, axis=1), axis=1)

    def metric(self,
               X_reconstruction: tf.Tensor,
               label: tf.Tensor) -> tf.Tensor:
        return self.metric_fn(label, self.model(self.preprocess_fn(X_reconstruction)))


class PredictionCrossEntropy:
    def __init__(self,
                 model: tf.keras.Model,
                 metric_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                 preprocess_fn: Callable[[tf.Tensor], tf.Tensor],
                 ) -> None:
        self.model = model
        self.metric_fn = metric_fn
        self.preprocess_fn = preprocess_fn

    def loss(self, item: Dict[str, tf.Tensor]) -> tf.Tensor:
        original_preds = self.model(self.preprocess_fn(item['X']))
        reconstruction_preds = self.model(self.preprocess_fn(item['X_reconstruction']))

        original_dist = tfd.Categorical(probs=original_preds)
        reconstruction_dist = tfd.Categorical(probs=reconstruction_preds)

        return tfd.kl_divergence(original_dist, reconstruction_dist)

    def metric(self, X_reconstruction: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        return self.metric_fn(label, self.model(self.preprocess_fn(X_reconstruction)))


class TaskCrossEntropy:
    def __init__(self,
                 model: tf.keras.Model,
                 metric_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                 preprocess_fn: Callable[[tf.Tensor], tf.Tensor],
                 ) -> None:
        self.model = model
        self.metric_fn = metric_fn
        self.preprocess_fn = preprocess_fn

    def loss(self, item: Dict[str, tf.Tensor]) -> tf.Tensor:
        reconstruction_preds = self.model(self.preprocess_fn(item['X_reconstruction']))
        return tf.keras.losses.categorical_crossentropy(item['label'], reconstruction_preds)

    def metric(self, X_reconstruction: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        return self.metric_fn(label, self.model(self.preprocess_fn(X_reconstruction)))


class Gradcam:
    def __init__(self,
                 model: tf.keras.Model,
                 metric_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                 preprocess_fn: Callable[[tf.Tensor], tf.Tensor],
                 ) -> None:
        self.model = model
        self.metric_fn = metric_fn
        self.preprocess_fn = preprocess_fn

    def loss(self, item: Dict[str, tf.Tensor]):
        return tf.reduce_mean(item['gradcam_heatmap'] * tf.squared_difference(item['X'], item['X_reconstruction']),
                              axis=[1, 2, 3])

    def metric(self, X_reconstruction: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        return self.metric_fn(label, self.model(self.preprocess_fn(X_reconstruction)))
