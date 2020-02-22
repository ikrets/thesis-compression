import tensorflow as tf


class DownstreamTask:
    def __init__(self, model, preprocess_fn, metric_fn, last_frozen_layer):
        self.model = model
        self.metric_fn = metric_fn
        self.preprocess_fn = preprocess_fn
        self.last_frozen_layer = last_frozen_layer

        self.frozen_variables = []
        for i in range(len(self.model.layers)):
            layer = self.model.get_layer(index=i)

            self.frozen_variables.extend(layer.variables)

            if layer.name == self.last_frozen_layer:
                break

    def loss(self, X, X_reconstruction, label):
        raise RuntimeError('not implemented')

    def metric(self, X_reconstruction, label):
        return self.metric_fn(label, self.model(self.preprocess_fn(X_reconstruction)))


class DownstreamTaskPerformance(DownstreamTask):
    def __init__(self, model_performance_loss, **kwargs):
        super(DownstreamTaskPerformance, self).__init__(**kwargs)

        self.model_performance_loss = model_performance_loss

    def loss(self, X, X_reconstruction, label):
        uncompressed_performance_loss = self.model_performance_loss(label,
                                                                    self.model(self.preprocess_fn(X)))

        compressed_performance_loss = self.model_performance_loss(label,
                                                                  self.model(self.preprocess_fn(X_reconstruction)))

        return tf.maximum(compressed_performance_loss - uncompressed_performance_loss, 0)


class DownstreamActivationDifference(DownstreamTask):
    def __init__(self, activation_layer, **kwargs):
        super(DownstreamActivationDifference, self).__init__(**kwargs)
        self.model_to_activation = tf.keras.Model(inputs=self.model.input,
                                                  outputs=self.model.get_layer(activation_layer).output)

    def loss(self, X, X_reconstruction, label):
        return tf.reduce_mean(tf.squared_difference(self.model_to_activation(self.preprocess_fn(X)),
                                                    self.model_to_activation(self.preprocess_fn(X_reconstruction))),
                              axis=(1, 2, 3))
