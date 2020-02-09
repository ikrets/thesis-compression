import tensorflow as tf

tfk = tf.keras
tfkl = tf.keras.layers


class FiLM(tfkl.Layer):
    def __init__(self, width, depth, activation, **kwargs):
        super(FiLM, self).__init__(**kwargs)

        self.width = width
        self.depth = depth
        self.activation = activation

    def build(self, input_shape):
        parameter_shape, feature_map_shape = input_shape

        feature_map_input = tfkl.Input(batch_shape=feature_map_shape)
        parameter_input = tfkl.Input(batch_shape=parameter_shape)

        out = parameter_input
        for i in range(self.depth - 1):
            out = tfkl.Dense(self.width, activation=self.activation, use_bias=True)(out)
        parameters = tfkl.Dense(feature_map_shape[-1] * 2, activation=None, use_bias=True)(out)
        parameters = parameters[..., tf.newaxis, tf.newaxis]
        parameters = tf.transpose(parameters, [0, 2, 3, 1])
        parameters_mu, parameters_sigma = tf.split(parameters, num_or_size_splits=2, axis=-1)

        modulated_feature_map = parameters_sigma * feature_map_input + parameters_mu

        self.net = tfk.Model(inputs=[parameter_input, feature_map_input], outputs=modulated_feature_map)

    def call(self, parameters_and_feature_map, **kwargs):
        return self.net(parameters_and_feature_map)