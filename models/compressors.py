import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

tfk = tf.keras
tfkl = tf.keras.layers
K = tf.keras.backend


class FiLM(tfkl.Layer):
    def __init__(self, width, depth, activation, **kwargs):
        super(FiLM, self).__init__(**kwargs)

        self.width = width
        self.depth = depth
        self.activation = activation

    def apply_mu_sigma(self, inputs):
        parameters_input, fm_input = inputs
        params = parameters_input[..., tf.newaxis, tf.newaxis]
        params = tf.transpose(params, [0, 2, 3, 1])
        mu, sigma = tf.split(params, num_or_size_splits=2, axis=-1)
        return sigma * fm_input + mu

    def build(self, input_shape):
        parameter_shape, feature_map_shape = input_shape

        self.layers = [tfkl.Dense(self.width, activation=self.activation, use_bias=True) for i in range(self.depth - 1)]
        self.layers.append(tfkl.Dense(feature_map_shape[-1] * 2, activation=None, use_bias=True))

        self.layers[0].build(parameter_shape)
        for i in range(1, self.depth):
            self.layers[i].build([parameter_shape[0], self.width])

    def compute_output_shape(self, input_shape):
        self.build(input_shape)
        parameter_shape, feature_map_shape = input_shape

        return feature_map_shape

    def call(self, parameters_and_feature_map):
        parameters, feature_map = parameters_and_feature_map
        out = parameters
        for l in self.layers:
            out = l(out)
        return self.apply_mu_sigma([out, feature_map])


class AnalysisTransform(tfkl.Layer):
    """The analysis transform."""

    def __init__(self, num_filters, depth, num_postproc, FiLM_depth=None, FiLM_width=None,
                 FiLM_activation=None, *args,
                 **kwargs):
        super(AnalysisTransform, self).__init__(*args, **kwargs)

        self.num_filters = num_filters
        self.depth = depth
        self.num_postproc = num_postproc

        self.use_FiLM = FiLM_depth
        self.FiLM_width = FiLM_width
        self.FiLM_depth = FiLM_depth
        self.FiLM_activation = FiLM_activation

        self.layers = []
        for i in range(self.depth):
            self.layers.append(tfc.SignalConv2D(
                self.num_filters,
                (5, 5), name=f"layer_{i}", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=None))
            if i != self.depth - 1 or self.num_postproc:
                if self.use_FiLM:
                    self.layers.append(FiLM(width=self.FiLM_width, depth=self.FiLM_depth,
                                            activation=self.FiLM_activation,
                                            name=f'layer_{i}_film'))
                self.layers.append(tfc.GDN(name=f'gdn_{i}'))

            for j in range(self.num_postproc):
                self.layers.append(tfc.SignalConv2D(self.num_filters, (5, 5), name=f'layer_{i}_postproc_{j}',
                                                    corr=True, padding='same_zeros', use_bias=True, activation=None))
                if self.use_FiLM:
                    self.layers.append(FiLM(width=self.FiLM_width, depth=self.FiLM_depth,
                                            activation=self.FiLM_activation,
                                            name=f'layer_{i}_postproc_{j}_film'))
                if j != self.num_postproc - 1 or i != self.depth - 1:
                    self.layers.append(tfc.GDN(name=f'gdn_{i}_postproc_{j}'))

    def build(self, input_shape):
        parameters_shape, feature_maps_shape = input_shape

        for layer in self.layers:
            if isinstance(layer, FiLM):
                feature_maps_shape = layer.compute_output_shape([parameters_shape, feature_maps_shape])
            else:
                feature_maps_shape = layer.compute_output_shape(feature_maps_shape)

    def call(self, input_tensor):
        parameters, feature_maps = input_tensor

        for layer in self.layers:
            if isinstance(layer, FiLM):
                feature_maps = layer([parameters, feature_maps])
            else:
                feature_maps = layer(feature_maps)

        return feature_maps


class SynthesisTransform(tfkl.Layer):
    """The synthesis transform."""

    def __init__(self, num_filters, depth, num_postproc, FiLM_depth=None, FiLM_width=None,
                 FiLM_activation=None, *args,
                 **kwargs):
        super(SynthesisTransform, self).__init__(*args, **kwargs)

        self.num_filters = num_filters
        self.depth = depth
        self.num_postproc = num_postproc
        self.use_FiLM = FiLM_depth
        self.FiLM_width = FiLM_width
        self.FiLM_depth = FiLM_depth
        self.FiLM_activation = FiLM_activation

        self.layers = []
        for i in range(self.depth):
            self.layers.append(tfc.SignalConv2D(
                self.num_filters if i != self.depth - 1 or self.num_postproc else 3, (5, 5), name=f"layer_{i}",
                corr=False,
                strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=None))
            if i != self.depth - 1 or self.num_postproc:
                if self.use_FiLM:
                    self.layers.append(FiLM(width=self.FiLM_width, depth=self.FiLM_depth,
                                            activation=self.FiLM_activation,
                                            name=f'layer_{i}_film'))
                self.layers.append(tfc.GDN(name=f'igdn_{i}', inverse=True))

            for j in range(self.num_postproc):
                self.layers.append(
                    tfc.SignalConv2D(self.num_filters if i != self.depth - 1 or j != self.num_postproc - 1 else 3,
                                     (5, 5), name=f'layer_{i}_postproc_{j}',
                                     corr=False, padding='same_zeros', use_bias=True, activation=None))
                if self.use_FiLM:
                    self.layers.append(FiLM(width=self.FiLM_width, depth=self.FiLM_depth,
                                            activation=self.FiLM_activation,
                                            name=f'layer_{i}_postproc_{j}_film'))

                if j != self.num_postproc - 1 or i != self.depth - 1:
                    self.layers.append(tfc.GDN(name=f'igdn_{i}_postproc_{j}', inverse=True))

    def build(self, input_shape):
        parameters_shape, feature_maps_shape = input_shape

        for layer in self.layers:
            if isinstance(layer, FiLM):
                feature_maps_shape = layer.compute_output_shape([parameters_shape, feature_maps_shape])
            else:
                feature_maps_shape = layer.compute_output_shape(feature_maps_shape)

    def call(self, input_tensor):
        parameters, feature_maps = input_tensor

        for layer in self.layers:
            if isinstance(layer, FiLM):
                feature_maps = layer([parameters, feature_maps])
            else:
                feature_maps = layer(feature_maps)

        return feature_maps


class SimpleFiLMCompressor(tfk.Model):
    def __init__(self, num_filters, depth, num_postproc, FiLM_depth, FiLM_width, FiLM_activation, **kwargs):
        super(SimpleFiLMCompressor, self).__init__(**kwargs)

        self.use_FiLM = FiLM_depth
        self.analysis_transform = AnalysisTransform(num_filters, depth, num_postproc,
                                                    FiLM_depth=FiLM_depth,
                                                    FiLM_width=FiLM_width, FiLM_activation=FiLM_activation)
        self.synthesis_transform = SynthesisTransform(num_filters, depth, num_postproc,
                                                      FiLM_depth=FiLM_depth,
                                                      FiLM_width=FiLM_width, FiLM_activation=FiLM_activation)
        self.entropy_bottleneck = tfc.EntropyBottleneck()

    def forward(self, item, training):
        parameters = tf.stack([item['alpha'], item['lambda']], axis=-1)
        Y = self.analysis_transform([parameters, item['X']])
        Y_tilde, Y_likelihoods = self.entropy_bottleneck(Y, training=training)
        X_tilde = self.synthesis_transform([parameters, Y_tilde])

        results = {'Y': Y,
                   'Y_tilde': Y_tilde,
                   'Y_likelihoods': Y_likelihoods,
                   'X_tilde': X_tilde}

        return results

    def forward_with_range_coding(self, item):
        parameters = tf.stack([item['alpha'], item['lambda']], axis=-1)
        Y = self.analysis_transform([parameters, item['X']])
        Y_range_coded = self.entropy_bottleneck.compress(Y)
        Y_decoded = self.entropy_bottleneck.decompress(Y_range_coded, shape=tf.shape(Y)[1:3],
                                                       channels=tf.shape(Y)[3])
        X_reconstructed = self.synthesis_transform([parameters, Y_decoded])
        height = tf.shape(item['X'])[1]
        width = tf.shape(item['X'])[2]
        range_coded_bpp = tf.strings.length(Y_range_coded) * 8 / (height * width)

        return {
            'X_reconstructed': X_reconstructed,
            'range_coded_bpp': range_coded_bpp
        }


def bits_per_pixel(Y_likelihoods, X_shape):
    num_pixels = tf.cast(X_shape[1] * X_shape[2], tf.float32)
    return tf.reduce_sum(tf.log(Y_likelihoods), axis=[1, 2, 3]) / (-tf.math.log(2.) * num_pixels)


def pipeline_add_sampled_parameters(dataset, alpha_range, lambda_range, sample_fn):
    return dataset.map(lambda X, label: {
        'X': X,
        'label': label,
        'alpha': sample_fn(tf.shape(X)[0], alpha_range),
        'lambda': sample_fn(tf.shape(X)[0], lambda_range)
    })


def pipeline_add_constant_parameters(item, alpha, lmbda):
    repeat = lambda value, X: tf.repeat(value, repeats=tf.shape(X)[0])

    item.update({
        'alpha': repeat(alpha, item['X']),
        'lambda': repeat(lmbda, item['X'])
    })

    return item


def pipeline_add_range_of_parameters(item, alpha_linspace, lambda_linspace):
    augmented_items = {k: [] for k in list(item.keys()) + ['alpha', 'lambda']}

    for alpha in alpha_linspace:
        for lmbda in lambda_linspace:
            augmented = pipeline_add_constant_parameters(item, alpha, lmbda)
            for k, value_list in augmented_items.items():
                value_list.append(augmented[k])

    return augmented_items
