import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
import numpy as np

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


class HyperAnalysisTransform(tf.keras.layers.Layer):
    """The analysis transform for the entropy model parameters."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(HyperAnalysisTransform, self).__init__(*args, **kwargs)

        self.layers = [
            tfc.SignalConv2D(
                self.num_filters, (3, 3), name="layer_0", corr=True, strides_down=1,
                padding="same_zeros", use_bias=True,
                activation=tf.nn.relu),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tf.nn.relu),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
                padding="same_zeros", use_bias=False,
                activation=None),
        ]

    def call(self, tensor):
        for layer in self.layers:
            tensor = layer(tensor)
        return tensor


class HyperSynthesisTransform(tf.keras.layers.Layer):
    """The synthesis transform for the entropy model parameters."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(HyperSynthesisTransform, self).__init__(*args, **kwargs)

        self.layers = [
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True, kernel_parameterizer=None,
                activation=tf.nn.relu),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True, kernel_parameterizer=None,
                activation=tf.nn.relu),
            tfc.SignalConv2D(
                self.num_filters, (3, 3), name="layer_2", corr=False, strides_up=1,
                padding="same_zeros", use_bias=True, kernel_parameterizer=None,
                activation=None),
        ]

    def call(self, tensor):
        for layer in self.layers:
            tensor = layer(tensor)
        return tensor



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
        Y = self.analysis_transform([0, item['X']])
        Y_tilde, Y_likelihoods = self.entropy_bottleneck(Y, training=training)
        X_tilde = self.synthesis_transform([0, Y_tilde])

        results = {'Y': Y,
                   'Y_tilde': Y_tilde,
                   'Y_likelihoods': Y_likelihoods,
                   'X_tilde': X_tilde}

        return results

    def forward_with_range_coding(self, item):
        Y = self.analysis_transform([0, item['X']])
        Y_range_coded = self.entropy_bottleneck.compress(Y)
        Y_decoded = self.entropy_bottleneck.decompress(Y_range_coded, shape=tf.shape(Y)[1:3],
                                                       channels=tf.shape(Y)[3])
        X_reconstructed = self.synthesis_transform([0, Y_decoded])
        height = tf.shape(item['X'])[1]
        width = tf.shape(item['X'])[2]
        range_coded_bpp = tf.strings.length(Y_range_coded) * 8 / (height * width)

        return {
            'X_reconstructed': X_reconstructed,
            'range_coded_bpp': range_coded_bpp
        }


class HyperpriorCompressor(tfk.Model):
    SCALES_MIN = 0.11
    SCALES_MAX = 256
    SCALES_LEVELS = 64

    def __init__(self, num_filters, **kwargs):
        super(HyperpriorCompressor, self).__init__(**kwargs)
        self.analysis_transform = AnalysisTransform(num_filters, depth=4, num_postproc=0)
        self.synthesis_transform = SynthesisTransform(num_filters, depth=4, num_postproc=0)
        self.hyper_analysis_transform = HyperAnalysisTransform(num_filters)
        self.hyper_synthesis_transform = HyperSynthesisTransform(num_filters)
        self.entropy_bottleneck = tfc.EntropyBottleneck()
        self.num_filters = num_filters

    def forward(self, item, training):
        Y = self.analysis_transform([0, item['X']])
        Z = self.hyper_analysis_transform(tf.abs(Y))
        Z_tilde, Z_likelihoods = self.entropy_bottleneck(Z, training=training)
        sigma = self.hyper_synthesis_transform(Z_tilde)
        scale_table = np.exp(np.linspace(np.log(self.SCALES_MIN), np.log(self.SCALES_MAX), self.SCALES_LEVELS))
        conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table)
        Y_tilde, Y_likelihoods = conditional_bottleneck(Y, training=training)
        X_tilde = self.synthesis_transform([0, Y_tilde])

        return {
            'Y': Y,
            'Y_tilde': Y_tilde,
            'Y_likelihoods': Y_likelihoods,
            'Z': Z,
            'Z_tilde': Z_tilde,
            'Z_likelihoods': Z_likelihoods,
            'X_tilde': X_tilde
        }

    def forward_with_range_coding(self, item):
        Y = self.analysis_transform([0, item['X']])
        Y_shape = tf.shape(Y)
        Z = self.hyper_analysis_transform(tf.abs(Y))
        Z_hat, Z_likelihoods = self.entropy_bottleneck(Z, training=False)
        sigma = self.hyper_synthesis_transform(Z_hat)
        sigma = sigma[:, :Y_shape[1], :Y_shape[2], :]
        scale_table = np.exp(np.linspace(np.log(self.SCALES_MIN), np.log(self.SCALES_MAX), self.SCALES_LEVELS))
        conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table, dtype=tf.float32)
        Z_coded = self.entropy_bottleneck.compress(Z)
        Y_coded = conditional_bottleneck.compress(Y)

        height = tf.shape(item['X'])[1]
        width = tf.shape(item['X'])[2]
        coded_bpp = (tf.strings.length(Z_coded) + tf.strings.length(Y_coded)) * 8 / (height * width)

        Z_hat = self.entropy_bottleneck.decompress(Z_coded, tf.shape(Z)[1:3], channels=tf.shape(Y)[3])
        sigma = self.hyper_synthesis_transform(Z_hat)
        sigma = sigma[:, :Y_shape[1], :Y_shape[2], :]
        Y_hat = conditional_bottleneck.decompress(Y_coded)
        X_hat = self.synthesis_transform([0, Y_hat])

        return {
            'X_reconstructed': X_hat,
            'range_coded_bpp': coded_bpp
        }



def bits_per_pixel(item, X_shape):
    num_pixels = tf.cast(X_shape[1] * X_shape[2], tf.float32)
    bpp = tf.reduce_sum(tf.log(item['Y_likelihoods']), axis=[1, 2, 3])
    if 'Z_likelihoods' in item:
        bpp += tf.reduce_sum(tf.log(item['Z_likelihoods']), axis=[1, 2, 3])

    return bpp / (-tf.math.log(2.) * num_pixels)


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
