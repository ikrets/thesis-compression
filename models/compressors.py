import copy
import numpy as np
from pathlib import Path

from visualization.tensorboard_logging import Logger
from visualization.tensorboard import original_reconstruction_comparison

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

    def __init__(self, num_filters, depth, use_FiLM, FiLM_depth=None, FiLM_width=None, FiLM_activation=None, *args,
                 **kwargs):
        super(AnalysisTransform, self).__init__(*args, **kwargs)

        self.num_filters = num_filters
        self.depth = depth

        self.convs = [
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name=f"layer_{i}", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name=f"gdn_{i}")) for i in range(self.depth)]

        self.use_FiLM = use_FiLM
        if self.use_FiLM:
            if not FiLM_width or not FiLM_depth or not FiLM_activation:
                raise ValueError('If use_FiLM is set, other FiLM parameters must be set too!')

            self.FiLM_width = FiLM_width
            self.FiLM_depth = FiLM_depth
            self.FiLM_activation = FiLM_activation

            self.films = [FiLM(width=self.FiLM_width, depth=self.FiLM_depth,
                               activation=self.FiLM_activation if i != self.depth - 1 else None,
                               name=f'layer_{i}_film') for i in range(self.depth)]

    def build(self, input_shape):
        if not self.use_FiLM:
            feature_maps_shape = input_shape
        else:
            parameters_shape, feature_maps_shape = input_shape

        for i in range(self.depth):
            feature_maps_shape = self.convs[i].compute_output_shape(feature_maps_shape)

            if self.use_FiLM:
                feature_maps_shape = self.films[i].compute_output_shape([parameters_shape, feature_maps_shape])

    def call(self, input_tensor):
        if not self.use_FiLM:
            feature_maps = input_tensor
        else:
            parameters, feature_maps = input_tensor

        for i in range(self.depth):
            feature_maps = self.convs[i](feature_maps)
            if self.use_FiLM:
                feature_maps = self.films[i]([parameters, feature_maps])

        return feature_maps


class SynthesisTransform(tfkl.Layer):
    """The synthesis transform."""

    def __init__(self, num_filters, depth, use_FiLM, FiLM_depth=None, FiLM_width=None, FiLM_activation=None, *args,
                 **kwargs):
        super(SynthesisTransform, self).__init__(*args, **kwargs)

        self.num_filters = num_filters
        self.depth = depth
        self.use_FiLM = use_FiLM

        self.convs = [
            tfc.SignalConv2D(
                self.num_filters if i != self.depth - 1 else 3, (5, 5), name=f"layer_{i}", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name=f"igdn_{i}" if i != self.depth - 1 else None, inverse=True))
            for i in range(self.depth)
        ]

        if self.use_FiLM:
            if not FiLM_width or not FiLM_depth or not FiLM_activation:
                raise ValueError('If use_FiLM is set, other FiLM parameters must be set too!')

            self.FiLM_width = FiLM_width
            self.FiLM_depth = FiLM_depth
            self.FiLM_activation = FiLM_activation

            self.films = [
                FiLM(width=self.FiLM_width, depth=self.FiLM_depth,
                     activation=self.FiLM_activation if i != self.depth - 1 else None,
                     name=f'layer_{i}_film')
                for i in range(self.depth)]

    def build(self, input_shape):
        if self.use_FiLM:
            parameters_shape, feature_maps_shape = input_shape
        else:
            feature_maps_shape = input_shape

        for i in range(self.depth):
            feature_maps_shape = self.convs[i].compute_output_shape(feature_maps_shape)
            if self.use_FiLM:
                feature_maps_shape = self.films[i].compute_output_shape([parameters_shape, feature_maps_shape])

    def call(self, input_tensor):
        if not self.use_FiLM:
            feature_maps = input_tensor
        else:
            parameters, feature_maps = input_tensor

        for i in range(self.depth):
            feature_maps = self.convs[i](feature_maps)
            if self.use_FiLM:
                feature_maps = self.films[i]([parameters, feature_maps])

        return feature_maps


class SimpleFiLMCompressor(tfk.Model):
    def __init__(self, num_filters, depth, FiLM_depth, FiLM_width, FiLM_activation, **kwargs):
        super(SimpleFiLMCompressor, self).__init__(**kwargs)

        self.analysis_transform = AnalysisTransform(num_filters, depth, use_FiLM=True, FiLM_depth=FiLM_depth,
                                                    FiLM_width=FiLM_width, FiLM_activation=FiLM_activation)
        self.synthesis_transform = SynthesisTransform(num_filters, depth, use_FiLM=True, FiLM_depth=FiLM_depth,
                                                      FiLM_width=FiLM_width, FiLM_activation=FiLM_activation)
        self.entropy_bottleneck = tfc.EntropyBottleneck()

    def forward(self, parameters, X, training):
        Y = self.analysis_transform([parameters, X])
        Y_tilde, Y_likelihoods = self.entropy_bottleneck(Y, training=training)
        X_tilde = self.synthesis_transform([parameters, Y_tilde])

        return {'Y_tilde': Y_tilde,
                'Y_likelihoods': Y_likelihoods,
                'X_tilde': X_tilde}


def bits_per_pixel(Y_likelihoods, X_shape):
    num_pixels = tf.cast(X_shape[1] * X_shape[2], tf.float32)
    return tf.reduce_sum(tf.log(Y_likelihoods), axis=[1, 2, 3]) / (-tf.math.log(2.) * num_pixels)


class CompressorWithDownstreamComparison:
    def __init__(self,
                 compressor,
                 downstream_model,
                 downstream_metric,
                 downstream_compressed_vs_uncompressed_layer,
                 downstream_compressed_vs_uncompressed_loss,
                 reverse_normalize):
        self.compressor = compressor
        self.downstream_model = downstream_model
        self.downstream_metric = downstream_metric

        downstream_compressed_vs_uncompressed_layer = self.downstream_model.get_layer(
            downstream_compressed_vs_uncompressed_layer)
        self.downstream_model = tfk.Model(
            inputs=self.downstream_model.input, outputs=[self.downstream_model.output,
                                                         downstream_compressed_vs_uncompressed_layer.output])

        self.downstream_compressed_vs_uncompressed_loss = downstream_compressed_vs_uncompressed_loss
        self.reverse_normalize = reverse_normalize

    def set_optimizers(self, main_optimizer, aux_optimizer):
        self.main_optimizer = main_optimizer
        self.aux_optimizer = aux_optimizer

    def _get_outputs_losses_metrics(self, dataset, training):
        batch = dataset.make_one_shot_iterator().get_next()
        stacked_parameters = tf.stack([batch['alpha'], batch['lambda']], axis=-1)
        compressor_outputs = self.compressor.forward(parameters=stacked_parameters, X=batch['X'], training=training)
        mse = tf.reduce_mean(tf.math.squared_difference(batch['X'], compressor_outputs['X_tilde']), axis=[1, 2, 3])
        bpp = bits_per_pixel(compressor_outputs['Y_likelihoods'], tf.shape(batch['X']))
        psnr = tf.image.psnr(self.reverse_normalize(batch['X']),
                             self.reverse_normalize(compressor_outputs['X_tilde']),
                             max_val=1.)

        _, uncompressed_comparison = self.downstream_model(batch['X'])
        compressed_preds, compressed_comparison = self.downstream_model(compressor_outputs['X_tilde'])
        downstream_comparison = self.downstream_compressed_vs_uncompressed_loss(uncompressed_comparison,
                                                                                compressed_comparison)
        compressed_metric = self.downstream_metric(batch['label'], compressed_preds)

        reconstruction_loss = (1 - batch['alpha']) * mse + batch['alpha'] * downstream_comparison
        reconstruction_loss *= (255 ** 2)

        total = batch['lambda'] * reconstruction_loss + bpp

        return {'mse': tf.reduce_mean(mse),
                'bpp': tf.reduce_mean(bpp),
                'reconstruction': tf.reduce_mean(reconstruction_loss),
                'total': tf.reduce_mean(total),
                'psnr': tf.reduce_mean(psnr),
                'downstream_comparison': tf.reduce_mean(downstream_comparison),
                'downstream_metric_on_compressed': tf.reduce_mean(compressed_metric),
                'X': batch['X'],
                'alpha': batch['alpha'],
                'lambda': batch['lambda'],
                'X_tilde': compressor_outputs['X_tilde']}

    def _reset_accumulators(self):
        self.train_metrics = {'mse': [],
                              'bpp': [],
                              'reconstruction': [],
                              'total': [],
                              'psnr': [],
                              'downstream_comparison': [],
                              'alpha': [],
                              'lambda': []}
        self.val_metrics = copy.deepcopy(self.train_metrics)

    def _accumulate(self, results, training):
        metrics = self.train_metrics if training else self.val_metrics

        for k in metrics.keys():
            metrics[k].append(results[k])

    def _log_accumulated(self, logger, epoch, training):
        metrics = self.train_metrics if training else self.val_metrics
        for key, value_list in metrics.items():
            if key not in ['alpha', 'lambda']:
                logger.log_scalar(key, np.mean(value_list), step=epoch)
            else:
                logger.log_histogram(key, np.concatenate(value_list), step=epoch)

    def _get_optimizer_variables(self):
        variables = []
        for opt in [self.main_optimizer, self.aux_optimizer]:
            variables.append(opt.variables())
            if hasattr(opt, '_optimizer'):
                variables.append(opt._optimizer.variables())

        return variables

    def fit(self, sess,
            dataset, dataset_steps,
            val_dataset, val_dataset_steps,
            epochs,
            log_dir,
            log_period,
            checkpoint_period):
        train_outputs_losses_metrics = self._get_outputs_losses_metrics(dataset, training=True)
        val_outputs_losses_metrics = self._get_outputs_losses_metrics(val_dataset, training=False)

        non_downstream_variables = [v for v in tf.global_variables() if v not in self.downstream_model.variables]

        main_step = self.main_optimizer.minimize(train_outputs_losses_metrics['total'],
                                                 var_list=non_downstream_variables)
        aux_step = self.aux_optimizer.minimize(self.compressor.entropy_bottleneck.losses[0])
        sess.run(tf.variables_initializer(
            non_downstream_variables + self.main_optimizer.variables() + self.aux_optimizer.variables()))

        train_steps = tf.group([main_step, aux_step, self.compressor.entropy_bottleneck.updates[0]])

        train_logger = Logger(Path(log_dir) / 'train')
        val_logger = Logger(Path(log_dir) / 'val')

        for epoch in range(epochs):
            self._reset_accumulators()

            for train_step in range(dataset_steps):
                train_results, _ = sess.run([train_outputs_losses_metrics, train_steps])
                self._accumulate(train_results, training=True)

                if train_step == 0 and epoch % log_period == 0:
                    comparison = original_reconstruction_comparison(
                        self.reverse_normalize(train_results['X']),
                        self.reverse_normalize(train_results['X_tilde']),
                        size=10,
                        alphas=train_results['alpha'],
                        lambdas=train_results['lambda'])
                    train_logger.log_image('original_vs_reconstruction', comparison, step=epoch)

            self._log_accumulated(train_logger, epoch=epoch, training=True)

            for val_step in range(val_dataset_steps):
                val_results = sess.run(val_outputs_losses_metrics)
                self._accumulate(val_results, training=False)
                if val_step == 0 and epoch % log_period == 0:
                    comparison = original_reconstruction_comparison(
                        self.reverse_normalize(val_results['X']),
                        self.reverse_normalize(val_results['X_tilde']),
                        size=10,
                        alphas=val_results['alpha'],
                        lambdas=val_results['lambda'])
                    val_logger.log_image('original_vs_reconstruction', comparison, step=epoch)

            self._log_accumulated(val_logger, epoch=epoch, training=False)

            if epoch % checkpoint_period == 0:
                self.compressor.save_weights(str(log_dir / f'compressor_epoch_{epoch}_weights.h5'))


def pipeline_add_parameters(dataset, alpha_range, lambda_range):
    sample_loguniform = lambda len, range: tf.math.exp(tf.random.uniform([len], minval=tf.math.log(range[0]),
                                                                         maxval=tf.math.log(range[1])))
    return dataset.map(lambda X, label: {
        'X': X,
        'label': label,
        'alpha': sample_loguniform(tf.shape(X)[0], alpha_range),
        'lambda': sample_loguniform(tf.shape(X)[0], lambda_range)
    })
