import copy
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange, tqdm

from visualization.tensorboard_logging import Logger
from visualization.tensorboard import original_reconstruction_comparison
from visualization.plots import rate_distortion_curve, figure_to_numpy

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

    def __init__(self, num_filters, depth, num_postproc, use_FiLM, FiLM_depth=None, FiLM_width=None,
                 FiLM_activation=None, *args,
                 **kwargs):
        super(AnalysisTransform, self).__init__(*args, **kwargs)

        self.num_filters = num_filters
        self.depth = depth
        self.num_postproc = num_postproc

        self.use_FiLM = use_FiLM
        self.FiLM_width = FiLM_width
        self.FiLM_depth = FiLM_depth
        self.FiLM_activation = FiLM_activation
        if self.use_FiLM and not FiLM_width or not FiLM_depth or not FiLM_activation:
            raise ValueError('If use_FiLM is set, other FiLM parameters must be set too!')

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
        if not self.use_FiLM:
            feature_maps_shape = input_shape
        else:
            parameters_shape, feature_maps_shape = input_shape

        for layer in self.layers:
            if isinstance(layer, FiLM):
                feature_maps_shape = layer.compute_output_shape([parameters_shape, feature_maps_shape])
            else:
                feature_maps_shape = layer.compute_output_shape(feature_maps_shape)

    def call(self, input_tensor):
        if not self.use_FiLM:
            feature_maps = input_tensor
        else:
            parameters, feature_maps = input_tensor

        for layer in self.layers:
            if isinstance(layer, FiLM):
                feature_maps = layer([parameters, feature_maps])
            else:
                feature_maps = layer(feature_maps)

        return feature_maps


class SynthesisTransform(tfkl.Layer):
    """The synthesis transform."""

    def __init__(self, num_filters, depth, num_postproc, use_FiLM, FiLM_depth=None, FiLM_width=None,
                 FiLM_activation=None, *args,
                 **kwargs):
        super(SynthesisTransform, self).__init__(*args, **kwargs)

        self.num_filters = num_filters
        self.depth = depth
        self.num_postproc = num_postproc
        self.use_FiLM = use_FiLM
        if not FiLM_width or not FiLM_depth or not FiLM_activation:
            raise ValueError('If use_FiLM is set, other FiLM parameters must be set too!')

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
        if not self.use_FiLM:
            feature_maps_shape = input_shape
        else:
            parameters_shape, feature_maps_shape = input_shape

        for layer in self.layers:
            if isinstance(layer, FiLM):
                feature_maps_shape = layer.compute_output_shape([parameters_shape, feature_maps_shape])
            else:
                feature_maps_shape = layer.compute_output_shape(feature_maps_shape)

    def call(self, input_tensor):
        if not self.use_FiLM:
            feature_maps = input_tensor
        else:
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

        self.analysis_transform = AnalysisTransform(num_filters, depth, num_postproc, use_FiLM=True,
                                                    FiLM_depth=FiLM_depth,
                                                    FiLM_width=FiLM_width, FiLM_activation=FiLM_activation)
        self.synthesis_transform = SynthesisTransform(num_filters, depth, num_postproc, use_FiLM=True,
                                                      FiLM_depth=FiLM_depth,
                                                      FiLM_width=FiLM_width, FiLM_activation=FiLM_activation)
        self.entropy_bottleneck = tfc.EntropyBottleneck()

    def forward(self, item, training, run_range_coder=False):
        parameters = tf.stack([item['alpha'], item['lambda']], axis=-1)
        Y = self.analysis_transform([parameters, item['X']])
        Y_tilde, Y_likelihoods = self.entropy_bottleneck(Y, training=training)
        X_tilde = self.synthesis_transform([parameters, Y_tilde])

        results = {'Y_tilde': Y_tilde,
                   'Y_likelihoods': Y_likelihoods,
                   'X_tilde': X_tilde}

        if run_range_coder:
            results['Y_compressed'] = self.entropy_bottleneck.compress(Y)

        return results


def bits_per_pixel(Y_likelihoods, X_shape):
    num_pixels = tf.cast(X_shape[1] * X_shape[2], tf.float32)
    return tf.reduce_sum(tf.log(Y_likelihoods), axis=[1, 2, 3]) / (-tf.math.log(2.) * num_pixels)


class CompressorWithDownstreamLoss:
    def __init__(self,
                 compressor,
                 downstream_model,
                 downstream_metric,
                 downstream_compressed_vs_uncompressed_layer,
                 downstream_compressed_vs_uncompressed_loss,
                 downstream_preprocess):
        self.compressor = compressor
        self.downstream_model = downstream_model
        self.downstream_metric = downstream_metric
        self.downstream_preprocess = downstream_preprocess

        downstream_compressed_vs_uncompressed_layer = self.downstream_model.get_layer(
            downstream_compressed_vs_uncompressed_layer)
        self.downstream_model = tfk.Model(
            inputs=self.downstream_model.input, outputs=[self.downstream_model.output,
                                                         downstream_compressed_vs_uncompressed_layer.output])

        self.downstream_compressed_vs_uncompressed_loss = downstream_compressed_vs_uncompressed_loss

    def set_optimizers(self, main_optimizer, aux_optimizer, main_lr, main_schedule):
        self.main_lr = main_lr

        self.main_schedule = main_schedule

        self.main_optimizer = main_optimizer
        self.aux_optimizer = aux_optimizer

    def _get_outputs_losses_metrics(self, dataset, training):
        batch = dataset.make_one_shot_iterator().get_next()

        compressor_outputs = self.compressor.forward(batch, training)
        clipped_X_tilde = tf.clip_by_value(compressor_outputs['X_tilde'], 0, 1)

        mse = tf.reduce_mean(tf.math.squared_difference(batch['X'], clipped_X_tilde), axis=[1, 2, 3])
        bpp = bits_per_pixel(compressor_outputs['Y_likelihoods'], tf.shape(batch['X']))
        psnr = tf.image.psnr(batch['X'], clipped_X_tilde, max_val=1.)

        _, uncompressed_comparison = self.downstream_model(self.downstream_preprocess(batch['X']))
        compressed_preds, compressed_comparison = self.downstream_model(self.downstream_preprocess(clipped_X_tilde))
        downstream_loss = self.downstream_compressed_vs_uncompressed_loss(uncompressed_comparison,
                                                                          compressed_comparison)
        compressed_metric = self.downstream_metric(batch['label'], compressed_preds)

        total = batch['lambda'] * mse * (255 ** 2) + batch['alpha'] * downstream_loss * (255 ** 2) + bpp

        return {'mse': mse,
                'bpp': bpp,
                'total': total,
                'psnr': psnr,
                'downstream_loss': downstream_loss,
                'downstream_metric': compressed_metric,
                'X': batch['X'],
                'alpha': batch['alpha'],
                'lambda': batch['lambda'],
                'X_tilde': clipped_X_tilde}

    def _reset_accumulators(self):
        self.train_metrics = {'mse': [],
                              'bpp': [],
                              'total': [],
                              'psnr': [],
                              'downstream_loss': [],
                              'downstream_metric': [],
                              'alpha': [],
                              'lambda': []}
        self.val_metrics = copy.deepcopy(self.train_metrics)

    def _accumulate(self, results, training):
        metrics = self.train_metrics if training else self.val_metrics

        for k in metrics.keys():
            if np.any(np.isnan(results[k])):
                raise RuntimeError('NaN encountered in ${k}!')
            metrics[k].extend(results[k])

    def _log_accumulated(self, logger, epoch, training):
        metrics = self.train_metrics if training else self.val_metrics
        for key, value_list in metrics.items():
            if key not in ['alpha', 'lambda']:
                logger.log_scalar(key, np.mean(value_list), step=epoch)
            else:
                logger.log_histogram(key, value_list, step=epoch)

    def _get_optimizer_variables(self):
        variables = []
        for opt in [self.main_optimizer, self.aux_optimizer]:
            variables.extend(opt.variables())
            if hasattr(opt, '_optimizer'):
                variables.extend(opt._optimizer.variables())

        return variables

    # TODO feed separate random parameter dataset and zip it with the train and validation sets!
    def fit(self, sess,
            dataset, dataset_steps,
            random_parameter_val_dataset, val_dataset_steps,
            const_parameter_val_datasets,
            epochs,
            log_dir,
            train_log_period,
            val_log_period,
            checkpoint_period):
        main_lr_placeholder = tf.placeholder(dtype=tf.float32)
        assign_lr = tf.assign(self.main_lr, main_lr_placeholder)

        train_outputs_losses_metrics = self._get_outputs_losses_metrics(dataset, training=True)
        val_outputs_losses_metrics = self._get_outputs_losses_metrics(random_parameter_val_dataset, training=False)

        const_parameter_outputs = {parameters: self._get_outputs_losses_metrics(d, training=False)
                                   for parameters, d in const_parameter_val_datasets.items()}

        non_downstream_variables = [v for v in tf.global_variables() if v not in self.downstream_model.variables]

        main_step = self.main_optimizer.minimize(tf.reduce_mean(train_outputs_losses_metrics['total']),
                                                 var_list=non_downstream_variables)
        aux_step = self.aux_optimizer.minimize(self.compressor.entropy_bottleneck.losses[0])
        sess.run(tf.variables_initializer(non_downstream_variables + self._get_optimizer_variables()))

        train_steps = tf.group([main_step, aux_step, self.compressor.entropy_bottleneck.updates[0]])

        train_logger = Logger(Path(log_dir) / 'train')
        val_logger = Logger(Path(log_dir) / 'val')

        for epoch in trange(epochs, desc='epoch'):
            self._reset_accumulators()
            sess.run(assign_lr, feed_dict={main_lr_placeholder: self.main_schedule(epoch)})
            train_logger.log_scalar('main_lr', self.main_schedule(epoch), step=epoch)

            for train_step in trange(dataset_steps, desc='train batch'):
                train_results, _ = sess.run([train_outputs_losses_metrics, train_steps])

                if epoch % train_log_period == 0:
                    self._accumulate(train_results, training=True)

                    if train_step == 0:
                        comparison = original_reconstruction_comparison(
                            train_results['X'],
                            train_results['X_tilde'],
                            size=10,
                            alphas=train_results['alpha'],
                            lambdas=train_results['lambda'])
                        train_logger.log_image('original_vs_reconstruction', comparison, step=epoch)

            if epoch % train_log_period == 0 and epoch:
                self._log_accumulated(train_logger, epoch=epoch, training=True)

            if epoch % val_log_period == 0 and epoch:
                for val_step in trange(val_dataset_steps, desc='val batch with random parameters'):
                    val_results = sess.run(val_outputs_losses_metrics)
                    self._accumulate(val_results, training=False)

                    if val_step == 0:
                        comparison = original_reconstruction_comparison(
                            val_results['X'],
                            val_results['X_tilde'],
                            size=10,
                            alphas=val_results['alpha'],
                            lambdas=val_results['lambda'])
                        val_logger.log_image('original_vs_reconstruction', comparison, step=epoch)

                self._log_accumulated(val_logger, epoch=epoch, training=False)

                evaluations = {
                    'alpha': [],
                    'lambda': [],
                    'bpp': [],
                    'psnr': [],
                    'downstream_loss': [],
                    'downstream_metric': []
                }
                for (alpha, lmbda), outputs in tqdm(const_parameter_outputs.items(), desc='val grid params dataset'):
                    psnrs = []
                    bpps = []
                    downstream_losses = []
                    downstream_metrics = []

                    for _ in trange(val_dataset_steps, desc='val grid batch'):
                        results = sess.run(outputs)
                        psnrs.extend(results['psnr'])
                        bpps.extend(results['bpp'])
                        downstream_losses.extend(results['downstream_loss'])
                        downstream_metrics.extend(results['downstream_metric'])

                    evaluations['alpha'].append(alpha)
                    evaluations['lambda'].append(lmbda)
                    evaluations['bpp'].append(np.mean(bpps))
                    evaluations['psnr'].append(np.mean(psnrs))
                    evaluations['downstream_loss'].append(np.mean(downstream_losses))
                    evaluations['downstream_metric'].append(np.mean(downstream_metrics))

                evaluation_df = pd.DataFrame(evaluations)

                plt_img = figure_to_numpy(rate_distortion_curve(evaluation_df, figsize=(12, 12),
                                                                downstream_metrics=['downstream_loss',
                                                                                    'downstream_metric']))
                val_logger.log_image('parameters_rate_distortion', plt_img, step=epoch)
                plt.close()

            if epoch % checkpoint_period == 0 and epoch:
                self.compressor.save_weights(str(log_dir / f'compressor_epoch_{epoch}_weights.h5'))

            train_logger.writer.flush()
            val_logger.writer.flush()


def pipeline_add_sampled_parameters(dataset, alpha_range, lambda_range, sample_fn):
    return dataset.map(lambda X, label: {
        'X': X,
        'label': label,
        'alpha': sample_fn(tf.shape(X)[0], alpha_range),
        'lambda': sample_fn(tf.shape(X)[0], lambda_range)
    })


def pipeline_add_constant_parameters(item, alpha, lmbda):
    repeat = lambda value, X: tf.tile(tf.constant([value], dtype=tf.float32), multiples=(tf.shape(X)[0],))

    item.update({
        'alpha': repeat(alpha, item['X']),
        'lambda': repeat(lmbda, item['X'])
    })

    return item
