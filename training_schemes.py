import copy
import json
import re
from pathlib import Path

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange, tqdm
from typing import Tuple, Callable, Dict, Sequence, Union, Optional, Any

import datasets
from models.bpp_range import BppRangeEvaluation, area_under_bpp_metric, BppRangeAdapter, LogarithmicOrLinearFit
from models.compressors import SimpleFiLMCompressor, bits_per_pixel
from models.downstream_losses import PerceptualLoss
from visualization.plots import alpha_comparison, figure_to_numpy, rate_distortion_curve
from visualization.tensorboard import original_reconstruction_comparison
from visualization.tensorboard_logging import Logger


class CompressorWithDownstreamLoss:
    def __init__(self,
                 compressor: SimpleFiLMCompressor,
                 downstream_loss: PerceptualLoss) -> None:
        self.compressor = compressor
        self.downstream_loss = downstream_loss

    def set_optimizers(self, main_optimizer, aux_optimizer, main_lr, main_schedule):
        self.main_lr = main_lr

        self.main_schedule = main_schedule

        self.main_optimizer = main_optimizer
        self.aux_optimizer = aux_optimizer

    def _get_outputs_losses_metrics(self, dataset, add_parameters_fn, training):
        batch = dataset.make_one_shot_iterator().get_next()
        batch = add_parameters_fn(batch)

        compressor_outputs = self.compressor.forward(batch, training)
        clipped_X_tilde = tf.clip_by_value(compressor_outputs['X_tilde'], 0, 1)

        mse = tf.reduce_mean(tf.math.squared_difference(batch['X'], clipped_X_tilde), axis=[1, 2, 3])
        bpp = bits_per_pixel(compressor_outputs['Y_likelihoods'], tf.shape(batch['X']))
        psnr = tf.image.psnr(batch['X'], clipped_X_tilde, max_val=1.)

        downstream_loss = self.downstream_loss.loss(X=batch['X'], X_reconstruction=clipped_X_tilde)
        compressed_metric = self.downstream_loss.metric(label=batch['label'], X_reconstruction=clipped_X_tilde)

        total = batch['lambda'] * mse * (255 ** 2) + batch['alpha'] * downstream_loss * (
                255 ** 2) + bpp

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
    def fit(self, dataset_setup: datasets.DatasetSetup, add_parameters_fn, epochs, log_dir,
            val_log_period,
            checkpoint_period, callbacks: Sequence[Callable[[int, tf.summary.FileWriter], None]] = ()) -> None:
        main_lr_placeholder = tf.placeholder(dtype=tf.float32)
        assign_lr = tf.assign(self.main_lr, main_lr_placeholder)

        train_outputs_losses_metrics = self._get_outputs_losses_metrics(
            dataset_setup.train_dataset,
            add_parameters_fn=add_parameters_fn,
            training=True)
        val_outputs_losses_metrics = self._get_outputs_losses_metrics(
            dataset_setup.val_dataset,
            add_parameters_fn=add_parameters_fn,
            training=False)

        sess = tf.keras.backend.get_session()
        already_initialized = sess.run({v: tf.is_variable_initialized(v) for v in tf.global_variables()})
        initialize_variables = [v for v in tf.global_variables() if
                                v not in self.downstream_loss.model.variables and not already_initialized[v]]
        optimize_variables = [v for v in tf.global_variables() if
                              v not in self.downstream_loss.model.variables and v.trainable]
        main_step = self.main_optimizer.minimize(tf.reduce_mean(train_outputs_losses_metrics['total']),
                                                 var_list=optimize_variables)
        aux_step = self.aux_optimizer.minimize(self.compressor.entropy_bottleneck.losses[0])
        train_steps = tf.group([main_step, aux_step, self.compressor.entropy_bottleneck.updates[0]])

        sess.run(tf.variables_initializer(initialize_variables + self._get_optimizer_variables()))

        train_logger = Logger(Path(log_dir) / 'train')
        val_logger = Logger(Path(log_dir) / 'val')

        for epoch in trange(epochs, desc='epoch'):
            self._reset_accumulators()
            sess.run(assign_lr, feed_dict={main_lr_placeholder: self.main_schedule(epoch)})
            train_logger.log_scalar('main_lr', self.main_schedule(epoch), step=epoch)

            for train_step in trange(dataset_setup.train_steps, desc='train batch'):
                train_results, _ = sess.run([train_outputs_losses_metrics, train_steps])

                self._accumulate(train_results, training=True)

                if train_step == 0:
                    comparison = original_reconstruction_comparison(
                        train_results['X'],
                        train_results['X_tilde'],
                        size=10,
                        alphas=train_results['alpha'],
                        lambdas=train_results['lambda'])
                    train_logger.log_image('original_vs_reconstruction', comparison, step=epoch)

            self._log_accumulated(train_logger, epoch=epoch, training=True)

            visualize_step = np.random.choice(np.arange(dataset_setup.val_steps))
            for val_step in trange(dataset_setup.val_steps, desc='val batch with random parameters'):
                val_results = sess.run(val_outputs_losses_metrics)
                self._accumulate(val_results, training=False)

                if val_step == visualize_step:
                    comparison = original_reconstruction_comparison(
                        val_results['X'],
                        val_results['X_tilde'],
                        size=10,
                        alphas=val_results['alpha'],
                        lambdas=val_results['lambda'])
                    val_logger.log_image('original_vs_reconstruction', comparison, step=epoch)

            self._log_accumulated(val_logger, epoch=epoch, training=False)
            for callback in callbacks:
                callback(epoch, val_logger)

            if epoch % checkpoint_period == 0 and epoch:
                self.compressor.save_weights(str(log_dir / f'compressor_epoch_{epoch}_weights.h5'))

            train_logger.writer.flush()
            val_logger.writer.flush()


def load_compressor_with_range(
        experiment_path: Union[str, Path],
        epoch: Optional[int] = None
) -> Tuple[SimpleFiLMCompressor, LogarithmicOrLinearFit, Dict[str, Any]]:
    experiment_path = Path(experiment_path)
    with (experiment_path / 'parameters.json').open('r') as fp:
        parameters = json.load(fp)

    model = SimpleFiLMCompressor(num_filters=parameters['num_filters'],
                                 depth=parameters['depth'],
                                 num_postproc=parameters['num_postproc'],
                                 FiLM_width=parameters['film_width'],
                                 FiLM_depth=parameters['film_depth'],
                                 FiLM_activation=parameters['film_activation'])
    model.forward(item={'X': np.zeros((128, 32, 32, 3)).astype(np.float32),
                        'alpha': np.zeros(128).astype(np.float32),
                        'lambda': np.zeros(128).astype(np.float32)},
                  training=False)

    weights = list(experiment_path.glob('compressor_epoch_*_weights.h5'))
    matches = [re.search('compressor_epoch_([0-9]+)_weights.h5', str(f)) for f in weights]
    epochs = [int(m.group(1)) for m in matches if m]

    if epoch is None:
        if not epochs:
            raise RuntimeError('No weights found!')

        epoch = max(epochs)

    print(f'Loading weights and alpha-bpp fit from epoch {epoch}')
    model.load_weights(str(experiment_path / f'compressor_epoch_{epoch}_weights.h5'))
    with (experiment_path / f'alpha_to_bpp_epoch_{epoch}.json').open('r') as fp:
        alpha_to_bpp = LogarithmicOrLinearFit.load(fp)

    return model, alpha_to_bpp, parameters
