# -*- coding: utf-8 -*-
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Nonlinear transform coder with hyperprior for RGB images.

This is the image compression model published in:
J. Ballé, D. Minnen, S. Singh, S.J. Hwang, N. Johnston:
"Variational Image Compression with a Scale Hyperprior"
Int. Conf. on Learning Representations (ICLR), 2018
https://arxiv.org/abs/1802.01436

This is meant as 'educational' code - you can use this to get started with your
own experiments. To reproduce the exact results from the paper, tuning of hyper-
parameters may be necessary. To compress images with published models, see
`tfci.py`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import sys

from absl import app
from absl.flags import argparse_flags
import numpy as np
import math
from coolname import generate_slug
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from experiment import save_experiment_params
from tensorboard.plugins.hparams import api as hp

import tensorflow_compression as tfc
import datasets.cifar10
from tqdm import trange

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def read_png(filename):
    """Loads a PNG image file."""
    string = tf.read_file(filename)
    image = tf.image.decode_image(string, channels=3)
    image = image[..., ::-1]
    image = tf.cast(image, tf.float32)
    image /= 255
    return image


def quantize_image(image):
    image = tf.round(image * 255)
    image = tf.saturate_cast(image, tf.uint8)
    return image


def write_png(filename, image):
    """Saves an image to a PNG file."""
    image = image[..., ::-1]
    image = quantize_image(image)
    string = tf.image.encode_png(image)
    return tf.write_file(filename, string)


class AnalysisTransform(tf.keras.layers.Layer):
    """The analysis transform."""

    def __init__(self, num_filters, kernel_size, steps, *args, **kwargs):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.steps = steps
        super(AnalysisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = []
        for i in range(self.steps):
            last_layer = i == self.steps - 1
            self._layers.append(
                tfc.SignalConv2D(
                    self.num_filters, self.kernel_size, name=f"layer_{i}", corr=True, strides_down=2,
                    padding="same_zeros", use_bias=True,
                    activation=tfc.GDN(name=f"gdn_{i}") if not last_layer else None))

        super(AnalysisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


class SynthesisTransform(tf.keras.layers.Layer):
    """The synthesis transform."""

    def __init__(self, num_filters, kernel_size, steps, *args, **kwargs):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.steps = steps

        super(SynthesisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = []

        for i in range(self.steps):
            last_layer = i == self.steps - 1
            self._layers.append(tfc.SignalConv2D(
                self.num_filters if not last_layer else 3,
                self.kernel_size, name=f"layer_{i}", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name=f"igdn_{i}", inverse=True) if not last_layer else None))

        super(SynthesisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


class HyperAnalysisTransform(tf.keras.layers.Layer):
    """The analysis transform for the entropy model parameters."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(HyperAnalysisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters, (3, 3), name="layer_0", corr=True, strides_down=1,
                padding="same_zeros", use_bias=True,
                activation=tf.nn.relu),
            tfc.SignalConv2D(
                self.num_filters, (3, 3), name="layer_1", corr=True, strides_down=2,
                padding="same_zeros", use_bias=False,
                activation=None),
        ]
        super(HyperAnalysisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


class HyperSynthesisTransform(tf.keras.layers.Layer):
    """The synthesis transform for the entropy model parameters."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(HyperSynthesisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters, (3, 3), name="layer_0", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True, kernel_parameterizer=None,
                activation=tf.nn.relu),
            tfc.SignalConv2D(
                self.num_filters, (3, 3), name="layer_1", corr=False, strides_up=1,
                padding="same_zeros", use_bias=True, kernel_parameterizer=None,
                activation=None),
        ]
        super(HyperSynthesisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


class Compressor(tf.Module):
    def __init__(self, hparams, model, **kwargs):
        super(Compressor, self).__init__(**kwargs)

        self.analysis_transform = AnalysisTransform(hparams['filters'], steps=hparams['steps'],
                                                    kernel_size=hparams['kernel_size'])
        self.synthesis_transform = SynthesisTransform(hparams['filters'], steps=hparams['steps'],
                                                      kernel_size=hparams['kernel_size'])

        if hparams['scale_hyperprior']:
            self.hyper_analysis_transform = HyperAnalysisTransform(hparams['filters'])
            self.hyper_synthesis_transform = HyperSynthesisTransform(hparams['filters'])

        self.entropy_bottleneck = tfc.EntropyBottleneck()
        self.model = model

        if hparams['downstream_loss_type'] != 'none':
            assert model is not None, 'A model should be provided when using downstream loss'

            self.downstream_layer = tf.keras.Model(inputs=model.input,
                                                   outputs=model.get_layer(hparams['downstream_loss_layer']).output)

            if hparams['downstream_loss_type'] == 'mse':
                self.downstream_loss = lambda X_compressed, X: tf.reduce_mean(tf.squared_difference(X_compressed, X))
            if hparams['downstream_loss_type'] == 'kld':
                self.downstream_loss = lambda X_compressed, X: tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(X_compressed, X))

        self.hparams = hparams
        self.fit_or_loaded = False

    def _losses(self, dataset, training=True):
        # Get training patch from dataset.
        normalized_x, label = dataset.make_one_shot_iterator().get_next()
        x = datasets.cifar10.normalize(normalized_x, inverse=True)
        num_pixels = tf.cast(tf.reduce_prod(tf.shape(normalized_x)[:3]), tf.float32)

        y = self.analysis_transform(x)
        if not self.hparams['scale_hyperprior']:
            y_tilde, y_likelihoods = self.entropy_bottleneck(y, training=training)
            x_tilde = self.synthesis_transform(y_tilde)
            normalized_x_tilde = datasets.cifar10.normalize(x_tilde)

            bpp = tf.reduce_sum(tf.log(y_likelihoods)) / (-np.log(2) * num_pixels)
        else:
            z = self.hyper_analysis_transform(abs(y))
            z_tilde, z_likelihoods = self.entropy_bottleneck(z, training=training)

            sigma = self.hyper_synthesis_transform(z_tilde)
            scale_table = np.exp(np.linspace(
                np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
            conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table)
            y_tilde, y_likelihoods = conditional_bottleneck(y, training=training)
            x_tilde = self.synthesis_transform(y_tilde)
            normalized_x_tilde = datasets.cifar10.normalize(x_tilde)

            bpp = (tf.reduce_sum(tf.log(y_likelihoods)) + tf.reduce_sum(tf.log(z_likelihoods))) / (
                    -np.log(2) * num_pixels)

        # Mean squared error across pixels.
        mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))

        if self.hparams['downstream_loss_type'] != 'none':
            downstream = self.downstream_loss(X_compressed=self.downstream_layer(normalized_x_tilde),
                                              X=self.downstream_layer(normalized_x))
            # Multiply by 255^2 to correct for rescaling.
            reconstruction = (1 - self.hparams['downstream_loss_alpha']) * mse
            reconstruction += self.hparams['downstream_loss_alpha'] * downstream
            reconstruction *= 255 ** 2
        else:
            downstream = tf.constant(0)
            reconstruction = mse * (255 ** 2)

        psnr = tf.reduce_mean(tf.image.psnr(x, x_tilde, max_val=1.))

        # The rate-distortion cost.
        total = self.hparams['lambda'] * reconstruction + bpp

        prediction = self.model(normalized_x_tilde)
        prediction = tf.argmax(prediction, axis=-1)
        label = tf.argmax(label, axis=-1)
        correct = tf.cast(tf.equal(prediction, label), tf.float32)
        num_samples = tf.cast(tf.shape(prediction)[0], tf.float32)

        accuracy = tf.reduce_mean(tf.reduce_sum(correct) / num_samples)

        return {'total': total,
                'bpp': bpp,
                'mse': mse,
                'downstream': downstream,
                'reconstruction': reconstruction,
                'metric_psnr': psnr,
                'accuracy': accuracy}

    def fit(self, train_dataset, epochs,
            train_steps, val_dataset, val_steps,
            main_lr_schedule, aux_lr_schedule, val_period, checkpoint_period, experiments_dir, trial_id):
        step = tf.train.create_global_step()

        main_lr = tf.Variable(main_lr_schedule(0))
        aux_lr = tf.Variable(aux_lr_schedule(0))

        main_optimizer = tf.train.AdamOptimizer(learning_rate=main_lr)
        main_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(main_optimizer)

        aux_optimizer = tf.train.AdamOptimizer(learning_rate=aux_lr)
        aux_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(aux_optimizer)

        train_losses = self._losses(train_dataset, training=True)
        val_losses = self._losses(val_dataset, training=False)

        filtered_variables = [v for v in tf.global_variables() if v not in self.model.variables]
        aux_step = aux_optimizer.minimize(self.entropy_bottleneck.losses[0], var_list=filtered_variables)
        main_step = main_optimizer.minimize(train_losses['total'], global_step=step, var_list=filtered_variables)
        optimizer_variables = main_optimizer.variables() + main_optimizer._optimizer.variables()
        optimizer_variables += aux_optimizer.variables() + aux_optimizer._optimizer.variables()

        scalar_summary_op = tf.summary.merge(
            [tf.summary.scalar(k, v) for k, v in train_losses.items()] + [tf.summary.scalar('main_lr', main_lr),
                                                                          tf.summary.scalar('aux_lr', aux_lr)])

        train_op = tf.group(main_step, aux_step, self.entropy_bottleneck.updates[0])

        checkpoint_path = Path(experiments_dir) / trial_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        self.sess = tf.keras.backend.get_session()
        with tf.compat.v2.summary.create_file_writer(str(checkpoint_path)).as_default() as w:
            self.sess.run(w.init())
            self.sess.run(hp.hparams(self.hparams, trial_id=trial_id))
            self.sess.run(w.flush())

        checkpoint = tf.train.Checkpoint(analysis_transform=self.analysis_transform,
                                         synthesis_transform=self.synthesis_transform,
                                         entropy_bottleneck=self.entropy_bottleneck)

        self.sess.run(tf.variables_initializer(filtered_variables + optimizer_variables))
        self.sess.run(tf.local_variables_initializer())
        tf.keras.backend.set_learning_phase(0)

        train_writer = tf.summary.FileWriter(str(checkpoint_path / 'train'), session=self.sess)
        val_writer = tf.summary.FileWriter(str(checkpoint_path / 'val'), session=self.sess)

        def scalar_summary(name, value):
            return tf.summary.Summary(value=[tf.summary.Summary.Value(tag=name, simple_value=value)])

        main_lr_placeholder = tf.placeholder(tf.float32)
        aux_lr_placeholder = tf.placeholder(tf.float32)
        assign_lrs = tf.group([tf.assign(main_lr, main_lr_placeholder),
                               tf.assign(aux_lr, aux_lr_placeholder)])

        for epoch in trange(epochs, desc='epoch'):
            self.sess.run(assign_lrs,
                          feed_dict={main_lr_placeholder: main_lr_schedule(epoch),
                                     aux_lr_placeholder: aux_lr_schedule(epoch)})

            for _ in trange(train_steps, desc='batch'):
                _, summaries, current_step = self.sess.run([train_op, scalar_summary_op, step])
                train_writer.add_summary(summaries, global_step=current_step)

            if epoch % checkpoint_period == 0:
                checkpoint.save(str(checkpoint_path / 'checkpoint'),
                                session=self.sess)

            if epoch % val_period == 0 or epoch == epochs - 1:
                accumulated_val_losses = {}

                for val_step in range(val_steps):
                    losses = self.sess.run(val_losses)
                    for k, v in losses.items():
                        if k not in accumulated_val_losses:
                            accumulated_val_losses[k] = v
                        else:
                            accumulated_val_losses[k] += v

                for k in accumulated_val_losses.keys():
                    accumulated_val_losses[k] /= val_steps
                    val_writer.add_summary(scalar_summary(k, accumulated_val_losses[k]), global_step=current_step)

        self.fit_or_loaded = True
        return accumulated_val_losses


def load_checkpoint(self, checkpoint_dir):
    latest = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
    self.sess = tf.Session()
    tf.train.Saver().restore(self.sess, save_path=latest)
    self.fit_or_loaded = True


def compress(self, dataset_folder, output_folder):
    if not self.fit_or_loaded:
        raise RuntimeError('The model must be fit or loaded from checkpoint before compressing.')

    if not self.hparams['scale_hyperprior']:
        raise NotImplementedError('Compression with a scale hyperprior model is not implemented.')

    files = list(Path(dataset_folder).glob('**/*.png'))

    # Load input image and add batch dimension.
    filename = tf.placeholder(tf.string)
    output_filename = tf.placeholder(tf.string)

    x = read_png(filename)
    x = tf.expand_dims(x, 0)
    x.set_shape([1, None, None, 3])
    x_shape = tf.shape(x)

    # Transform and compress the image.
    y = self.analysis_transform(x)
    string = self.entropy_bottleneck.compress(y)

    # Transform the quantized image back (if requested).
    y_hat, y_likelihoods = self.entropy_bottleneck(y, training=False)
    x_hat = self.synthesis_transform(y_hat)
    x_hat = x_hat[:, :x_shape[1], :x_shape[2], :]
    write_file = write_png(output_filename, x_hat[0])

    # Load the latest model checkpoint, get the compressed string and the tensor
    # shapes.
    tensors = [string, tf.shape(x)[1:-1], tf.shape(y)[1:-1], write_file]

    for file in tqdm(files):
        Path(str(file).replace(dataset_folder, output_folder)).parent.mkdir(parents=True, exist_ok=True)

        arrays = self.sess.run(tensors, {filename: str(file),
                                         output_filename: str(file).replace(dataset_folder, output_folder)})

        # Write a binary file with the shape information and the compressed string.
        packed = tfc.PackedTensors()
        packed.pack(tensors[:-1], arrays[:-1])
        with open(str(file).replace(dataset_folder, output_folder).replace('.png', '.tfci'), "wb") as f:
            f.write(packed.string)
