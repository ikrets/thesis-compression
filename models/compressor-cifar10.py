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
from coolname import generate_slug
import tensorflow.compat.v1 as tf
from pathlib import Path
from tqdm import tqdm

import tensorflow_compression as tfc
import datasets.cifar10

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

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(AnalysisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_0", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_0")),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_1")),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_2")),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_3", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_3"))
        ]
        super(AnalysisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


class SynthesisTransform(tf.keras.layers.Layer):
    """The synthesis transform."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(SynthesisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="igdn_0", inverse=True)),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="igdn_1", inverse=True)),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_2", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="igdn_2", inverse=True)),
            tfc.SignalConv2D(
                3, (5, 5), name="layer_3", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=None),
        ]
        super(SynthesisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


def train(args):
    """Trains the model."""

    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    # Create input data pipeline.
    with tf.device("/cpu:0"):
        dataset = Path(args.dataset)
        train_dataset = datasets.cifar10.pipeline(filenames=list((dataset / 'train').glob('**/*.png')),
                                                  flip=True,
                                                  crop=False,
                                                  batch_size=args.batchsize,
                                                  num_parallel_calls=8)
        val_dataset = datasets.cifar10.pipeline(filenames=list((dataset / 'test').glob('**/*.png')),
                                                flip=False,
                                                crop=False,
                                                batch_size=args.batchsize,
                                                num_parallel_calls=8)

    num_pixels = args.batchsize * 32 * 32

    # Get training patch from dataset.
    normalized_x, label = train_dataset.make_one_shot_iterator().get_next()
    x = datasets.cifar10.normalize(normalized_x, inverse=True)

    # Instantiate model.
    analysis_transform = AnalysisTransform(args.num_filters)
    synthesis_transform = SynthesisTransform(args.num_filters)
    entropy_bottleneck = tfc.EntropyBottleneck()

    # Build autoencoder and hyperprior.
    y = analysis_transform(x)
    y_tilde, y_likelihoods = entropy_bottleneck(y, training=True)
    x_tilde = synthesis_transform(y_tilde)
    normalized_x_tilde = datasets.cifar10.normalize(x_tilde)

    # Total number of bits divided by number of pixels.
    train_bpp = tf.reduce_sum(tf.log(y_likelihoods)) / (-np.log(2) * num_pixels)

    # Mean squared error across pixels.
    train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))

    # Minimize loss and auxiliary loss, and execute update op.
    step = tf.train.create_global_step()
    main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    main_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(main_optimizer)

    aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    aux_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(aux_optimizer)

    model = tf.keras.models.load_model(args.model)
    if args.model_weights:
        model.load_weights(args.model_weights)
    model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(args.perceptual_loss_layer).output)

    train_perceptual_loss = tf.reduce_mean(tf.squared_difference(model(normalized_x), model(normalized_x_tilde)))
    # Multiply by 255^2 to correct for rescaling.
    train_reconstruction_loss = (1 - args.perceptual_loss_alpha) * train_mse
    train_reconstruction_loss += args.perceptual_loss_alpha * train_perceptual_loss
    train_reconstruction_loss *= 255 ** 2

    # The rate-distortion cost.
    train_loss = args.lmbda * train_reconstruction_loss + train_bpp

    aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])
    filtered_variables = [v for v in tf.global_variables() if v not in model.trainable_variables]
    main_step = main_optimizer.minimize(train_loss, global_step=step, var_list=filtered_variables)
    optimizer_variables = main_optimizer.variables() + main_optimizer._optimizer.variables()
    optimizer_variables += aux_optimizer.variables() + aux_optimizer._optimizer.variables()

    tf.summary.scalar("loss", train_loss)
    tf.summary.scalar("bpp", train_bpp)
    tf.summary.scalar("mse", train_mse)
    tf.summary.scalar("perceptual_loss", train_perceptual_loss)
    tf.summary.scalar("reconstruction_loss", train_reconstruction_loss)

    tf.summary.image("original", quantize_image(x))
    tf.summary.image("reconstruction", quantize_image(x_tilde))

    train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

    checkpoint_path = Path(args.checkpoint_dir) / generate_slug()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    with (checkpoint_path / 'parameters.json').open('w') as fp:
        json.dump(vars(args), fp, indent=4)
    checkpoint = tf.train.Checkpoint(analysis_transform=analysis_transform,
                                     synthesis_transform=synthesis_transform,
                                     entropy_bottleneck=entropy_bottleneck)

    sess = tf.keras.backend.get_session()
    sess.run(tf.variables_initializer(filtered_variables + optimizer_variables))
    sess.run(tf.local_variables_initializer())
    writer = tf.summary.FileWriter(str(checkpoint_path), session=sess)
    summary_op = tf.summary.merge_all()

    for step in range(args.last_step):
        if step % args.summary_period == 0:
            results = sess.run([train_op, summary_op])
            writer.add_summary(results[-1], global_step=step)
        else:
            sess.run(train_op)

        if step % args.checkpoint_period == 0:
            checkpoint.save(str(checkpoint_path / 'checkpoint'),
                            session=sess)


def compress(args):
    """Compresses an image."""
    files = list(Path(args.dataset).glob('**/*.png'))

    # Load input image and add batch dimension.
    filename = tf.placeholder(tf.string)
    output_filename = tf.placeholder(tf.string)

    x = read_png(filename)
    x = tf.expand_dims(x, 0)
    x.set_shape([1, None, None, 3])
    x_shape = tf.shape(x)

    # Instantiate model.
    analysis_transform = AnalysisTransform(args.num_filters)
    synthesis_transform = SynthesisTransform(args.num_filters)
    entropy_bottleneck = tfc.EntropyBottleneck()

    # Transform and compress the image.
    y = analysis_transform(x)
    string = entropy_bottleneck.compress(y)

    # Transform the quantized image back (if requested).
    y_hat, y_likelihoods = entropy_bottleneck(y, training=False)
    x_hat = synthesis_transform(y_hat)
    x_hat = x_hat[:, :x_shape[1], :x_shape[2], :]
    write_file = write_png(output_filename, x_hat[0])

    with tf.Session() as sess:
        # Load the latest model checkpoint, get the compressed string and the tensor
        # shapes.
        latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
        tf.train.Saver().restore(sess, save_path=latest)
        tensors = [string, tf.shape(x)[1:-1], tf.shape(y)[1:-1], write_file]

        for file in tqdm(files):
            Path(str(file).replace(args.dataset, args.output_dataset)).parent.mkdir(parents=True, exist_ok=True)

            arrays = sess.run(tensors, {filename: str(file),
                                        output_filename: str(file).replace(args.dataset, args.output_dataset)})

            # Write a binary file with the shape information and the compressed string.
            packed = tfc.PackedTensors()
            packed.pack(tensors[:-1], arrays[:-1])
            with open(str(file).replace(args.dataset, args.output_dataset).replace('.png', '.tfci'), "wb") as f:
                f.write(packed.string)


def parse_args(argv):
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # High-level options.
    parser.add_argument(
        "--verbose", "-V", action="store_true",
        help="Report bitrate and distortion when training or compressing.")
    parser.add_argument(
        "--num_filters", type=int, default=192,
        help="Number of filters per layer.")
    parser.add_argument(
        "--checkpoint_dir", default="train",
        help="Directory where to save/load model checkpoints.")
    subparsers = parser.add_subparsers(
        title="commands", dest="command",
        help="What to do: 'train' loads training data and trains (or continues "
             "to train) a new model. 'compress' reads an image file (lossless "
             "PNG format) and writes a compressed binary file. 'decompress' "
             "reads a binary file and reconstructs the image (in PNG format). "
             "input and output filenames need to be provided for the latter "
             "two options. Invoke '<command> -h' for more information.")

    # 'train' subcommand.
    train_cmd = subparsers.add_parser(
        "train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Trains (or continues to train) a new model.")
    train_cmd.add_argument(
        "--batchsize", type=int, default=8,
        help="Batch size for training.")
    train_cmd.add_argument(
        "--lambda", type=float, default=0.01, dest="lmbda",
        help="Lambda for rate-distortion tradeoff.")
    train_cmd.add_argument('--summary_period', type=int, required=True)
    train_cmd.add_argument('--checkpoint_period', type=int, default=1000)
    train_cmd.add_argument('--dataset', type=str, required=True)
    train_cmd.add_argument('--model', type=str)
    train_cmd.add_argument('--model_weights', type=str)
    train_cmd.add_argument('--perceptual_loss_layer', type=str)
    train_cmd.add_argument('--perceptual_loss_alpha', type=float)
    train_cmd.add_argument(
        "--last_step", type=int, default=1000000,
        help="Train up to this number of steps.")
    train_cmd.add_argument(
        "--preprocess_threads", type=int, default=16,
        help="Number of CPU threads to use for parallel decoding of training "
             "images.")

    # 'compress' subcommand.
    compress_cmd = subparsers.add_parser(
        "compress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a PNG file, compresses it, and writes a TFCI file.")
    compress_cmd.add_argument('--dataset', type=str, required=True)
    compress_cmd.add_argument('--output_dataset', type=str, required=True)

    # Parse arguments.
    args = parser.parse_args(argv[1:])
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    return args


def main(args):
    # Invoke subcommand.
    if args.command == "train":
        train(args)
    elif args.command == "compress":
        compress(args)


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
