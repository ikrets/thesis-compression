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
import sys

from absl import app
from absl.flags import argparse_flags
import numpy as np
import math
from coolname import generate_slug
import tensorflow.compat.v1 as tf
from pathlib import Path
from tqdm import tqdm
from experiment import save_experiment_params
from models.layers import FiLM
from models.compressors import AnalysisTransform, SynthesisTransform

tfk = tf.keras
tfkl = tf.keras.layers

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


def train(args):
    """Trains the model."""
    if args.perceptual_loss_alpha_range or args.lambda_range:
        use_FiLM = True
    else:
        use_FiLM = False

    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    # Create input data pipeline.
    with tf.device("/cpu:0"):
        dataset = Path(args.dataset)
        train_filenames = list((dataset / 'train').glob('**/*.png'))
        train_dataset = datasets.cifar10.pipeline(filenames=train_filenames,
                                                  flip=True,
                                                  crop=False,
                                                  batch_size=args.batchsize,
                                                  num_parallel_calls=8)
        train_steps = math.ceil(len(train_filenames) / args.batchsize)
        val_filenames = list((dataset / 'test').glob('**/*.png'))
        val_dataset = datasets.cifar10.pipeline(filenames=val_filenames,
                                                flip=False,
                                                crop=False,
                                                batch_size=args.batchsize,
                                                num_parallel_calls=8)
        val_steps = math.ceil(len(val_filenames) / args.batchsize)

    num_pixels = args.batchsize * 32 * 32

    # Instantiate model.
    if use_FiLM:
        analysis_transform = AnalysisTransform(args.num_filters, use_FiLM=True,
                                               FiLM_depth=2,
                                               FiLM_width=128,
                                               FiLM_activation='relu')
        synthesis_transform = SynthesisTransform(args.num_filters, use_FiLM=True,
                                                 FiLM_depth=2,
                                                 FiLM_width=128,
                                                 FiLM_activation='relu')

        def get_parameters(batch_size):
            return {
                'lambda': tf.repeat(args.lmbda, batch_size)[..., tf.newaxis] if args.lmbda else tf.math.exp(
                    tf.random.uniform([batch_size, 1],
                                      minval=tf.math.log(
                                          args.lambda_range[0]),
                                      maxval=tf.math.log(
                                          args.lambda_range[1]))),
                'perceptual_loss_alpha': tf.repeat(args.perceptual_loss_alpha, batch_size)[
                    ..., tf.newaxis] if args.perceptual_loss_alpha is not None else tf.math.exp(
                    tf.random.uniform([batch_size, 1],
                                      minval=tf.math.log(args.perceptual_loss_alpha_range[0]),
                                      maxval=tf.math.log(args.perceptual_loss_alpha_range[1]))
                )
            }
    else:
        analysis_transform = AnalysisTransform(args.num_filters, use_FiLM=False)
        synthesis_transform = SynthesisTransform(args.num_filters, use_FiLM=False)

        def get_parameters(_):
            return {'lambda': args.lmbda, 'perceptual_loss_alpha': args.perceptual_loss_alpha}

    entropy_bottleneck = tfc.EntropyBottleneck()

    # Minimize loss and auxiliary loss, and execute update op.
    step = tf.train.create_global_step()
    main_lr = tf.Variable(args.main_lr[0])
    aux_lr = tf.Variable(args.aux_lr[0])
    main_optimizer = tf.train.AdamOptimizer(learning_rate=main_lr)
    # main_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(main_optimizer)

    aux_optimizer = tf.train.AdamOptimizer(learning_rate=aux_lr)
    # aux_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(aux_optimizer)

    model = tf.keras.models.load_model(args.model)
    if args.model_weights:
        model.load_weights(args.model_weights)
    perceptual_layer = tf.keras.Model(inputs=model.input, outputs=model.get_layer(args.perceptual_loss_layer).output)

    def losses(dataset, training=True):
        # Get training patch from dataset.
        normalized_x, label = dataset.make_one_shot_iterator().get_next()

        parameters = get_parameters(args.batchsize)

        x = datasets.cifar10.normalize(normalized_x, inverse=True)
        if use_FiLM:
            concat_parameters = tf.concat([parameters['lambda'], parameters['perceptual_loss_alpha']],
                                          axis=-1)
            y = analysis_transform([concat_parameters, x])
        else:
            y = analysis_transform(x)

        y_tilde, y_likelihoods = entropy_bottleneck(y, training=training)
        if use_FiLM:
            x_tilde = synthesis_transform([concat_parameters, y_tilde])
        else:
            x_tilde = synthesis_transform(y_tilde)

        normalized_x_tilde = datasets.cifar10.normalize(x_tilde)

        bpp = tf.reduce_sum(tf.log(y_likelihoods)) / (-np.log(2) * num_pixels)

        # Mean squared error across pixels.
        mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))

        perceptual = tf.reduce_mean(
            tf.squared_difference(perceptual_layer(normalized_x), perceptual_layer(normalized_x_tilde)))
        # Multiply by 255^2 to correct for rescaling.
        reconstruction = (1 - parameters['perceptual_loss_alpha']) * mse
        reconstruction += parameters['perceptual_loss_alpha'] * perceptual
        reconstruction *= 255 ** 2

        psnr = tf.reduce_mean(tf.image.psnr(x, x_tilde, max_val=1.))

        # The rate-distortion cost.
        total = parameters['lambda'] * reconstruction + bpp

        prediction = model(normalized_x_tilde)
        prediction = tf.argmax(prediction, axis=-1)
        label = tf.argmax(label, axis=-1)
        correct = tf.cast(tf.equal(prediction, label), tf.float32)
        num_samples = tf.cast(tf.shape(prediction)[0], tf.float32)

        accuracy = tf.reduce_mean(tf.reduce_sum(correct) / num_samples)

        return {'total': total,
                'bpp': bpp,
                'mse': mse,
                'perceptual': perceptual,
                'reconstruction': reconstruction,
                'metric_psnr': psnr,
                'accuracy': accuracy
                }

    train_losses = losses(train_dataset)
    val_losses = losses(val_dataset, training=False)

    filtered_variables = [v for v in tf.global_variables() if v not in model.variables]
    aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0], var_list=filtered_variables)
    main_step = main_optimizer.minimize(train_losses['total'], global_step=step, var_list=filtered_variables)
    # optimizer_variables = main_optimizer.variables() + main_optimizer._optimizer.variables()
    # optimizer_variables += aux_optimizer.variables() + aux_optimizer._optimizer.variables()

    optimizer_variables = main_optimizer.variables() + aux_optimizer.variables()

    scalar_summary_op = tf.summary.merge(
        [tf.summary.scalar(k, tf.reduce_mean(v)) for k, v in train_losses.items()] + [
            tf.summary.scalar('main_lr', main_lr),
            tf.summary.scalar('aux_lr', aux_lr)])

    # tf.summary.image("original", quantize_image(x))
    # tf.summary.image("reconstruction", quantize_image(x_tilde))

    train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

    checkpoint_path = Path(args.checkpoint_dir) / generate_slug()
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    save_experiment_params(checkpoint_path, args)

    checkpoint = tf.train.Checkpoint(analysis_transform=analysis_transform,
                                     synthesis_transform=synthesis_transform,
                                     entropy_bottleneck=entropy_bottleneck)

    sess = tf.keras.backend.get_session()
    sess.run(tf.variables_initializer(filtered_variables + optimizer_variables))
    sess.run(tf.local_variables_initializer())
    tf.keras.backend.set_learning_phase(0)

    train_writer = tf.summary.FileWriter(str(checkpoint_path / 'train'), session=sess)
    val_writer = tf.summary.FileWriter(str(checkpoint_path / 'val'), session=sess)

    def scalar_summary(name, value):
        return tf.summary.Summary(value=[tf.summary.Summary.Value(tag=name, simple_value=value)])

    for epoch in range(args.epochs):
        if epoch in args.lr_boundaries:
            index = args.lr_boundaries.index(epoch)
            sess.run([tf.assign(main_lr, args.main_lr[index + 1]),
                      tf.assign(aux_lr, args.aux_lr[index + 1])])

        for epoch_step in range(train_steps):
            _, summaries, current_step = sess.run([train_op, scalar_summary_op, step])
            train_writer.add_summary(summaries, global_step=current_step)

        if epoch % args.checkpoint_period == 0:
            checkpoint.save(str(checkpoint_path / 'checkpoint'),
                            session=sess)

        if epoch % args.validation_period == 0:
            acc = {}

            for val_step in range(val_steps):
                losses = sess.run(val_losses)
                for k, v in losses.items():
                    if k not in acc:
                        acc[k] = np.mean(v)
                    else:
                        acc[k] += np.mean(v)

            for k, v in acc.items():
                val_writer.add_summary(scalar_summary(k, v / val_steps), global_step=current_step)


def compress(args):
    if args.lambdas is not None and args.perceptual_loss_alphas is not None:
        use_FiLM = True
    else:
        use_FiLM = False

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
    # TODO read this from the model (will be done when going over to TF2)
    analysis_transform = AnalysisTransform(args.num_filters, use_FiLM=use_FiLM,
                                           FiLM_activation='relu',
                                           FiLM_width=128,
                                           FiLM_depth=2)
    synthesis_transform = SynthesisTransform(args.num_filters, use_FiLM=use_FiLM,
                                             FiLM_activation='relu',
                                             FiLM_width=128,
                                             FiLM_depth=2)
    entropy_bottleneck = tfc.EntropyBottleneck()

    with tf.Session() as sess:
        for lmbda in args.lambdas:
            for perceptual_loss_alpha in args.perceptual_loss_alphas:
                # Transform and compress the image.
                if use_FiLM:
                    parameters = tf.convert_to_tensor([lmbda, perceptual_loss_alpha])[tf.newaxis, ...]
                    y = analysis_transform([parameters, x])
                else:
                    y = analysis_transform(x)

                string = entropy_bottleneck.compress(y)
                # Transform the quantized image back (if requested).
                y_hat, y_likelihoods = entropy_bottleneck(y, training=False)

                if use_FiLM:
                    x_hat = synthesis_transform([parameters, y_hat])
                else:
                    x_hat = synthesis_transform(y_hat)

                # Load the latest model checkpoint, get the compressed string and the tensor
                # shapes.
                latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
                tf.train.Saver().restore(sess, save_path=latest)

                x_hat = x_hat[:, :x_shape[1], :x_shape[2], :]
                write_file = write_png(output_filename, x_hat[0])

                tensors = [string, tf.shape(x)[1:-1], tf.shape(y)[1:-1], write_file]

                for file in tqdm(files,
                                 desc=f'Processing lambda {lmbda:.03f}, perceptual loss alpha {perceptual_loss_alpha:.03f}'):
                    target_folder = Path(args.output_dataset) / f'lambda_{lmbda:.03f}_pla_{perceptual_loss_alpha:.03f}'
                    target_file = str(file).replace(args.dataset, f'{str(target_folder)}/')
                    target_file = Path(target_file)
                    target_file.parent.mkdir(parents=True, exist_ok=True)

                    arrays = sess.run(tensors, {filename: str(file),
                                                output_filename: str(target_file)})

                    target_file = Path(str(target_file).replace('.png', '.tfci'))
                    # Write a binary file with the shape information and the compressed string.
                    packed = tfc.PackedTensors()
                    packed.pack(tensors[:-1], arrays[:-1])
                    with target_file.open('wb') as f:
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
        "--lambda", type=float, dest="lmbda",
        help="Lambda for rate-distortion tradeoff.")
    train_cmd.add_argument('--lambda_range', type=float, nargs=2)

    train_cmd.add_argument('--lr_boundaries', type=int, nargs='+', required=True)
    train_cmd.add_argument('--main_lr', type=float, nargs='+', required=True)
    train_cmd.add_argument('--aux_lr', type=float, nargs='+', required=True)
    train_cmd.add_argument('--epochs', type=int, required=True)
    train_cmd.add_argument('--validation_period', type=int, required=True)
    train_cmd.add_argument('--checkpoint_period', type=int, default=1000)
    train_cmd.add_argument('--dataset', type=str, required=True)
    train_cmd.add_argument('--model', type=str)
    train_cmd.add_argument('--model_weights', type=str)

    train_cmd.add_argument('--perceptual_loss_layer', type=str)
    train_cmd.add_argument('--perceptual_loss_alpha', type=float)
    train_cmd.add_argument('--perceptual_loss_alpha_range', type=float, nargs=2)
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
    compress_cmd.add_argument('--lambdas', type=float, nargs='+')
    compress_cmd.add_argument('--perceptual_loss_alphas', type=float, nargs='+')
    compress_cmd.add_argument('--output_dataset', type=str, required=True)

    # Parse arguments.
    args = parser.parse_args(argv[1:])
    if args.command is None:
        parser.print_usage()
        sys.exit(2)

    if args.command == 'train':
        if (args.lmbda and args.lambda_range) or (not args.lmbda and not args.lambda_range):
            parser.error('Set only one of lmbda or lambda_range')
            sys.exit(2)

        if (args.perceptual_loss_alpha is None and args.perceptual_loss_alpha_range is None) or (
                args.perceptual_loss_alpha is not None and args.perceptual_loss_alpha_range is not None):
            parser.error('Set only one of perceptual_loss_alpha or perceptual_loss_alpha_range')
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
