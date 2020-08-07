import argparse
import math
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary

import datasets
from models.bpp_range import BppRangeAdapter, BppRangeEvaluation
from models.compressors import SimpleFiLMCompressor
from models import compressors
from training_schemes import CompressorWithDownstreamLoss
import models.downstream_losses
import tensorflow.compat.v1 as tf
import coolname

from models.utils import make_stepwise

from experiment import save_experiment_params
import datasets.cifar10


def prepare_dataset(args) -> datasets.DatasetSetup:
    with tf.device("/cpu:0"):
        dataset = Path(args.dataset)
        data_train, train_examples = datasets.cifar10.read_images(dataset / 'train')
        data_test, val_examples = datasets.cifar10.read_images(dataset / 'test')

        train_dataset = datasets.cifar10.pipeline(data_train,
                                                  flip=True,
                                                  crop=True,
                                                  batch_size=args.batchsize,
                                                  shuffle_buffer_size=10000,
                                                  classifier_normalize=False)
        train_dataset = train_dataset.map(lambda X, label: {'X': X, 'label': label},
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.prefetch(1)
        train_steps = math.ceil(train_examples / args.batchsize)
        val_dataset = datasets.cifar10.pipeline(data_test,
                                                flip=False,
                                                crop=False,
                                                batch_size=args.eval_batchsize,
                                                classifier_normalize=False)
        val_dataset = val_dataset.map(lambda X, label: {'X': X, 'label': label},
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.prefetch(1)
        val_steps = math.ceil(val_examples / args.eval_batchsize)

    return datasets.DatasetSetup(
        train_dataset=train_dataset,
        train_examples=train_examples,
        train_steps=train_steps,
        val_dataset=val_dataset,
        val_examples=val_examples,
        val_steps=val_steps
    )


def run_yoto_bpp_adapter(args: argparse.Namespace) -> None:
    dataset_setup = prepare_dataset(args)

    experiment_dir = Path(args.experiment_dir)
    if not args.no_slug:
        experiment_dir /= coolname.generate_slug()

    experiment_dir.mkdir(parents=True)
    save_experiment_params(experiment_dir, args)

    opt = tf.train.AdamOptimizer if args.optimizer == 'adam' else tf.train.MomentumOptimizer
    main_lr = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    main_optimizer = opt(main_lr)
    aux_optimizer = opt(args.aux_lr)
    if args.fp16:
        main_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(main_optimizer)
        aux_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(aux_optimizer)

    compressor = SimpleFiLMCompressor(num_filters=args.num_filters,
                                      depth=args.depth,
                                      num_postproc=args.num_postproc,
                                      FiLM_depth=args.film_depth,
                                      FiLM_width=args.film_width,
                                      FiLM_activation=args.film_activation)

    downstream_model = tf.keras.models.load_model(args.downstream_model)
    if args.downstream_model_weights:
        downstream_model.load_weights(args.downstream_model_weights)
    preprocess_fn = lambda X: datasets.cifar10.normalize(X if not args.correct_bgr else X[..., ::-1])

    downstream_loss = models.downstream_losses.PerceptualLoss(
        model=downstream_model,
        preprocess_fn=preprocess_fn,
        metric_fn=tf.keras.metrics.categorical_accuracy,
        readout_layers=args.perceptual_loss_readouts,
        normalize_activations=args.perceptual_loss_normalize_activations,
    )

    bpp_range_adapter = BppRangeAdapter(compressor=compressor,
                                        eval_dataset=dataset_setup.val_dataset,
                                        eval_dataset_steps=dataset_setup.val_steps,
                                        target_bpp_range=args.target_bpp_range,
                                        lmbda=args.lmbda,
                                        linspace_steps=10)
    bpp_range_evaluator = BppRangeEvaluation(compressor=compressor,
                                             downstream_loss=downstream_loss,
                                             bpp_range_adapter=bpp_range_adapter,
                                             val_dataset=dataset_setup.val_dataset,
                                             val_dataset_steps=dataset_setup.val_steps,
                                             log_dir=experiment_dir)
    bpp_range_adapter.alpha_to_bpp.fit(args.initial_alpha_range, args.target_bpp_range)

    compressor_with_downstream_comparison = CompressorWithDownstreamLoss(compressor,
                                                                         downstream_loss)

    if not args.drop_lr_epochs:
        main_schedule = lambda _: args.main_lr
    else:
        main_schedule = make_stepwise(args.main_lr, args.drop_lr_epochs, args.drop_lr_multiplier)

    compressor_with_downstream_comparison.set_optimizers(main_optimizer=main_optimizer,
                                                         aux_optimizer=aux_optimizer,
                                                         main_lr=main_lr,
                                                         main_schedule=main_schedule)

    hparams = {
        'main_lr': args.main_lr,
        'filters': args.num_filters,
        'depth': args.depth,
        'film_depth': args.film_depth,
        'film_width': args.film_width,
        'batch_size': args.batchsize,
        'perceptual_loss_readouts': args.perceptual_loss_readouts,
        'perceptual_loss_normalize_activations': args.perceptual_loss_normalize_activations,
        'lambda': args.lmbda,
        'target_bpp_range': args.target_bpp_range
    }
    writer = tf.summary.FileWriter(experiment_dir)
    writer.add_summary(summary.session_start_pb(hparams=hparams))
    writer.flush()

    compressor_with_downstream_comparison.fit(dataset_setup,
                                              add_parameters_fn=bpp_range_adapter.add_computed_parameters,
                                              epochs=args.epochs,
                                              log_dir=experiment_dir, val_log_period=args.val_summary_period,
                                              checkpoint_period=args.checkpoint_period,
                                              callbacks=[bpp_range_evaluator.callback])
    writer.add_summary(summary.session_end_pb(api_pb2.STATUS_SUCCESS))


def run_fixed_parameters(args: argparse.Namespace) -> None:
    dataset_setup = prepare_dataset(args)

    experiment_dir = Path(args.experiment_dir)
    if not args.no_slug:
        experiment_dir /= coolname.generate_slug()

    experiment_dir.mkdir(parents=True)
    save_experiment_params(experiment_dir, args)

    opt = tf.train.AdamOptimizer if args.optimizer == 'adam' else tf.train.MomentumOptimizer
    main_lr = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    main_optimizer = opt(main_lr)
    aux_optimizer = opt(args.aux_lr)
    if args.fp16:
        main_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(main_optimizer)
        aux_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(aux_optimizer)

    compressor = SimpleFiLMCompressor(num_filters=args.num_filters,
                                      depth=args.depth,
                                      num_postproc=0,
                                      FiLM_depth=0,
                                      FiLM_width=0,
                                      FiLM_activation=None)

    downstream_model = tf.keras.models.load_model(args.downstream_model)
    if args.downstream_model_weights:
        downstream_model.load_weights(args.downstream_model_weights)
    preprocess_fn = lambda X: datasets.cifar10.normalize(X if not args.correct_bgr else X[..., ::-1])

    downstream_loss = models.downstream_losses.PerceptualLoss(
        model=downstream_model,
        preprocess_fn=preprocess_fn,
        metric_fn=tf.keras.metrics.categorical_accuracy,
        readout_layers=args.perceptual_loss_readouts,
        normalize_activations=args.perceptual_loss_normalize_activations,
        backbone_layer=args.perceptual_loss_backbone_prefix
    )

    compressor_with_downstream_comparison = CompressorWithDownstreamLoss(compressor,
                                                                         downstream_loss)

    if not args.drop_lr_epochs:
        main_schedule = lambda _: args.main_lr
    else:
        main_schedule = make_stepwise(args.main_lr, args.drop_lr_epochs, args.drop_lr_multiplier)

    compressor_with_downstream_comparison.set_optimizers(main_optimizer=main_optimizer,
                                                         aux_optimizer=aux_optimizer,
                                                         main_lr=main_lr,
                                                         main_schedule=main_schedule)

    hparams = {
        'main_lr': args.main_lr,
        'filters': args.num_filters,
        'depth': args.depth,
        'film_depth': 0,
        'film_width': 0,
        'batch_size': args.batchsize,
        'perceptual_loss_readouts': args.perceptual_loss_readouts,
        'perceptual_loss_normalize_activations': args.perceptual_loss_normalize_activations,
        'alpha': args.alpha,
        'lambda': args.lmbda
    }
    writer = tf.summary.FileWriter(experiment_dir)
    writer.add_summary(summary.session_start_pb(hparams=hparams))
    writer.flush()

    add_parameters_fn = lambda item: compressors.pipeline_add_constant_parameters(item, args.alpha, args.lmbda)
    compressor_with_downstream_comparison.fit(dataset_setup,
                                              add_parameters_fn=add_parameters_fn,
                                              epochs=args.epochs,
                                              log_dir=experiment_dir,
                                              val_log_period=args.val_summary_period,
                                              checkpoint_period=args.checkpoint_period)
    writer.add_summary(summary.session_end_pb(api_pb2.STATUS_SUCCESS))


parser = argparse.ArgumentParser()
parser.add_argument('--num_filters', type=int, default=192)
parser.add_argument('--depth', type=int, required=True)
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--eval_batchsize', type=int, default=256)

parser.add_argument('--optimizer', choices=['momentum', 'adam'], required=True)
parser.add_argument('--main_lr', type=float, required=True)
parser.add_argument('--aux_lr', type=float, required=True)
parser.add_argument('--drop_lr_epochs', type=int, nargs='+')
parser.add_argument('--drop_lr_multiplier', type=float, default=0.5)

parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--correct_bgr', action='store_true')

parser.add_argument('--val_summary_period', type=int, default=5)
parser.add_argument('--checkpoint_period', type=int, default=50)
parser.add_argument('--dataset', type=str, required=True)

parser.add_argument('--downstream_model', type=str, required=True)
parser.add_argument('--downstream_model_weights', type=str)
parser.add_argument('--experiment_dir', type=str, required=True)

parser.add_argument('--perceptual_loss_readouts', type=str, nargs='+')
parser.add_argument('--perceptual_loss_normalize_activations', action='store_true')
parser.add_argument('--perceptual_loss_backbone_prefix', type=str)

parser.add_argument('--no_slug', action='store_true')
parser.add_argument('--fp16', action='store_true')

subparsers = parser.add_subparsers(help='The training scheme')

yoto_bpp_adapter = subparsers.add_parser('yoto_bpp_adapter')
yoto_bpp_adapter.add_argument('--num_postproc', type=int, required=True)
yoto_bpp_adapter.add_argument('--film_depth', type=int)
yoto_bpp_adapter.add_argument('--film_width', type=int)
yoto_bpp_adapter.add_argument('--film_activation', type=str)
yoto_bpp_adapter.add_argument('--lambda', type=float, required=True, dest='lmbda')
yoto_bpp_adapter.add_argument('--initial_alpha_range', type=float, nargs=2, required=True)
yoto_bpp_adapter.add_argument('--val_bpp_linspace_steps', type=int, default=10)
yoto_bpp_adapter.add_argument('--target_bpp_range', nargs=2, type=float, required=True)
yoto_bpp_adapter.add_argument('--reevaluate_bpp_range_period', type=int, required=True)
yoto_bpp_adapter.set_defaults(func=run_yoto_bpp_adapter)

fixed_parameters = subparsers.add_parser('fixed_parameters')
fixed_parameters.add_argument('--lambda', type=float, required=True, dest='lmbda')
fixed_parameters.add_argument('--alpha', type=float, required=True)
fixed_parameters.set_defaults(func=run_fixed_parameters)

args = parser.parse_args()
args.func(args)
