import argparse
import math
from pathlib import Path
from models.compressors import SimpleFiLMCompressor, CompressorWithDownstreamLoss, \
    pipeline_add_sampled_parameters, pipeline_add_constant_parameters
import models.downstream_task
import tensorflow.compat.v1 as tf
import numpy as np
import coolname

from models.utils import make_stepwise

from experiment import save_experiment_params
import datasets.cifar10


def loguniform(len, range):
    return tf.math.exp(tf.random.uniform([len], minval=tf.math.log(range[0]),
                                         maxval=tf.math.log(range[1])))


def uniform(len, range):
    return tf.random.uniform([len], minval=range[0], maxval=range[1])


sample_functions = {
    'loguniform': loguniform,
    'uniform': uniform
}

model_performance_losses = {
    'categorical_crossentropy': tf.keras.losses.categorical_crossentropy
}

parser = argparse.ArgumentParser()
parser.add_argument('--num_filters', type=int, default=192)
parser.add_argument('--depth', type=int, required=True)
parser.add_argument('--num_postproc', type=int, required=True)
parser.add_argument('--film_depth', type=int)
parser.add_argument('--film_width', type=int)
parser.add_argument('--film_activation', type=str)
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--eval_batchsize', type=int, default=256)
parser.add_argument('--lambda_range', type=float, nargs=2, required=True)
parser.add_argument('--alpha_range', type=float, nargs=2, required=True)
parser.add_argument('--sample_function', choices=sample_functions.keys(), required=True)
parser.add_argument('--eval_points', type=int, default=10)

parser.add_argument('--optimizer', choices=['momentum', 'adam'], required=True)
parser.add_argument('--main_lr', type=float, required=True)
parser.add_argument('--aux_lr', type=float, required=True)
parser.add_argument('--drop_lr_epochs', type=int, nargs='+')
parser.add_argument('--drop_lr_multiplier', type=float, default=0.5)

parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--correct_bgr', action='store_true')

parser.add_argument('--train_summary_period', type=int, default=1)
parser.add_argument('--val_summary_period', type=int, default=5)
parser.add_argument('--checkpoint_period', type=int, default=50)
parser.add_argument('--dataset', type=str, required=True)

parser.add_argument('--downstream_model', type=str, required=True)
parser.add_argument('--downstream_model_weights', type=str)
parser.add_argument('--last_frozen_layer', type=str, required=True)

subparsers = parser.add_subparsers(dest='downstream_loss_type')
activation_difference_cmd = subparsers.add_parser('activation_difference')
activation_difference_cmd.add_argument('--downstream_layer', type=str)

task_performance_cmd = subparsers.add_parser('task_performance')
task_performance_cmd.add_argument('--model_performance_loss', choices=model_performance_losses.keys(),
                                  required=True)

parser.add_argument('experiment_dir', type=str)

args = parser.parse_args()

experiment_dir = Path(args.experiment_dir) / coolname.generate_slug()
experiment_dir.mkdir(parents=True)
save_experiment_params(experiment_dir, args)

opt = tf.train.AdamOptimizer if args.optimizer == 'adam' else tf.train.MomentumOptimizer
main_lr = tf.Variable(0.0, tf.float32)
main_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt(main_lr))
aux_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt(args.aux_lr))

with tf.device("/cpu:0"):
    dataset = Path(args.dataset)
    train_filenames = list((dataset / 'train').glob('**/*.png'))
    train_dataset = datasets.cifar10.pipeline(filenames=train_filenames,
                                              flip=True,
                                              crop=False,
                                              batch_size=args.batchsize,
                                              shuffle=True,
                                              classifier_normalize=False,
                                              num_parallel_calls=8)
    train_dataset = pipeline_add_sampled_parameters(train_dataset, args.alpha_range, args.lambda_range,
                                                    sample_fn=sample_functions[args.sample_function])
    train_steps = math.ceil(len(train_filenames) / args.batchsize)
    val_filenames = list((dataset / 'test').glob('**/*.png'))
    val_dataset = datasets.cifar10.pipeline(filenames=val_filenames,
                                            flip=False,
                                            crop=False,
                                            batch_size=args.eval_batchsize,
                                            shuffle=False,
                                            classifier_normalize=False,
                                            num_parallel_calls=8)
    val_steps = math.ceil(len(val_filenames) / args.eval_batchsize)
    random_parameters_val_dataset = pipeline_add_sampled_parameters(val_dataset, args.alpha_range, args.lambda_range,
                                                                    sample_fn=sample_functions[args.sample_function])

    if args.lambda_range[0] != args.lambda_range[1]:
        lambda_eval_linspace = np.linspace(args.lambda_range[0], args.lambda_range[1], args.eval_points)
    else:
        lambda_eval_linspace = [args.lambda_range[0]]

    if args.alpha_range[0] != args.alpha_range[1]:
        alpha_eval_linspace = np.linspace(args.alpha_range[0], args.alpha_range[1], args.eval_points)
    else:
        alpha_eval_linspace = [args.alpha_range[0]]

    const_parameter_val_datasets = {}
    for alpha in alpha_eval_linspace:
        for lmbda in lambda_eval_linspace:
            def add_params(X, label):
                return pipeline_add_constant_parameters({'X': X, 'label': label}, alpha, lmbda)


            const_parameter_val_datasets[(alpha, lmbda)] = val_dataset.map(add_params, num_parallel_calls=8)

compressor = SimpleFiLMCompressor(num_filters=args.num_filters,
                                  depth=args.depth,
                                  num_postproc=args.num_postproc,
                                  FiLM_depth=args.film_depth,
                                  FiLM_width=args.film_width,
                                  FiLM_activation=args.film_activation)
downstream_model = tf.keras.models.load_model(args.downstream_model)
if args.downstream_model_weights:
    downstream_model.load_weights(args.downstream_model_weights)

common_downstream_arguments = {
    'model': downstream_model,
    'metric_fn': tf.keras.metrics.categorical_accuracy,
    'preprocess_fn': lambda X: datasets.cifar10.normalize(X[..., ::-1] if args.correct_bgr else X),
    'last_frozen_layer': args.last_frozen_layer,
}
if args.downstream_loss_type == 'task_performance':
    downstream_task = models.downstream_task.DownstreamTaskPerformance(
        model_performance_loss=tf.keras.losses.categorical_crossentropy, **common_downstream_arguments)
if args.downstream_loss_type == 'activation_difference':
    downstream_task = models.downstream_task.DownstreamActivationDifference(activation_layer=args.downstream_layer,
                                                                            **common_downstream_arguments)

compressor_with_downstream_comparison = CompressorWithDownstreamLoss(compressor,
                                                                     downstream_task)

if not args.drop_lr_epochs:
    main_schedule = lambda _: args.main_lr
else:
    main_schedule = make_stepwise(args.main_lr, args.drop_lr_epochs, args.drop_lr_multiplier)

compressor_with_downstream_comparison.set_optimizers(main_optimizer=main_optimizer,
                                                     aux_optimizer=aux_optimizer,
                                                     main_lr=main_lr,
                                                     main_schedule=main_schedule)

compressor_with_downstream_comparison.fit(tf.keras.backend.get_session(),
                                          train_dataset, train_steps,
                                          random_parameters_val_dataset, val_steps,
                                          const_parameter_val_datasets=const_parameter_val_datasets,
                                          epochs=args.epochs,
                                          log_dir=experiment_dir,
                                          train_log_period=args.train_summary_period,
                                          val_log_period=args.val_summary_period,
                                          checkpoint_period=args.checkpoint_period)
