import argparse
import math
from pathlib import Path
from models.compressors import SimpleFiLMCompressor, CompressorWithDownstreamComparison, pipeline_add_parameters
import tensorflow.compat.v1 as tf
import numpy as np
import coolname

from experiment import save_experiment_params
import datasets.cifar10

parser = argparse.ArgumentParser()
parser.add_argument('--num_filters', type=int, default=192)
parser.add_argument('--depth', type=int, required=True)
parser.add_argument('--film_depth', type=int)
parser.add_argument('--film_width', type=int)
parser.add_argument('--film_activation', type=str)
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--lambda_range', type=float, nargs=2, required=True)
parser.add_argument('--alpha_range', type=float, nargs=2, required=True)

parser.add_argument('--main_lr', type=float, required=True)
parser.add_argument('--aux_lr', type=float, required=True)

parser.add_argument('--epochs', type=int, required=True)

parser.add_argument('--summary_period', type=int, default=1)
parser.add_argument('--checkpoint_period', type=int, default=50)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--downstream_model', type=str)
parser.add_argument('--downstream_model_weights', type=str)
parser.add_argument('--downstream_loss_layer', type=str)

parser.add_argument('experiment_dir', type=str)

args = parser.parse_args()

experiment_dir = Path(args.experiment_dir) / coolname.generate_slug()
experiment_dir.mkdir(parents=True)
save_experiment_params(experiment_dir, args)

with tf.device("/cpu:0"):
    dataset = Path(args.dataset)
    train_filenames = list((dataset / 'train').glob('**/*.png'))
    train_dataset = datasets.cifar10.pipeline(filenames=train_filenames,
                                              flip=True,
                                              crop=False,
                                              batch_size=args.batchsize,
                                              num_parallel_calls=8)
    train_dataset = pipeline_add_parameters(train_dataset, args.alpha_range, args.lambda_range)
    train_steps = math.ceil(len(train_filenames) / args.batchsize)
    val_filenames = list((dataset / 'test').glob('**/*.png'))
    val_dataset = datasets.cifar10.pipeline(filenames=val_filenames,
                                            flip=False,
                                            crop=False,
                                            batch_size=args.batchsize,
                                            num_parallel_calls=8)
    val_dataset = pipeline_add_parameters(val_dataset, args.alpha_range, args.lambda_range)
    val_steps = math.ceil(len(val_filenames) / args.batchsize)

    reverse_normalize = lambda X: datasets.cifar10.normalize(X, inverse=True)

compressor = SimpleFiLMCompressor(num_filters=args.num_filters,
                                  depth=args.depth,
                                  FiLM_depth=args.film_depth,
                                  FiLM_width=args.film_width,
                                  FiLM_activation=args.film_activation)
downstream_model = tf.keras.models.load_model(args.downstream_model)
if args.downstream_model_weights:
    downstream_model.load_weights(args.downstream_model_weights)
downstream_loss = lambda original, compressed: tf.reduce_mean(tf.keras.losses.mean_squared_error(original, compressed))

compressor_with_downstream_comparison = CompressorWithDownstreamComparison(
    compressor,
    downstream_model=downstream_model,
    downstream_metric=tf.keras.metrics.categorical_accuracy,
    downstream_compressed_vs_uncompressed_layer='activation_5',
    downstream_compressed_vs_uncompressed_loss=downstream_loss,
    reverse_normalize=reverse_normalize)
main_optimizer = tf.train.AdamOptimizer(args.main_lr)
aux_optimizer = tf.train.AdamOptimizer(args.aux_lr)

compressor_with_downstream_comparison.set_optimizers(main_optimizer=main_optimizer,
                                                     aux_optimizer=aux_optimizer)
compressor_with_downstream_comparison.fit(tf.keras.backend.get_session(),
                                          train_dataset, train_steps,
                                          val_dataset, val_steps,
                                          epochs=args.epochs,
                                          log_dir=experiment_dir,
                                          log_period=args.summary_period,
                                          checkpoint_period=args.checkpoint_period)
