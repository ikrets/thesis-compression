import argparse
import math
from pathlib import Path

from models.bpp_range_adapter import BppRangeAdapter
from models.compressors import SimpleFiLMCompressor, CompressorWithDownstreamLoss
import models.downstream_losses
import tensorflow.compat.v1 as tf
import coolname

from models.utils import make_stepwise

from experiment import save_experiment_params
import datasets.cifar10

parser = argparse.ArgumentParser()
parser.add_argument('--num_filters', type=int, default=192)
parser.add_argument('--depth', type=int, required=True)
parser.add_argument('--num_postproc', type=int, required=True)
parser.add_argument('--film_depth', type=int)
parser.add_argument('--film_width', type=int)
parser.add_argument('--film_activation', type=str)
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--eval_batchsize', type=int, default=256)
parser.add_argument('--lambda', type=float, required=True, dest='lmbda')
parser.add_argument('--initial_alpha_range', type=float, nargs=2, required=True)
parser.add_argument('--val_bpp_linspace_steps', type=int, default=10)

parser.add_argument('--optimizer', choices=['momentum', 'adam'], required=True)
parser.add_argument('--main_lr', type=float, required=True)
parser.add_argument('--aux_lr', type=float, required=True)
parser.add_argument('--drop_lr_epochs', type=int, nargs='+')
parser.add_argument('--drop_lr_multiplier', type=float, default=0.5)

parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--correct_bgr', action='store_true')

parser.add_argument('--target_bpp_range', nargs=2, type=float, required=True)
parser.add_argument('--reevaluate_bpp_range_period', type=int, required=True)

parser.add_argument('--val_summary_period', type=int, default=5)
parser.add_argument('--checkpoint_period', type=int, default=50)
parser.add_argument('--dataset', type=str, required=True)

parser.add_argument('--downstream_model', type=str, required=True)
parser.add_argument('--downstream_model_weights', type=str)
parser.add_argument('--experiment_dir', type=str)

parser.add_argument('--perceptual_loss_readouts', type=str, nargs='+')
parser.add_argument('--perceptual_loss_normalize_activations', action='store_true')

args = parser.parse_args()

experiment_dir = Path(args.experiment_dir) / coolname.generate_slug()
experiment_dir.mkdir(parents=True)
save_experiment_params(experiment_dir, args)

opt = tf.train.AdamOptimizer if args.optimizer == 'adam' else tf.train.MomentumOptimizer
main_lr = tf.Variable(0.0, dtype=tf.float32, trainable=False)
main_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt(main_lr))
aux_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt(args.aux_lr))

compressor = SimpleFiLMCompressor(num_filters=args.num_filters,
                                  depth=args.depth,
                                  num_postproc=args.num_postproc,
                                  FiLM_depth=args.film_depth,
                                  FiLM_width=args.film_width,
                                  FiLM_activation=args.film_activation)

with tf.device("/cpu:0"):
    dataset = Path(args.dataset)
    train_filenames = list((dataset / 'train').glob('**/*.png'))
    train_dataset = datasets.cifar10.pipeline(filenames=train_filenames,
                                              flip=True,
                                              crop=False,
                                              batch_size=args.batchsize,
                                              shuffle_buffer_size=10000,
                                              classifier_normalize=False,
                                              num_parallel_calls=8)
    train_dataset = train_dataset.map(lambda X, label: {'X': X, 'label': label},
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.prefetch(1)
    train_steps = math.ceil(len(train_filenames) / args.batchsize)
    val_filenames = list((dataset / 'test').glob('**/*.png'))
    val_dataset = datasets.cifar10.pipeline(filenames=val_filenames,
                                            flip=False,
                                            crop=False,
                                            batch_size=args.eval_batchsize,
                                            classifier_normalize=False,
                                            num_parallel_calls=8)
    val_dataset = val_dataset.map(lambda X, label: {'X': X, 'label': label},
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(1)
    val_steps = math.ceil(len(val_filenames) / args.eval_batchsize)

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
                                    eval_dataset=val_dataset,
                                    eval_dataset_steps=val_steps,
                                    bpp_range=args.target_bpp_range,
                                    lmbda=args.lmbda,
                                    initial_alpha_range=args.initial_alpha_range,
                                    alpha_linspace_steps=10)

compressor_with_downstream_comparison = CompressorWithDownstreamLoss(compressor,
                                                                     downstream_loss,
                                                                     bpp_range_adapter=bpp_range_adapter)

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
                                          val_dataset, val_steps,
                                          val_bpp_linspace_steps=args.val_bpp_linspace_steps,
                                          epochs=args.epochs,
                                          log_dir=experiment_dir,
                                          val_log_period=args.val_summary_period,
                                          checkpoint_period=args.checkpoint_period,
                                          reevaluate_bpp_range_period=args.reevaluate_bpp_range_period)
