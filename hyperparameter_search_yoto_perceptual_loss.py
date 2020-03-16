import optuna
import argparse
import numpy as np
import math
from pathlib import Path
import tensorflow.compat.v1 as tf
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary
import coolname

from models.bpp_range import BppRangeAdapter
from models.compressors import SimpleFiLMCompressor
from training_schemes import CompressorWithDownstreamLoss
import models.downstream_losses
from experiment import save_experiment_params
import datasets.cifar10

parser = argparse.ArgumentParser()
# hparams
parser.add_argument('--num_filters_range', type=int, nargs=2, required=True)
parser.add_argument('--depth_range', type=int, nargs=2, required=True)
parser.add_argument('--film_depth_range', type=int, nargs=2, required=True)
parser.add_argument('--film_width_range', type=int, nargs=2, required=True)
parser.add_argument('--lambda_range', type=float, nargs=2, required=True)
parser.add_argument('--batch_size_range', type=int, nargs='+', required=True)
parser.add_argument('--main_lr_range', type=float, nargs=2, required=True)

# non-hparam settings
parser.add_argument('--perceptual_loss_readouts', type=str, nargs='+')

parser.add_argument('--eval_batchsize', type=int, default=256)
parser.add_argument('--aux_lr', type=float, required=True)
parser.add_argument('--initial_alpha_range', type=float, nargs=2, required=True)
parser.add_argument('--val_bpp_linspace_steps', type=int, default=10)

parser.add_argument('--optimizer', choices=['momentum', 'adam'], required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--correct_bgr', action='store_true')

parser.add_argument('--target_bpp_range', nargs=2, type=float, required=True)
parser.add_argument('--reevaluate_bpp_range_period', type=int, required=True)

parser.add_argument('--val_summary_period', type=int, default=5)
parser.add_argument('--checkpoint_period', type=int, default=50)
parser.add_argument('--dataset', type=str, required=True)

parser.add_argument('--downstream_model', type=str, required=True)
parser.add_argument('--downstream_model_weights', type=str)
parser.add_argument('--experiment_dir', type=str, required=True)
parser.add_argument('--study_name', type=str, required=True)
parser.add_argument('--num_trials', type=int)

args = parser.parse_args()

(Path(args.experiment_dir) / args.study_name).mkdir(parents=True, exist_ok=True)
study = optuna.create_study(study_name=args.study_name,
                            storage=f'sqlite:///{args.experiment_dir}/{args.study_name}/study.db',
                            load_if_exists=True)


def objective(trial: optuna.Trial):
    tf.keras.backend.clear_session()

    trial_dir = Path(args.experiment_dir) / args.study_name / coolname.generate_slug()
    trial_dir.mkdir(parents=True)

    main_lr = trial.suggest_loguniform('main_lr', *args.main_lr_range)
    main_lr_var = tf.Variable(main_lr, dtype=tf.float32, trainable=False)
    main_schedule = lambda _: main_lr

    opt = tf.train.AdamOptimizer if args.optimizer == 'adam' else tf.train.MomentumOptimizer
    main_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt(main_lr_var))
    aux_optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt(args.aux_lr))

    compressor = SimpleFiLMCompressor(
        num_filters=int(trial.suggest_discrete_uniform('filters', *args.num_filters_range, q=16)),
        depth=int(trial.suggest_discrete_uniform('depth', *args.depth_range, q=1)),
        num_postproc=0,
        FiLM_depth=int(trial.suggest_discrete_uniform('film_depth', *args.film_depth_range, q=1)),
        FiLM_width=int(trial.suggest_discrete_uniform('film_width', *args.film_width_range, q=16)),
        FiLM_activation='relu')

    with tf.device("/cpu:0"):
        dataset = Path(args.dataset)
        train_filenames = list((dataset / 'train').glob('**/*.png'))
        batch_size = int(trial.suggest_discrete_uniform('batch_size', *args.batch_size_range, q=16))
        train_dataset = datasets.cifar10.pipeline(filenames=train_filenames,
                                                  flip=True,
                                                  crop=True,
                                                  batch_size=batch_size,
                                                  shuffle_buffer_size=10000,
                                                  classifier_normalize=False,
                                                  num_parallel_calls=8)
        train_dataset = train_dataset.map(lambda X, label: {'X': X, 'label': label},
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.prefetch(1)
        train_steps = math.ceil(len(train_filenames) / batch_size)
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

    single_readout = trial.suggest_categorical('perceptual_loss_single_readout', [True, False])
    if single_readout:
        perceptual_loss_readouts = [trial.suggest_categorical('perceptual_loss_single_readout_name',
                                                              args.perceptual_loss_readouts)]
    else:
        perceptual_loss_readouts = args.perceptual_loss_readouts
    downstream_loss = models.downstream_losses.PerceptualLoss(
        model=downstream_model,
        preprocess_fn=preprocess_fn,
        metric_fn=tf.keras.metrics.categorical_accuracy,
        readout_layers=perceptual_loss_readouts,
        normalize_activations=True
    )

    bpp_range_adapter = BppRangeAdapter(compressor=compressor,
                                        eval_dataset=val_dataset,
                                        eval_dataset_steps=val_steps,
                                        target_bpp_range=args.target_bpp_range,
                                        lmbda=trial.suggest_loguniform('lambda', *args.lambda_range),
                                        linspace_steps=args.val_bpp_linspace_steps)

    compressor_with_downstream_comparison = CompressorWithDownstreamLoss(compressor,
                                                                         downstream_loss,
                                                                         bpp_range_adapter=bpp_range_adapter,
                                                                         initial_alpha_range=args.initial_alpha_range
                                                                         )

    compressor_with_downstream_comparison.set_optimizers(main_optimizer=main_optimizer,
                                                         aux_optimizer=aux_optimizer,
                                                         main_lr=main_lr_var,
                                                         main_schedule=main_schedule)

    params_namespace = argparse.Namespace()
    vars(params_namespace).update(vars(args))
    vars(params_namespace).update(trial.params)
    save_experiment_params(trial_dir, params_namespace)

    writer = tf.summary.FileWriter(trial_dir)
    writer.add_summary(summary.session_start_pb(hparams=trial.params))
    writer.flush()

    def pruning_callback(epoch, metrics):
        if np.mean(metrics['bpp']) > 5 * np.mean(args.target_bpp_range) and epoch > 10:
            return True

        if np.mean(metrics['bpp']) > 1.5 * np.mean(args.target_bpp_range) and epoch > 50:
            return True

        return False

    try:
        value = compressor_with_downstream_comparison.fit(train_dataset, train_steps, val_dataset, val_steps,
                                                          epochs=args.epochs, log_dir=trial_dir,
                                                          val_log_period=args.val_summary_period,
                                                          checkpoint_period=args.checkpoint_period,
                                                          pruning_callback=pruning_callback)
        writer.add_summary(summary.session_end_pb(api_pb2.STATUS_SUCCESS))
        return value


    except:
        writer.add_summary(summary.session_end_pb(api_pb2.STATUS_FAILURE))
        return 0.


study.optimize(objective)
