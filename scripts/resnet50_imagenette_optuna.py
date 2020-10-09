import tensorflow as tf
import numpy as np
import argparse
import math
from pathlib import Path
from weight_decay_optimizers import SGDW, AdamW
import tensorflow.keras.backend as K
import coolname
import optuna

from datasets.imagenette import pipeline, read_images
from experiment import save_experiment_params
from models.utils import LRandWDScheduler

tfk = tf.keras
AUTO = tf.data.experimental.AUTOTUNE

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--image_size', type=int, default=160)
parser.add_argument('--min_image_size', type=int, nargs=2, default=(300, 300))
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--bn_momentum', type=float, default=0.9)
parser.add_argument('--base_lr_range', type=float, nargs=2)
parser.add_argument('--base_wd_range', type=float, nargs=2)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--validation_freq', type=int, default=10)
parser.add_argument('--drop_lr_multiplier_choices', type=float, nargs='+', required=True)
parser.add_argument('--drop_lr_epochs', type=int, nargs='+', default=(60, 90))
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--study_name', type=str, required=True)
parser.add_argument('--study_storage_dir', type=str, required=True)
parser.add_argument('--n_trials', type=int, required=True)
args = parser.parse_args()


def objective(trial):
    K.clear_session()

    base_lr = trial.suggest_loguniform('base_lr', *args.base_lr_range)
    base_wd = trial.suggest_loguniform('wd', *args.base_wd_range)
    drop_lr_multiplier = trial.suggest_categorical('drop_lr_multiplier', args.drop_lr_multiplier_choices)

    optimizer = AdamW(lr=base_lr,
                     weight_decay=base_wd)

    if args.fp16:
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    data_train, train_examples = read_images(Path(args.dataset) / 'train')
    data_test, test_examples = read_images(Path(args.dataset) / 'val')
    data_train = pipeline(data_train, batch_size=args.batch_size, size=args.image_size, is_training=True,
                          min_height=args.min_image_size[0],
                          min_width=args.min_image_size[1])
    data_test = pipeline(data_test, batch_size=args.batch_size, size=args.image_size, is_training=False,
                         min_height=args.min_image_size[0],
                         min_width=args.min_image_size[1])
    preprocess_fn = lambda item: (item['X'], item['label'])
    data_train = data_train.map(preprocess_fn, AUTO)
    data_test = data_test.map(preprocess_fn, AUTO)

    model = tf.keras.applications.ResNet50V2(weights=None, pooling='avg', classes=10,
                                             input_shape=[args.image_size, args.image_size, 3])
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = args.bn_momentum
    model = tf.keras.models.model_from_json(model.to_json())
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    def schedule(epoch):
        times_dropped = sum(1 for drop_epoch in args.drop_lr_epochs if epoch > drop_epoch)
        return drop_lr_multiplier ** times_dropped

    lr_and_wd_scheduler = LRandWDScheduler(multiplier_schedule=schedule,
                                           base_lr=base_lr,
                                           base_wd=base_wd,
                                           fp16=args.fp16)

    experiment_path = Path(args.study_storage_dir) / coolname.generate_slug()
    experiment_path.mkdir(parents=True, exist_ok=True)
    tensorboard_callback = tfk.callbacks.TensorBoard(profile_batch=0,
                                                     log_dir=experiment_path)
    pruning_callback = optuna.integration.TFKerasPruningCallback(trial, 'val_categorical_accuracy')

    model.save(experiment_path / 'model.hdf5',
               include_optimizer=False)
    history = model.fit(data_train.prefetch(AUTO),
                        epochs=args.epochs,
                        steps_per_epoch=math.ceil(train_examples / args.batch_size),
                        validation_data=data_test.prefetch(AUTO),
                        validation_steps=math.ceil(test_examples / args.batch_size),
                        validation_freq=args.validation_freq,
                        callbacks=[lr_and_wd_scheduler, tensorboard_callback, pruning_callback])
    model.save(experiment_path / 'final_model.hdf5', include_optimizer=False)
    return history.history['val_categorical_accuracy'][-1]


Path(args.study_storage_dir).mkdir(parents=True, exist_ok=True)
save_experiment_params(args.study_storage_dir, args)
study = optuna.create_study(study_name=args.study_name,
                            direction='maximize',
                            storage='sqlite:///{}/study.db'.format(args.study_storage_dir),
                            pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
                            load_if_exists=True)
study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
