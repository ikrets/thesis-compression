import tensorflow as tf
import numpy as np
import argparse
import math
from pathlib import Path
from weight_decay_optimizers import AdamW
import tensorflow.keras.backend as K
import coolname

from datasets.imagenette import pipeline, read_images
from experiment import save_experiment_params
from models.utils import LRandWDScheduler
from models.resnet18 import resnet18_proper

tfk = tf.keras
AUTO = tf.data.experimental.AUTOTUNE

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--cache', action='store_true')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--bn_momentum', type=float, default=0.9)
parser.add_argument('--base_lr', type=float, default=0.1)
parser.add_argument('--base_wd', type=float, default=5e-4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--drop_lr_multiplier', type=float, default=0.1)
parser.add_argument('--drop_lr_epochs', type=int, nargs='+', default=(60, 90))
parser.add_argument('--experiment_dir', type=str, required=True)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--no_slug', action='store_true')
parser.add_argument('--validation_freq', type=int, default=10)
args = parser.parse_args()

optimizer = AdamW(lr=args.base_lr, weight_decay=args.base_wd)
if args.fp16:
    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

sess = tf.keras.backend.get_session()

data_train, train_examples = read_images(Path(args.dataset) / 'train')
data_test, test_examples = read_images(Path(args.dataset) / 'val')

data_train = pipeline(data_train, batch_size=args.batch_size, size=args.image_size, is_training=True, cache=args.cache)
data_test = pipeline(data_test, batch_size=args.batch_size, size=args.image_size, is_training=False, cache=args.cache)

input = tfk.layers.Input(shape=[args.image_size, args.image_size, 3])
model = resnet18_proper(input)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])


def schedule(epoch):
    times_dropped = sum(1 for drop_epoch in args.drop_lr_epochs if epoch > drop_epoch)
    return args.drop_lr_multiplier ** times_dropped


lr_and_wd_scheduler = LRandWDScheduler(multiplier_schedule=schedule,
                                       base_lr=args.base_lr,
                                       base_wd=args.base_wd,
                                       fp16=args.fp16)

tensorboard_callback = tfk.callbacks.TensorBoard(profile_batch=0,
                                                 log_dir=args.experiment_dir)
experiment_path = Path(args.experiment_dir)
if not args.no_slug:
    experiment_path = experiment_path / coolname.generate_slug()

experiment_path.mkdir(parents=True, exist_ok=True)
save_experiment_params(experiment_path, args)

model.save(experiment_path / 'model.hdf5',
           include_optimizer=False)

flatten = lambda item: (item['X'], item['label'])
data_train = data_train.map(flatten, AUTO)
data_test = data_test.map(flatten, AUTO)

model.fit(data_train.prefetch(AUTO),
          epochs=args.epochs,
          steps_per_epoch=math.ceil(train_examples / args.batch_size),
          validation_data=data_test.prefetch(AUTO),
          validation_steps=math.ceil(test_examples / args.batch_size),
          validation_freq=args.validation_freq,
          callbacks=[lr_and_wd_scheduler, tensorboard_callback])
model.save(experiment_path / 'final_model.hdf5', include_optimizer=False)
