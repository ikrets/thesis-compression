import tensorflow as tf
import numpy as np
import argparse
import math
from pathlib import Path
from weight_decay_optimizers import SGDW
import tensorflow.keras.backend as K
import coolname

from datasets.imagenette import pipeline, read_images
from experiment import save_experiment_params
from models.utils import LRandWDScheduler

tfk = tf.keras

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--dataset_type', choices=['files', 'compressed_tfrecords'], required=True)
parser.add_argument('--image_size', type=int, default=160)
parser.add_argument('--min_image_size', type=int, nargs=2, default=(300, 300))
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--bn_momentum', type=float, default=0.9)
parser.add_argument('--base_lr', type=float, default=0.1)
parser.add_argument('--base_wd', type=float, default=5e-4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--drop_lr_multiplier', type=float, default=0.1)
parser.add_argument('--drop_lr_epochs', type=int, nargs='+', default=(60, 90))
parser.add_argument('--experiment_dir', type=str, required=True)
parser.add_argument('--no_slug', action='store_true')
args = parser.parse_args()

optimizer = SGDW(lr=args.base_lr, weight_decay=args.base_wd, momentum=0.9, name='sgdw')

sess = tf.keras.backend.get_session()

if args.dataset_type == 'files':
    data_train, train_examples = read_images(Path(args.dataset) / 'train')
    data_test, test_examples = read_images(Path(args.dataset) / 'val')
else:
    assert False

data_train = pipeline(data_train, batch_size=args.batch_size, size=args.image_size, is_training=True,
                      min_height=args.min_image_size[0],
                      min_width=args.min_image_size[1])
data_test = pipeline(data_test, batch_size=args.batch_size, size=args.image_size, is_training=False,
                     min_height=args.min_image_size[0],
                     min_width=args.min_image_size[1])

input = tfk.layers.Input(shape=[args.image_size, args.image_size, 3])
model = tf.keras.applications.ResNet50V2(weights=None, pooling='avg', classes=10,
                                         input_shape=[args.image_size, args.image_size, 3])
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.momentum = args.bn_momentum
model = tf.keras.models.model_from_json(model.to_json())
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])


def schedule(epoch):
    times_dropped = sum(1 for drop_epoch in args.drop_lr_epochs if epoch > drop_epoch)
    return args.drop_lr_multiplier ** times_dropped


lr_and_wd_scheduler = LRandWDScheduler(multiplier_schedule=schedule,
                                       base_lr=args.base_lr,
                                       base_wd=args.base_wd)

tensorboard_callback = tfk.callbacks.TensorBoard(profile_batch=0,
                                                 log_dir=args.experiment_dir)
experiment_path = Path(args.experiment_dir)
if not args.no_slug:
    experiment_path = experiment_path / coolname.generate_slug()

experiment_path.mkdir(parents=True, exist_ok=True)
save_experiment_params(experiment_path, args)

model.save(experiment_path / 'model.hdf5',
           include_optimizer=False)
model.fit(data_train.prefetch(1),
          epochs=args.epochs,
          steps_per_epoch=math.ceil(train_examples / args.batch_size),
          validation_data=data_test.prefetch(1),
          validation_steps=math.ceil(test_examples / args.batch_size),
          callbacks=[lr_and_wd_scheduler, tensorboard_callback])
model.save(experiment_path / 'final_model.hdf5', include_optimizer=False)
