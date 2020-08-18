import tensorflow as tf
import numpy as np
import argparse
import math
from pathlib import Path
from weight_decay_optimizers import SGDW
import tensorflow.keras.backend as K
import coolname

from models.resnet18 import resnet18
from models.vgg16 import vgg16

from datasets.cifar10 import pipeline, read_images, read_compressed_tfrecords
from experiment import save_experiment_params

tfk = tf.keras

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--dataset_type', choices=['files', 'compressed_tfrecords'], required=True)
parser.add_argument('--model', choices=['resnet18', 'vgg16'], required=True)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--base_lr', type=float, default=0.1)
parser.add_argument('--base_wd', type=float, default=5e-4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--drop_lr_multiplier', type=float, default=0.1)
parser.add_argument('--drop_lr_epochs', type=int, nargs='+', default=(60, 90))
parser.add_argument('--experiment_dir', type=str, required=True)
parser.add_argument('--no_slug', action='store_true')
args = parser.parse_args()

optimizer = SGDW(lr=args.base_lr, weight_decay=args.base_wd, momentum=0.9, name='sgdw')


class LRandWDScheduler(tfk.callbacks.Callback):
    def __init__(self, multiplier_schedule, base_lr, base_wd):
        super(LRandWDScheduler, self).__init__()
        self.multiplier_schedule = multiplier_schedule
        self.base_lr = base_lr
        self.base_wd = base_wd

    def on_epoch_begin(self, epoch, logs=None):
        multiplier = self.multiplier_schedule(epoch)
        K.set_value(self.model.optimizer.lr, self.base_lr * multiplier)
        K.set_value(self.model.optimizer.weight_decay, self.base_wd * multiplier)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        logs['lr'] = K.get_value(self.model.optimizer.lr)
        logs['weight_decay'] = K.get_value(self.model.optimizer.weight_decay)


sess = tf.keras.backend.get_session()

if args.dataset_type == 'files':
    data_train, train_examples = read_images(Path(args.dataset) / 'train')
    data_test, test_examples = read_images(Path(args.dataset) / 'test')
if args.dataset_type == 'compressed_tfrecords':
    data_train, data_test = read_compressed_tfrecords([args.dataset])
    data_train = data_train.cache()
    data_test = data_test.cache()
    train_examples = sess.run(data_train.reduce(np.int64(0), lambda x, _: x + 1))
    test_examples = sess.run(data_test.reduce(np.int64(0), lambda x, _: x + 1))

data_train = pipeline(data_train, flip=True, crop=True, batch_size=args.batch_size, shuffle_buffer_size=10000)
data_test = pipeline(data_test, flip=False, crop=False, batch_size=args.batch_size)

input = tfk.layers.Input(shape=[32, 32, 3])
if args.model == 'resnet18':
    model = resnet18(input)
if args.model == 'vgg16':
    model = vgg16(input)

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
model.fit(data_train,
          epochs=args.epochs,
          steps_per_epoch=math.ceil(train_examples / args.batch_size),
          validation_data=data_test,
          validation_steps=math.ceil(test_examples / args.batch_size),
          callbacks=[lr_and_wd_scheduler, tensorboard_callback])
model.save(experiment_path / 'final_model.hdf5', include_optimizer=False)
