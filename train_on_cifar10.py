import tensorflow as tf
import numpy as np
import argparse
import math
from pathlib import Path
from weight_decay_optimizers import SGDW
import tensorflow.keras.backend as K

from models.resnet18 import resnet18
from datasets.cifar10 import pipeline, read_images, read_compressed_tfrecords
from experiment import save_experiment_params

tfk = tf.keras

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--dataset_type', choices=['files', 'compressed_tfrecords'], required=True)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--base_lr', type=float, default=0.1)
parser.add_argument('--base_wd', type=float, default=5e-4)
parser.add_argument('--experiment_dir', type=str, required=True)
args = parser.parse_args()

optimizer = SGDW(lr=0.1, weight_decay=5e-4, momentum=0.9, name='sgdw')
optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

class LRandWDScheduler(tfk.callbacks.Callback):
    def __init__(self, multiplier_schedule, base_lr, base_wd):
        super(LRandWDScheduler, self).__init__()
        self.multiplier_schedule = multiplier_schedule
        self.base_lr = base_lr
        self.base_wd = base_wd

    def on_epoch_begin(self, epoch, logs=None):
        multiplier = self.multiplier_schedule(epoch)
        K.set_value(self.model.optimizer._optimizer.lr, self.base_lr * multiplier)
        K.set_value(self.model.optimizer._optimizer.weight_decay, self.base_wd * multiplier)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        logs['lr'] = K.get_value(self.model.optimizer._optimizer.lr)
        logs['weight_decay'] = K.get_value(self.model.optimizer._optimizer.weight_decay)

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
model = resnet18(input)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])


def schedule(epoch):
    if epoch < 60:
        return 1
    elif epoch < 80:
        return 0.1
    else:
        return 0.01


lr_and_wd_scheduler = LRandWDScheduler(multiplier_schedule=schedule,
                                       base_lr=args.base_lr,
                                       base_wd=args.base_wd)

tensorboard_callback = tfk.callbacks.TensorBoard(profile_batch=0,
                                                 log_dir=args.experiment_dir)
Path(args.experiment_dir).mkdir(parents=True, exist_ok=True)
save_experiment_params(args.experiment_dir, args)

model.save(Path(args.experiment_dir) / 'model.hdf5',
           include_optimizer=False)
model.fit(data_train,
          epochs=100,
          steps_per_epoch=math.ceil(train_examples / args.batch_size),
          validation_data=data_test,
          validation_steps=math.ceil(test_examples / args.batch_size),
          callbacks=[lr_and_wd_scheduler, tensorboard_callback])
model.save(Path(args.experiment_dir) / 'final_model.hdf5', include_optimizer=False)
