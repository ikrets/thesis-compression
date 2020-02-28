import tensorflow as tf
import coolname
import argparse
import math
from pathlib import Path
from models.resnet18 import resnet18
from weight_decay_optimizers import SGDW
import tensorflow.keras.backend as K
from datasets.cifar10 import pipeline
from experiment import save_experiment_params

tfk = tf.keras

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--base_lr', type=float, default=0.1)
parser.add_argument('--base_wd', type=float, default=5e-4)
parser.add_argument('--experiment_dir', type=str, required=True)
args = parser.parse_args()


class LRandWDScheduler(tfk.callbacks.Callback):
    def __init__(self, multiplier_schedule, base_lr, base_wd):
        super(LRandWDScheduler, self).__init__()
        self.multiplier_schedule = multiplier_schedule
        self.base_lr = base_lr
        self.base_wd = base_wd

    def on_epoch_begin(self, epoch, _):
        multiplier = self.multiplier_schedule(epoch)
        K.set_value(self.model.optimizer._optimizer.lr, self.base_lr * multiplier)
        K.set_value(self.model.optimizer._optimizer.weight_decay, self.base_wd * multiplier)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        logs['lr'] = K.get_value(self.model.optimizer._optimizer.lr)
        logs['weight_decay'] = K.get_value(self.model.optimizer._optimizer.weight_decay)


data_train = list(Path(args.dataset).glob('**/train/*/*/*.png'))
train_len = len(data_train)
data_train = pipeline(data_train, flip=True, crop=True, batch_size=args.batch_size, num_parallel_calls=8)

data_test = list(Path(args.dataset).glob('**/test/*/*/*.png'))
test_len = len(data_test)
data_test = pipeline(data_test, flip=False, crop=False, batch_size=args.batch_size, num_parallel_calls=8)

input = tfk.layers.Input(shape=[32, 32, 3])
model = resnet18(input)
optimizer = SGDW(lr=0.1, weight_decay=5e-4, momentum=0.9, name='sgdw')
optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
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
log_dir = f'{args.experiment_dir}/{coolname.generate_slug()}'

tensorboard_callback = tfk.callbacks.TensorBoard(profile_batch=0,
                                                 log_dir=log_dir)
Path(log_dir).mkdir(parents=True, exist_ok=True)
save_experiment_params(log_dir, args)

model.save(log_dir + '/model.hdf5',
           include_optimizer=False)
model.fit(data_train,
          epochs=100,
          steps_per_epoch=math.ceil(train_len / args.batch_size),
          validation_data=data_test,
          validation_steps=math.ceil(test_len / args.batch_size),
          callbacks=[lr_and_wd_scheduler, tensorboard_callback])
model.save(log_dir + '/final_model.hdf5', include_optimizer=False)
