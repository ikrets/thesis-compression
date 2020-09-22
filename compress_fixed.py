import argparse
import tensorflow as tf
import cv2
import numpy as np
import json
from pathlib import Path
import re
from tqdm import trange, tqdm
from threading import Thread
from queue import Queue
from typing import Dict, Tuple, Union

import datasets
from models.compressors import SimpleFiLMCompressor
from experiment import save_experiment_params

AUTO = tf.data.experimental.AUTOTUNE

parser = argparse.ArgumentParser()

parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--experiment', type=str)

parser.add_argument('--data_to_compress', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)

args = parser.parse_args()

Path(args.output_dir).mkdir(parents=True, exist_ok=True)
save_experiment_params(args.output_dir, args)

files = [str(f) for f in Path(args.data_to_compress).glob('**/*.png')]


def read_and_preprocess_cifar10(fname):
    img_string = tf.io.read_file(fname)
    img = tf.io.decode_png(img_string)
    img = tf.reshape(img, shape=[32, 32, 3])
    img = tf.cast(img, dtype=tf.float32) / 255.

    return img


def writer_fn(queue: Queue) -> None:
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    writer = tf.io.TFRecordWriter(str(output_path / f'compressed.tfrecord'))

    while True:
        item = queue.get()
        if item is None:
            break

        name_result, range_coded_bpp_result, X_reconstructed_result, experiment_name, alpha, lmbda = item
        for name, range_coded_bpp, reconstructed_img in zip(name_result, range_coded_bpp_result,
                                                            X_reconstructed_result):
            name_parts = Path(name.decode('ascii')).parts[-4:]

            _, reconstructed_img_string = cv2.imencode('.png', reconstructed_img[..., ::-1])
            example = datasets.serialize_example(name='/'.join(name_parts),
                                                 range_coded_bpp=range_coded_bpp,
                                                 X=reconstructed_img_string.tobytes(),
                                                 alpha=alpha,
                                                 lmbda=lmbda)
            writer.write(example)

    writer.close()


experiment = Path(args.experiment)
results_queue: 'Queue[Union[Tuple[np.array, np.array, np.array, str, float, float], None]]' = Queue(maxsize=20)
writer_thread = Thread(target=writer_fn, args=(results_queue,))
writer_thread.start()

input_data = tf.data.Dataset.from_tensor_slices(files)
input_data = input_data.map(
    lambda fname: {'X': read_and_preprocess_cifar10(fname),
                   'name': fname},
    AUTO)

input_data = input_data.batch(args.batchsize).repeat().prefetch(AUTO)

with (experiment / 'parameters.json').open('r') as fp:
    parameters = json.load(fp)

model = SimpleFiLMCompressor(num_filters=parameters['num_filters'],
                             depth=parameters['depth'],
                             num_postproc=0,
                             FiLM_width=0,
                             FiLM_depth=0,
                             FiLM_activation=None)
model.forward(item={'X': np.zeros((128, 32, 32, 3)).astype(np.float32),
                    'alpha': np.zeros(128).astype(np.float32),
                    'lambda': np.zeros(128).astype(np.float32)},
              training=False)

weights = list(experiment.glob('compressor_epoch_*_weights.h5'))
matches = [re.search('compressor_epoch_([0-9]+)_weights.h5', str(f)) for f in weights]
epochs = [int(m.group(1)) for m in matches if m]
print(f'Loading weights from epoch {max(epochs)}')
model.load_weights(str(weights[max(range(len(epochs)), key=lambda x: epochs[x])]))

sess = tf.keras.backend.get_session()
item = input_data.make_one_shot_iterator().get_next()
alpha_placeholder = tf.placeholder(tf.float32)

alphas = tf.repeat([parameters['alpha']], tf.shape(item['X'])[0])
lambdas = tf.repeat([parameters['lmbda']], tf.shape(item['X'])[0])
item_with_params = {'X': item['X'],
                    'name': item['name'],
                    'alpha': alphas,
                    'lambda': lambdas}
batch = model.forward_with_range_coding(item_with_params)
parameters_stacked = tf.stack([alphas, lambdas], axis=-1)
X_reconstructed = tf.cast(tf.clip_by_value(batch['X_reconstructed'], 0, 1) * 255, tf.uint8)

for _ in trange(np.ceil(len(files) / args.batchsize).astype(int), desc='Batches'):
    range_coded_bpp_result, X_reconstructed_result, name_result = sess.run(
        [batch['range_coded_bpp'], X_reconstructed, item['name']])
    results_queue.put((name_result, range_coded_bpp_result, X_reconstructed_result, experiment.name,
                       parameters['alpha'], parameters['lmbda']))

results_queue.put(None)
writer_thread.join()
