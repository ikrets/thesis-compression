import argparse
import tensorflow as tf
import cv2
import numpy as np
import json
from pathlib import Path
import re
from tqdm import trange

from models.compressors import SimpleFiLMCompressor, pipeline_add_constant_parameters

parser = argparse.ArgumentParser()

parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--lambda_range', type=float, nargs=2, required=True)
parser.add_argument('--alpha', type=float, required=True)
parser.add_argument('--experiment', type=str, required=True)
parser.add_argument('--data_to_compress', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--steps', type=int, required=True)

args = parser.parse_args()

with (Path(args.experiment) / 'parameters.json').open('r') as fp:
    parameters = json.load(fp)

lambda_linspace = np.linspace(args.lambda_range[0], args.lambda_range[1], args.steps)

model = SimpleFiLMCompressor(num_filters=parameters['num_filters'],
                             depth=parameters['depth'],
                             num_postproc=parameters['num_postproc'],
                             FiLM_width=parameters['film_width'],
                             FiLM_depth=parameters['film_depth'],
                             FiLM_activation=parameters['film_activation'])
model.forward(item={'X': np.zeros((128, 32, 32, 3)).astype(np.float32),
                    'alpha': np.zeros(128).astype(np.float32),
                    'lambda': np.zeros(128).astype(np.float32)},
              training=False, run_range_coder=True)

weights = list(Path(args.experiment).glob('compressor_epoch_*_weights.h5'))
epochs = [re.search('compressor_epoch_([0-9]+)_weights.h5', str(f)).groups(1) for f in weights]
print(f'Loading weights from epoch {max(epochs)}')
model.load_weights(str(weights[np.argmax(epochs)]))

files = [str(f) for f in Path(args.data_to_compress).glob('**/*.png')]
input_data = tf.data.Dataset.from_tensor_slices(files)
input_data = input_data.map(
    lambda fname: {'X': tf.io.read_file(fname),
                   'name': tf.strings.regex_replace(fname, f'{args.data_to_compress}/', '')})


def decode_and_preprocess_cifar10(img_string):
    img = tf.io.decode_png(img_string)
    img = tf.cast(img, dtype=tf.float32) / 255.

    return img


input_data = input_data.map(lambda item: {'X': decode_and_preprocess_cifar10(item['X']),
                                          'name': item['name']},
                            num_parallel_calls=8)
input_data = input_data.batch(args.batchsize).cache()

sess = tf.keras.backend.get_session()
for lmbda in lambda_linspace:
    input_data = input_data.map(lambda X: pipeline_add_constant_parameters(X, args.alpha, lmbda),
                                num_parallel_calls=8)
    input_data = input_data.prefetch(1)

    item = input_data.make_one_shot_iterator().get_next()
    compressed_batch = model.forward(item, training=False, run_range_coder=True)
    Y_compressed = compressed_batch['Y_compressed']
    X_reconstructed = tf.cast(
        tf.clip_by_value(compressed_batch['X_tilde'], 0, 1) * 255, tf.uint8)

    for batch in trange(np.ceil(len(files) / args.batchsize).astype(int), desc=f'Processing lambda {lmbda:0.3f}'):
        Y_compressed_result, X_reconstructed_result, name_result = sess.run(
            [Y_compressed, X_reconstructed, item['name']])
        for name, compressed_img, reconstructed_img in zip(name_result, Y_compressed_result, X_reconstructed_result):
            target_filename = Path(args.output_dir) / f'alpha_{args.alpha:0.3f}_lambda_{lmbda:0.3f}' / name.decode(
                'ascii')
            target_filename.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(target_filename), reconstructed_img[..., ::-1])
            with open(str(target_filename).replace('.png', '.tfci'), 'wb') as fp:
                fp.write(compressed_img)
