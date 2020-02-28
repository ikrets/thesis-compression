import argparse
import tensorflow as tf
import cv2
import numpy as np
import json
from pathlib import Path
import re
from tqdm import trange

from models.compressors import SimpleFiLMCompressor, pipeline_add_constant_parameters
from experiment import save_experiment_params

parser = argparse.ArgumentParser()

parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--alpha_range', type=float, nargs=2, required=True)
parser.add_argument('--lambda_range', type=float, nargs=2, required=True)
parser.add_argument('--experiment', type=str, required=True)
parser.add_argument('--data_to_compress', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--steps', type=int, required=True)

args = parser.parse_args()

Path(args.output_dir).mkdir(parents=True, exist_ok=True)
save_experiment_params(args.output_dir, args)

with (Path(args.experiment) / 'parameters.json').open('r') as fp:
    parameters = json.load(fp)

if args.alpha_range[0] != args.alpha_range[1]:
    alpha_linspace = np.linspace(args.alpha_range[0], args.alpha_range[1], args.steps)
else:
    alpha_linspace = [args.alpha_range[0]]

if args.lambda_range[0] != args.lambda_range[1]:
    lambda_linspace = np.linspace(args.lambda_range[0], args.lambda_range[1], args.steps)
else:
    lambda_linspace = [args.lambda_range[0]]

model = SimpleFiLMCompressor(num_filters=parameters['num_filters'],
                             depth=parameters['depth'],
                             num_postproc=parameters['num_postproc'],
                             FiLM_width=parameters['film_width'],
                             FiLM_depth=parameters['film_depth'],
                             FiLM_activation=parameters['film_activation'])
model.forward(item={'X': np.zeros((128, 32, 32, 3)).astype(np.float32),
                    'alpha': np.zeros(128).astype(np.float32),
                    'lambda': np.zeros(128).astype(np.float32)},
              training=False)

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
    img = tf.reshape(img, shape=[32, 32, 3])
    img = tf.cast(img, dtype=tf.float32) / 255.

    return img


input_data = input_data.map(lambda item: {'X': decode_and_preprocess_cifar10(item['X']),
                                          'name': item['name']},
                            num_parallel_calls=8)
input_data = input_data.batch(args.batchsize).cache()

sess = tf.keras.backend.get_session()
for alpha in alpha_linspace:
    for lmbda in lambda_linspace:
        input_params = input_data.map(lambda X: pipeline_add_constant_parameters(X, alpha, lmbda),
                                      num_parallel_calls=8)
        input_params = input_params.prefetch(1)

        item = input_params.make_one_shot_iterator().get_next()
        compressed_batch = model.forward(item, training=False)
        Y_range_coded = model.entropy_bottleneck.compress(compressed_batch['Y'])
        Y_decoded = model.entropy_bottleneck.decompress(Y_range_coded, shape=compressed_batch['Y'].shape[1:3],
                                                        channels=compressed_batch['Y'].shape[3])
        parameters_stacked = tf.stack([item['alpha'], item['lambda']], axis=-1)
        X_reconstructed = model.synthesis_transform([parameters_stacked, Y_decoded])
        X_reconstructed = tf.cast(tf.clip_by_value(X_reconstructed, 0, 1) * 255, tf.uint8)

        for batch in trange(np.ceil(len(files) / args.batchsize).astype(int),
                            desc=f'Processing alpha {alpha:0.4f}, lambda {lmbda:0.4f}'):
            Y_range_coded_result, X_reconstructed_result, name_result = sess.run(
                [Y_range_coded, X_reconstructed, item['name']])
            for name, compressed_img, reconstructed_img in zip(name_result, Y_range_coded_result,
                                                               X_reconstructed_result):
                target_filename = Path(args.output_dir) / f'alpha_{alpha:0.4f}_lambda_{lmbda:0.4f}' / name.decode(
                    'ascii')
                target_filename.parent.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(str(target_filename), reconstructed_img[..., ::-1])
                with open(str(target_filename).replace('.png', '.tfci'), 'wb') as fp:
                    fp.write(compressed_img)
