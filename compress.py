import argparse
import tensorflow as tf
import cv2
import numpy as np
import json
from pathlib import Path
import re
from tqdm import trange, tqdm

from models.compressors import SimpleFiLMCompressor
from experiment import save_experiment_params

parser = argparse.ArgumentParser()

parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--experiment', type=str)
parser.add_argument('--experiment_collection_dir', type=str)

parser.add_argument('--alpha_linspace_steps', type=int, required=True)
parser.add_argument('--lambda_linspace_steps', type=int, required=True)
parser.add_argument('--alpha_range', type=float, nargs=2)
parser.add_argument('--lambda_range', type=float, nargs=2)
parser.add_argument('--sample_function', choices=['uniform', 'loguniform'], required=True)

parser.add_argument('--data_to_compress', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)

args = parser.parse_args()

if args.sample_function == 'uniform':
    alpha_linspace = np.linspace(args.alpha_range[0], args.alpha_range[1], args.alpha_linspace_steps)
    lambda_linspace = np.linspace(args.lambda_range[0], args.lambda_range[1], args.lambda_linspace_steps)
if args.sample_function == 'loguniform':
    alpha_linspace = np.exp(
        np.linspace(np.log(args.alpha_range[0]), np.log(args.alpha_range[1]),
                    args.alpha_linspace_steps))
    lambda_linspace = np.exp(
        np.linspace(np.log(args.lambda_range[0]), np.log(args.lambda_range[1]),
                    args.lambda_linspace_steps))

Path(args.output_dir).mkdir(parents=True, exist_ok=True)
save_experiment_params(args.output_dir, args)

files = [str(f) for f in Path(args.data_to_compress).glob('**/*.png')]


def decode_and_preprocess_cifar10(img_string):
    img = tf.io.decode_png(img_string)
    img = tf.reshape(img, shape=[32, 32, 3])
    img = tf.cast(img, dtype=tf.float32) / 255.

    return img


if args.experiment_collection_dir:
    experiments = [e for e in Path(args.experiment_collection_dir).iterdir() if
                   e.is_dir() and (e / 'parameters.json').exists()]
elif args.experiment_dir:
    experiments = [Path(args.experiment_dir)]

for experiment in tqdm(experiments, desc='Experiments'):
    tf.reset_default_graph()
    input_data = tf.data.Dataset.from_tensor_slices(files)
    input_data = input_data.map(
        lambda fname: {'X': tf.io.read_file(fname),
                       'name': tf.strings.regex_replace(fname, f'{args.data_to_compress}/', '')})

    input_data = input_data.map(lambda item: {'X': decode_and_preprocess_cifar10(item['X']),
                                              'name': item['name']},
                                num_parallel_calls=8)
    input_data = input_data.batch(args.batchsize).prefetch(tf.data.experimental.AUTOTUNE)

    with (experiment / 'parameters.json').open('r') as fp:
        parameters = json.load(fp)

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

    weights = list(experiment.glob('compressor_epoch_*_weights.h5'))
    matches = [re.search('compressor_epoch_([0-9]+)_weights.h5', str(f)) for f in weights]
    epochs = [int(m.group(1)) for m in matches if m]
    if epochs:
        print(f'Loading weights from epoch {max(epochs)}')
        model.load_weights(str(weights[max(range(len(epochs)), key=lambda x: epochs[x])]))
    else:
        print(f'No weights for experiment {experiment} found!')
        continue

    sess = tf.keras.backend.get_session()
    for alpha in alpha_linspace:
        for lmbda in lambda_linspace:
            item = input_data.make_one_shot_iterator().get_next()

            alphas = tf.repeat([alpha], tf.shape(item['X'])[0])
            lambdas = tf.repeat([lmbda], tf.shape(item['X'])[0])
            item_with_params = {'X': item['X'],
                                'name': item['name'],
                                'alpha': alphas,
                                'lambda': lambdas}
            compressed_batch = model.forward(item_with_params, training=False)
            Y_range_coded = model.entropy_bottleneck.compress(compressed_batch['Y'])
            Y_decoded = model.entropy_bottleneck.decompress(Y_range_coded, shape=compressed_batch['Y'].shape[1:3],
                                                            channels=compressed_batch['Y'].shape[3])
            parameters_stacked = tf.stack([alphas, lambdas], axis=-1)
            X_reconstructed = model.synthesis_transform([parameters_stacked, Y_decoded])
            X_reconstructed = tf.cast(tf.clip_by_value(X_reconstructed, 0, 1) * 255, tf.uint8)

            for batch in trange(np.ceil(len(files) / args.batchsize).astype(int),
                                desc=f'Processing alpha {alpha:0.4f}, lambda {lmbda:0.4f}'):
                Y_range_coded_result, X_reconstructed_result, name_result = sess.run(
                    [Y_range_coded, X_reconstructed, item['name']])
                for name, compressed_img, reconstructed_img in zip(name_result, Y_range_coded_result,
                                                                   X_reconstructed_result):
                    target_filename = Path(
                        args.output_dir) / experiment.name / f'alpha_{alpha:0.4f}_lambda_{lmbda:0.4f}' / name.decode(
                        'ascii')
                    target_filename.parent.mkdir(parents=True, exist_ok=True)

                    cv2.imwrite(str(target_filename), reconstructed_img[..., ::-1])
                    with open(str(target_filename).replace('.png', '.tfci'), 'wb') as fp:
                        fp.write(compressed_img)
