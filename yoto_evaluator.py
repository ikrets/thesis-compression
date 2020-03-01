import argparse
import tensorflow.compat.v1 as tf
import numpy as np
import json
from pathlib import Path
import re
import pandas as pd
from tqdm import trange, tqdm

from models.compressors import SimpleFiLMCompressor, pipeline_add_range_of_parameters
import datasets.cifar10

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--uncompressed_dataset', type=str, required=True)
parser.add_argument('--experiment_collection_dir', type=str)
parser.add_argument('--experiment_dir', type=str)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--alpha_linspace_steps', type=int, required=True)
parser.add_argument('--lambda_linspace_steps', type=int, required=True)
parser.add_argument('--alpha_range', type=float, nargs=2)
parser.add_argument('--lambda_range', type=float, nargs=2)
parser.add_argument('--sample_function', choices=['uniform', 'loguniform'], required=True)

args = parser.parse_args()

Path(args.output_dir).mkdir(parents=True, exist_ok=True)

files = [str(f) for f in (Path(args.uncompressed_dataset) / 'test').glob('**/*.png')]
labels = datasets.cifar10.filename_to_label(files)
input_data_files = tf.data.Dataset.from_tensor_slices(files)
input_data_labels = tf.data.Dataset.from_tensor_slices(labels)
input_data = tf.data.Dataset.zip((input_data_files, input_data_labels))
input_data = input_data.map(
    lambda fname, label: {'X': tf.io.read_file(fname),
                          'name': fname,
                          'label': label})


def decode_and_preprocess_cifar10(img_string):
    img = tf.io.decode_png(img_string)
    img = tf.cast(img, dtype=tf.float32) / 255.
    img = tf.reshape(img, [32, 32, 3])

    return img


input_data = input_data.map(lambda item: {'X': decode_and_preprocess_cifar10(item['X']),
                                          'name': item['name'],
                                          'label': item['label']},
                            num_parallel_calls=8)
input_data = input_data.batch(args.batch_size)
input_num_batches = np.ceil(len(files) / args.batch_size).astype(int)

if args.experiment_collection_dir:
    experiments = [e for e in Path(args.experiment_collection_dir).iterdir() if
                   e.is_dir() and (e / 'parameters.json').exists()]
elif args.experiment_dir:
    experiments = [Path(args.experiment_dir)]

for experiment in tqdm(experiments, desc='Experiment'):
    with (experiment / 'parameters.json').open('r') as fp:
        parameters = json.load(fp)

    downstream_model = tf.keras.models.load_model(parameters['downstream_model'])
    if parameters['downstream_model_weights']:
        downstream_model.load_weights(parameters['downstream_model_weights'])

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


    def add_params(item):
        return pipeline_add_range_of_parameters(item, alpha_linspace, lambda_linspace)


    input_data_with_params = input_data.map(add_params, num_parallel_calls=8)
    input_data_with_params = input_data_with_params.unbatch().prefetch(tf.data.experimental.AUTOTUNE)

    model = SimpleFiLMCompressor(num_filters=parameters['num_filters'],
                                 depth=parameters['depth'],
                                 num_postproc=parameters['num_postproc'],
                                 FiLM_width=parameters['film_width'],
                                 FiLM_depth=parameters['film_depth'],
                                 FiLM_activation=parameters['film_activation'])
    model.forward(item={'X': np.zeros((args.batch_size, 32, 32, 3)).astype(np.float32),
                        'alpha': np.zeros(args.batch_size).astype(np.float32),
                        'lambda': np.zeros(args.batch_size).astype(np.float32)},
                  training=False)

    weights = list(experiment.glob('compressor_epoch_*_weights.h5'))
    if not weights:
        print(f'No saves found: {experiment}, skipping.')
        continue

    epochs = [int(re.search('compressor_epoch_([0-9]+)_weights.h5', str(f)).group(1)) for f in weights]
    print(f'Loading weights from epoch {max(epochs)}')
    model.load_weights(str(weights[np.argmax(epochs)]))

    sess = tf.keras.backend.get_session()

    item = tf.data.make_one_shot_iterator(input_data_with_params).get_next()
    compressed_batch = model.forward(item, training=False)
    img_area = input_data.element_spec['X'].shape[1] * input_data.element_spec['X'].shape[2]

    Y_compressed = model.entropy_bottleneck.compress(compressed_batch['Y'])
    range_coder_bpp = tf.cast(tf.strings.length(Y_compressed) * 8, tf.float32) / tf.cast(img_area, tf.float32)
    Y_decompressed = model.entropy_bottleneck.decompress(Y_compressed, shape=compressed_batch['Y'].shape[1:3],
                                                         channels=parameters['num_filters'])
    stacked_alpha_lambda = tf.stack([item['alpha'], item['lambda']], axis=-1)
    X_reconstructed = model.synthesis_transform([stacked_alpha_lambda, Y_decompressed])
    X_reconstructed = tf.clip_by_value(X_reconstructed, 0, 1)

    reconstruction_psnr = tf.image.psnr(item['X'], X_reconstructed, max_val=1.)
    # TODO cifar10 specific and BGR correction!
    downstream_prediction = downstream_model(datasets.cifar10.normalize(X_reconstructed[..., ::-1]))
    downstream_loss = tf.keras.losses.categorical_crossentropy(item['label'], downstream_prediction)

    result = {
        'reconstruction_psnr': reconstruction_psnr,
        'range_coder_bpp': range_coder_bpp,
        'name': item['name'],
        'alpha': item['alpha'],
        'lambda': item['lambda'],
        'label': tf.argmax(item['label'], axis=-1),
        'prediction': tf.argmax(downstream_prediction, axis=-1),
        'loss': downstream_loss
    }

    collected_results = {k: [] for k in result.keys()}

    for batch in trange(input_num_batches * len(alpha_linspace) * len(lambda_linspace),
                        desc=f'Bottlenecking batches'):
        result_batch = sess.run(result)

        for k, key_batch in result_batch.items():
            collected_results[k].extend(key_batch)

    pd.DataFrame(collected_results).to_csv(Path(args.output_dir) / f'{experiment.name}_results.csv')
