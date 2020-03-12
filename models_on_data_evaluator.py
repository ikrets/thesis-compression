import tensorflow as tf
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
from typing import List, Dict, Union

import datasets.cifar10

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_collection_dir', type=str)
parser.add_argument('--experiment_dir', type=str)
parser.add_argument('--compressed_extension', type=str, required=True)
parser.add_argument('--original_model', type=str, required=True)
parser.add_argument('--original_model_weights', type=str)
parser.add_argument('--original_dataset', type=str, required=True)
parser.add_argument('--original_model_correct_bgr', action='store_true')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('output_file', type=str)
args = parser.parse_args()


def calculate_file_sizes(test_filenames):
    data = tf.data.Dataset.from_tensor_slices(test_filenames)
    total_bytes = data.map(tf.io.read_file, 8).map(tf.strings.length, 8).reduce(np.int32(0),
                                                                                lambda acc, length: acc + length)
    sess = tf.keras.backend.get_session()
    total_bytes = sess.run(total_bytes)

    return total_bytes * 8 / len(test_filenames) / 32 / 32


original_model = tf.keras.models.load_model(args.original_model)
if args.original_model_weights:
    original_model.load_weights(args.original_model_weights)
original_model.compile('sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

if args.experiment_collection_dir:
    experiments = [e for e in Path(args.experiment_collection_dir).iterdir() if
                   e.is_dir() and (e / 'parameters.json').exists()]
elif args.experiment_dir:
    experiments = [Path(args.experiment_dir)]

test_files = lambda path: [str(f) for f in Path(path).glob('**/test/*/*/*.png')]
compressed_test_files = lambda path: [str(f) for f in Path(path).glob(f'**/test/*/*/*.{args.compressed_extension}')]

results: Dict[str, List[Union[float, str]]] = {'bpp': [], 'experiment_dir': [], 'c2o_crossentropy': [], 'c2o_accuracy': [],
                                   'c2c_crossentropy': [],
                                   'c2c_accuracy': [], 'o2c_crossentropy': [], 'o2c_accuracy': []}

for experiment in tqdm(experiments, desc='Experiment'):
    with (experiment / 'parameters.json').open('r') as fp:
        parameters = json.load(fp)

    original_data_for_compressed_model = datasets.cifar10.pipeline(test_files(args.original_dataset), flip=False,
                                                                   crop=False,
                                                                   batch_size=args.batch_size,
                                                                   correct_bgr=(experiment / 'bgr').exists(),
                                                                   repeat=False,
                                                                   num_parallel_calls=8)

    compressed_data = datasets.cifar10.pipeline(test_files(parameters['dataset']), flip=False, crop=False,
                                                batch_size=args.batch_size,
                                                correct_bgr=(experiment / 'bgr').exists(),
                                                repeat=False,
                                                num_parallel_calls=8)

    compressed_data_for_original_model = datasets.cifar10.pipeline(test_files(parameters['dataset']),
                                                                   flip=False,
                                                                   crop=False,
                                                                   batch_size=args.batch_size,
                                                                   correct_bgr=args.original_model_correct_bgr,
                                                                   repeat=False,
                                                                   num_parallel_calls=8)

    compressed_data_bpp = calculate_file_sizes(compressed_test_files(parameters['dataset']))
    compressed_model = tf.keras.models.load_model(str(experiment / 'final_model.hdf5'))
    compressed_model.compile('sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    c2o_crossentropy, c2o_accuracy = compressed_model.evaluate(original_data_for_compressed_model)
    c2c_crossentropy, c2c_accuracy = compressed_model.evaluate(compressed_data)
    o2c_crossentropy, o2c_accuracy = original_model.evaluate(compressed_data_for_original_model)

    results['bpp'].append(compressed_data_bpp)
    results['experiment_dir'].append(str(experiment))

    results['c2o_crossentropy'].append(c2o_crossentropy)
    results['c2o_accuracy'].append(c2o_accuracy)
    results['c2c_crossentropy'].append(c2c_crossentropy)
    results['c2c_accuracy'].append(c2c_accuracy)
    results['o2c_crossentropy'].append(o2c_crossentropy)
    results['o2c_accuracy'].append(o2c_accuracy)

pd.DataFrame(results).to_csv(args.output_file)
