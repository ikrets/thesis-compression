import argparse
import tensorflow.compat.v1 as tf
import numpy as np
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import datasets.cifar10
from experiment import save_experiment_params

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--uncompressed_dataset', type=str, required=True)
parser.add_argument('--downstream_O_model', type=str, required=True)
parser.add_argument('--downstream_O_model_weights', type=str)
parser.add_argument('--downstream_O_model_correct_bgr', action='store_true')
parser.add_argument('--compressed_dataset', type=str, required=True)
parser.add_argument('--compressed_dataset_type', choices=['files', 'tfrecords'], default='tfrecords')
parser.add_argument('--downstream_C_model', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)

args = parser.parse_args()

Path(args.output_dir).mkdir(parents=True, exist_ok=True)

sess = tf.keras.backend.get_session()
O_data, O_examples = datasets.cifar10.read_images(Path(args.uncompressed_dataset) / 'test')
O_num_batches = np.ceil(O_examples / args.batch_size).astype(int)

C2O_data = datasets.cifar10.pipeline(O_data,
                                     batch_size=args.batch_size,
                                     flip=False,
                                     crop=False,
                                     classifier_normalize=True,
                                     repeat=False)

downstream_O_model = tf.keras.models.load_model(args.downstream_O_model, compile=False)
if args.downstream_O_model_weights:
    downstream_O_model.load_weights(args.downstream_O_model_weights)
downstream_O_model.compile('sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

downstream_C_model = tf.keras.models.load_model(args.downstream_C_model, compile=False)
downstream_C_model.compile('sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

if args.compressed_dataset_type == 'tfrecords':
    _, C_data = datasets.cifar10.read_compressed_tfrecords([args.compressed_dataset])
    bpp = sess.run(C_data.reduce(np.float32(0.), lambda acc, item: acc + item['range_coded_bpp']) / O_examples)
elif args.compressed_dataset_type == 'files':
    C_data, _ = datasets.cifar10.read_images(Path(args.compressed_dataset) / 'test')
    bpp = sess.run(datasets.cifar10.count_bpg_bpps(Path(args.compressed_dataset) / 'test'))

O2C_data = datasets.cifar10.pipeline(C_data,
                                     batch_size=args.batch_size,
                                     flip=False,
                                     crop=False,
                                     classifier_normalize=True,
                                     correct_bgr=args.downstream_O_model_correct_bgr,
                                     repeat=False)
C2C_data = datasets.cifar10.pipeline(C_data,
                                     batch_size=args.batch_size,
                                     flip=False,
                                     crop=False,
                                     classifier_normalize=True,
                                     repeat=False)

C2C_accuracy = downstream_C_model.evaluate(C2C_data)[1]
C2O_accuracy = downstream_C_model.evaluate(C2O_data)[1]
O2C_accuracy = downstream_O_model.evaluate(O2C_data)[1]

results = [{'dataset': args.compressed_dataset,
            'bpp': bpp,
            'c2o_accuracy': C2O_accuracy,
            'o2c_accuracy': O2C_accuracy,
            'c2c_accuracy': C2C_accuracy}]

results_df = pd.DataFrame(results)
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
results_df.to_csv(Path(args.output_dir) / 'results.csv', index=False)
save_experiment_params(Path(args.output_dir), args)
