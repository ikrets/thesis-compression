import argparse
import tensorflow.compat.v1 as tf
import numpy as np
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import datasets.cifar10

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--uncompressed_dataset', type=str, required=True)
parser.add_argument('--downstream_O_model', type=str, required=True)
parser.add_argument('--downstream_O_model_weights', type=str)
parser.add_argument('--downstream_O_model_correct_bgr', action='store_true')
parser.add_argument('--compressed_dataset_dir', type=str, required=True)
parser.add_argument('--downstream_C_models_dir', type=str, required=True)
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

downstream_C_model_filenames = [e.parent for e in Path(args.downstream_C_models_dir).glob('*/*/final_model.hdf5')]

results = []
for downstream_C_model_filename in tqdm(downstream_C_model_filenames, desc='Downstream C model'):
    with (downstream_C_model_filename / 'parameters.json').open('r') as fp:
        parameters = json.load(fp)

    downstream_C_model = tf.keras.models.load_model(downstream_C_model_filename / 'final_model.hdf5', compile=False)
    downstream_C_model.compile('sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    _, C_data = datasets.cifar10.read_compressed_tfrecords([Path(args.compressed_dataset_dir) / parameters['dataset']])
    bpp = sess.run(C_data.reduce(np.float32(0.), lambda acc, item: acc + item['range_coded_bpp']) / O_examples)
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

    results.append({'dataset': downstream_C_model_filename.parent,
                    'bpp': bpp * 8,
                    'c2o_accuracy': C2O_accuracy,
                    'o2c_accuracy': O2C_accuracy,
                    'c2c_accuracy': C2C_accuracy})

results_df = pd.DataFrame(results)
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
results_df.to_csv(Path(args.output_dir) / 'results.csv', index=False)
