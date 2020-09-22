import argparse
import tensorflow.compat.v1 as tf
import numpy as np
from pathlib import Path
import pandas as pd

import datasets.cifar10
from experiment import save_experiment_params

AUTO = tf.data.experimental.AUTOTUNE

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--uncompressed_dataset', type=str, required=True)
parser.add_argument('--downstream_O_model')
parser.add_argument('--downstream_O_model_weights', type=str)
parser.add_argument('--downstream_O_model_correct_bgr', action='store_true')
parser.add_argument('--compressed_dataset_1', type=str, required=True,
                    help='the compressed dataset used to train the downstream_C_model')
parser.add_argument('--compressed_dataset_1_type', choices=['files', 'tfrecords'], default='tfrecords')
parser.add_argument('--compressed_dataset_2', type=str, required=True,
                    help='the compressed dataset for evaluating performance in O2C and C2C')
parser.add_argument('--compressed_dataset_2_type', choices=['files', 'tfrecords'], default='tfrecords')
parser.add_argument('--downstream_C_model', type=str, required=True)
parser.add_argument('--skip_C2C', action='store_true')
parser.add_argument('--output_dir', type=str, required=True)

args = parser.parse_args()

Path(args.output_dir).mkdir(parents=True, exist_ok=True)

sess = tf.keras.backend.get_session()


def load_compressed_dataset(dataset: str, dataset_type: str):
    if dataset_type == 'tfrecords':
        _, C_data = datasets.cifar10.read_compressed_tfrecords([dataset])
        num_examples = sess.run(C_data.reduce(np.int32(0), lambda acc, _: acc + 1))
        bpp = sess.run(C_data.reduce(np.float32(0.), lambda acc, item: acc + item['range_coded_bpp']) / num_examples)
    elif dataset_type == 'files':
        C_data, _ = datasets.cifar10.read_images(Path(dataset) / 'test')
        num_examples = sess.run(C_data.reduce(np.int32(0), lambda acc, _: acc + 1))
        bpp = sess.run(datasets.cifar10.count_bpg_bpps(Path(dataset) / 'test'))
    else:
        assert False

    mse = C_data.map(lambda item: datasets.get_mse(item, args.uncompressed_dataset), AUTO)
    mse = sess.run(mse.reduce(np.float32(0.), lambda acc, v: acc + v / num_examples))

    return {'data': C_data,
            'mse': mse,
            'bpp': bpp}


downstream_C_model = tf.keras.models.load_model(args.downstream_C_model, compile=False)
downstream_C_model.compile('sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

compressed_dataset_1 = load_compressed_dataset(args.compressed_dataset_1, args.compressed_dataset_1_type)
compressed_dataset_2 = load_compressed_dataset(args.compressed_dataset_2, args.compressed_dataset_2_type)

if args.downstream_O_model:
    O_data, _ = datasets.cifar10.read_images(Path(args.uncompressed_dataset) / 'test')
    C2O_data = datasets.cifar10.pipeline(O_data,
                                         batch_size=args.batch_size,
                                         flip=False,
                                         crop=False,
                                         classifier_normalize=True,
                                         repeat=False)
    C2O_data = C2O_data.map(datasets.dict_to_tuple, AUTO)

    downstream_O_model = tf.keras.models.load_model(args.downstream_O_model, compile=False)
    if args.downstream_O_model_weights:
        downstream_O_model.load_weights(args.downstream_O_model_weights)
    downstream_O_model.compile('sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

results = {'dataset_1': args.compressed_dataset_1,
           'bpp_1': compressed_dataset_1['bpp'],
           'mse_1': compressed_dataset_1['mse'],
           'dataset_2': args.compressed_dataset_2,
           'bpp_2': compressed_dataset_2['bpp'],
           'mse_2': compressed_dataset_2['mse'],
           }

if args.downstream_O_model:
    O2C_data = datasets.cifar10.pipeline(compressed_dataset_2['data'],
                                         batch_size=args.batch_size,
                                         flip=False,
                                         crop=False,
                                         classifier_normalize=True,
                                         correct_bgr=args.downstream_O_model_correct_bgr,
                                         repeat=False)
    O2C_data = O2C_data.map(datasets.dict_to_tuple, AUTO)
    results['c2o_accuracy'] = downstream_C_model.evaluate(C2O_data)[1]
    results['o2c_accuracy'] = downstream_O_model.evaluate(O2C_data)[1]

if not args.skip_C2C:
    C2C_data = datasets.cifar10.pipeline(compressed_dataset_2['data'],
                                         batch_size=args.batch_size,
                                         flip=False,
                                         crop=False,
                                         classifier_normalize=True,
                                         repeat=False)
    C2C_data = C2C_data.map(datasets.dict_to_tuple, AUTO)
    results['c2c_accuracy'] = downstream_C_model.evaluate(C2C_data)[1]

results_df = pd.DataFrame([results])
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
results_df.to_csv(Path(args.output_dir) / 'results.csv', index=False)
save_experiment_params(Path(args.output_dir), args)
