import argparse
import tensorflow.compat.v1 as tf
import numpy as np
from pathlib import Path
import pandas as pd

import datasets.imagenette
from experiment import save_experiment_params

AUTO = tf.data.experimental.AUTOTUNE

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--uncompressed_dataset', type=str, required=True)
parser.add_argument('--downstream_O_model')
parser.add_argument('--compressed_dataset_2', type=str, required=True,
                    help='the compressed dataset for evaluating performance in O2C and C2C')
parser.add_argument('--compressed_dataset_2_type', choices=['files', 'tfrecords'], default='tfrecords')
parser.add_argument('--output_dir', type=str, required=True)

args = parser.parse_args()

Path(args.output_dir).mkdir(parents=True, exist_ok=True)

sess = tf.keras.backend.get_session()


def load_compressed_dataset(dataset: str, dataset_type: str):
    if dataset_type == 'tfrecords':
        _, C_data = datasets.imagenette.read_compressed_tfrecords([dataset])
        num_examples = sess.run(C_data.reduce(np.int32(0), lambda acc, _: acc + 1))
        bpp = sess.run(C_data.reduce(np.float32(0.), lambda acc, item: acc + item['range_coded_bpp']) / num_examples)
        class_to_label_map = datasets.imagenette.get_class_to_label_map(args.uncompressed_dataset)

        def name_to_label(name):
            result = tf.py_func(lambda n: class_to_label_map[n.decode('ascii').split('/')[-2]],
                                [name],
                                tf.int64)
            result.set_shape(tuple())
            return result

        def add_label(item, label):
            item['label'] = label
            return item

        C_data_labels = C_data.map(lambda item: name_to_label(item['name']), AUTO)
        C_data = tf.data.Dataset.zip((C_data, C_data_labels))
        C_data = C_data.map(add_label, AUTO)

    elif dataset_type == 'files':
        C_data, _ = datasets.imagenette.read_images(Path(dataset) / 'val')
        num_examples = sess.run(C_data.reduce(np.int32(0), lambda acc, _: acc + 1))
        bpp = sess.run(datasets.imagenette.count_bpg_bpps(Path(dataset) / 'val'))
    else:
        assert False

    mse = C_data.map(lambda item: datasets.get_mse(item, args.uncompressed_dataset), AUTO)
    mse = sess.run(mse.reduce(np.float32(0.), lambda acc, v: acc + v / num_examples))

    return {'data': C_data,
            'mse': mse,
            'bpp': bpp}


compressed_dataset_2 = load_compressed_dataset(args.compressed_dataset_2, args.compressed_dataset_2_type)

downstream_O_model = tf.keras.models.load_model(args.downstream_O_model, compile=False)
downstream_O_model.compile('sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

O2C_data = datasets.imagenette.pipeline(compressed_dataset_2['data'],
                                        batch_size=args.batch_size,
                                        is_training=False,
                                        repeat=False)
O2C_data = O2C_data.map(datasets.dict_to_tuple, AUTO)
O2C_data = O2C_data.map(lambda img, label: (datasets.imagenette.normalize(img), label), AUTO)

results = {'dataset_2': args.compressed_dataset_2,
           'bpp_2': compressed_dataset_2['bpp'],
           'mse_2': compressed_dataset_2['mse'],
           'o2c_accuracy': downstream_O_model.evaluate(O2C_data)[1]
           }

results_df = pd.DataFrame([results])
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
results_df.to_csv(Path(args.output_dir) / 'results.csv', index=False)
save_experiment_params(Path(args.output_dir), args)
