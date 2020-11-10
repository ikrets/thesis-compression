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
parser.add_argument('--downstream_O_model_weights', type=str)
parser.add_argument('--downstream_O_model_correct_bgr', action='store_true')
parser.add_argument('--output_dir', type=str, required=True)

args = parser.parse_args()

Path(args.output_dir).mkdir(parents=True, exist_ok=True)

sess = tf.keras.backend.get_session()


O_data, _ = datasets.imagenette.read_images(Path(args.uncompressed_dataset) / 'val')
O_data = datasets.imagenette.pipeline(O_data, batch_size=args.batch_size, is_training=False, repeat=False)
flatten_and_normalize = lambda item: (datasets.imagenette.normalize(item['X']), item['label'])
O_data = O_data.map(flatten_and_normalize, AUTO)

downstream_O_model = tf.keras.models.load_model(args.downstream_O_model, compile=False)
if args.downstream_O_model_weights:
    downstream_O_model.load_weights(args.downstream_O_model_weights)
downstream_O_model.compile('sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

o2o_accuracy = downstream_O_model.evaluate(O_data)[1]

with (Path(args.output_dir) / 'o2o_accuracy.txt').open('w') as fp:
    fp.write(str(o2o_accuracy))

save_experiment_params(Path(args.output_dir), args)
