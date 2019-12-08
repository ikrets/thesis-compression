import tensorflow as tf
from pathlib import Path
import argparse
import math
import pandas as pd
from datasets.cifar10 import pipeline
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str)
parser.add_argument('--dataset', type=str, nargs='+', required=True)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--result_dataframe', type=str, required=True)
args = parser.parse_args()

if not Path(args.result_dataframe).exists():
    df = pd.DataFrame(columns={'model', 'dataset', 'categorical_accuracy'})
else:
    df = pd.read_csv(args.result_dataframe)

model = tf.keras.models.load_model(args.model)
if args.weights:
    model.load_weights(args.weights)
model.compile('sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

rows = []
for dataset in tqdm(args.dataset):
    files = list(Path(dataset).glob('**/*.png'))
    data_test = pipeline(files, flip=False, crop=False, batch_size=args.batch_size, num_parallel_calls=8)
    result = model.evaluate(data_test, steps=math.ceil(len(files) / args.batch_size))
    rows.append({'model': args.model, 'dataset': dataset, 'categorical_accuracy': result[1]})

df = df.append(rows, ignore_index=True)
df.to_csv(args.result_dataframe, index=False)
