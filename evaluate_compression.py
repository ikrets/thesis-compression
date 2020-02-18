import tensorflow as tf
from pathlib import Path
import argparse
import math
import pandas as pd
import cv2
import numpy as np
from skimage.measure import compare_psnr
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--original_dataset', type=str, required=True)
parser.add_argument('--compressed_dataset', type=str, nargs='+', required=True)
parser.add_argument('--compressed_extension', choices=['bpg', 'tfci'], required=True)
parser.add_argument('--result_dataframe', type=str, required=True)
args = parser.parse_args()

if not Path(args.result_dataframe).exists():
    df = pd.DataFrame(columns={'original_dataset', 'compressed_dataset', 'bpp', 'psnr'})
else:
    df = pd.read_csv(args.result_dataframe)

original_dataset = Path(args.original_dataset)
original_files = {}
files = list(original_dataset.glob('**/*.png'))
for file in tqdm(files, desc='reading the original dataset'):
    original_files[file.name] = cv2.imread(str(file))

rows = []
for dataset in tqdm(args.compressed_dataset, desc='processing compressed datasets'):
    files = list(Path(dataset).glob('**/*.png'))

    bpps = []
    psnrs = []

    for file in tqdm(files, desc='file'):
        image = cv2.imread(str(file))
        compressed_file = Path(file.parent / (str(file.stem) + f'.{args.compressed_extension}'))
        bpps.append(compressed_file.stat().st_size / image.shape[0] / image.shape[1] * 8)
        psnrs.append(compare_psnr(original_files[file.name], image))

    rows.append(
        {'original_dataset': args.original_dataset, 'compressed_dataset': dataset, 'bpp': np.mean(bpps),
         'psnr': np.mean(psnrs)})

df = df.append(rows, ignore_index=True)
df.to_csv(args.result_dataframe, index=False)
