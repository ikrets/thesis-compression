import numpy as np
import cv2
import pickle
from pathlib import Path
import argparse
from tqdm import tqdm, trange

parser = argparse.ArgumentParser()
parser.add_argument('--cifar_folder', type=str, required=True)
parser.add_argument('--output_folder', type=str, required=True)
args = parser.parse_args()

cifar_folder = Path(args.cifar_folder)
output_folder = Path(args.output_folder)

train = cifar_folder.glob('data_batch_*')
test = cifar_folder.glob('test_batch')


def extract_pngs(file, output_folder):
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(str(file), 'rb') as fp:
        images = pickle.load(fp, encoding='bytes')
        for i in trange(len(images[b'labels']), desc='images'):
            img_folder = output_folder / str(images[b'labels'][i])
            img_folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img_folder / images[b'filenames'][i].decode('ascii')),
                        np.transpose(images[b'data'][i].reshape(3, 32, 32), [1, 2, 0]))



for i, train_batch in enumerate(tqdm(train, desc='train files')):
    extract_pngs(train_batch, output_folder / 'train' / f'{i:02d}')

for i, test_batch in enumerate(tqdm(test, desc='test files')):
    extract_pngs(test_batch, output_folder / 'test' / f'{i:02d}')
