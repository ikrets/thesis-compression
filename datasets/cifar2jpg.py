import numpy as np
import argparse
import subprocess
from pathlib import Path
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--quality', nargs='+', type=int, required=True)
parser.add_argument('--output_folder', type=str, required=True)
args = parser.parse_args()

dataset = list(Path(args.dataset).glob('**/*.png'))

for img_name in tqdm(dataset):
    img_name = str(img_name)
    for quality in args.quality:
        target_filename = img_name.replace(args.dataset, f'{args.output_folder}/quality{quality}').replace('.png', '.jpg')
        Path(target_filename).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(f'convert {img_name} -quality {quality} {target_filename}', shell=True)
