import cv2
from pathlib import Path
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--extension', type=str, required=True)
parser.add_argument('--minimum_width', type=int, required=True)
parser.add_argument('--minimum_height', type=int, required=True)
parser.add_argument('--output_folder', type=str, required=True)
args = parser.parse_args()

dataset = list(Path(args.dataset).glob(f'**/*.{args.extension}'))
for file in tqdm(dataset):
    file = str(file)
    img = cv2.imread(file)
    if img.shape[0] >= args.minimum_height and img.shape[1] >= args.minimum_width:
        target_file = file.replace(args.dataset, args.output_folder)
        Path(target_file).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(target_file, img)