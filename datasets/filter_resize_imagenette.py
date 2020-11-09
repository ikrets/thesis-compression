import tensorflow as tf
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
from tqdm import tqdm

from experiment import save_experiment_params

AUTO = tf.data.experimental.AUTOTUNE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--min_image_size', type=int, default=300)
    parser.add_argument('--output_dataset', type=str, required=True)
    args = parser.parse_args()

    Path(args.output_dataset).mkdir(parents=True, exist_ok=True)
    save_experiment_params(args.output_dataset, args)

    images = [str(p) for p in Path(args.dataset).glob('**/*.JPEG')]
    for file_name in tqdm(images):
        image = np.array(Image.open(file_name))
        min_side = min(image.shape[0], image.shape[1])
        if min_side < args.min_image_size:
            continue

        h_start = (image.shape[0] - min_side) // 2
        w_start = (image.shape[1] - min_side) // 2
        image = image[h_start:h_start + min_side, w_start:w_start + min_side]
        image = Image.fromarray(image).resize((args.image_size, args.image_size), Image.BICUBIC)
        new_file_name = file_name.replace(args.dataset, args.output_dataset).replace('.JPEG', '.png')
        new_file_name = Path(new_file_name)
        new_file_name.parent.mkdir(parents=True, exist_ok=True)

        with open(new_file_name, 'wb') as fp:
            image.save(fp, format='png')
