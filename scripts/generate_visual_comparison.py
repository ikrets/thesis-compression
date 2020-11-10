import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from PIL import Image
import argparse
from pathlib import Path

from experiment import save_experiment_params

parser = argparse.ArgumentParser()
parser.add_argument('--uncompressed_dataset', type=str, required=True)
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--bpg_qps', type=int, nargs=3, required=True)
parser.add_argument('--compressor_alphas', type=float, nargs=3, required=True)
parser.add_argument('--compressor', type=str, required=True)
parser.add_argument('--activation', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()

input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
uncompressed_dataset = Path(args.uncompressed_dataset)

save_experiment_params(output_dir, args)

all_images = [p for p in input_dir.glob('*.png') if p.name[-8:] != '.mse.png']


def image_panel(original_img, bpg_reconstruction, bpg_mse, bpg_activation_mse,
                compressor_reconstruction, compressor_mse, compressor_activation_mse):
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    axes[0, 0].imshow(bpg_reconstruction)
    axes[0, 1].imshow(compressor_reconstruction)

    diff_mse = (compressor_mse - bpg_mse) / 2**16
    max_diff = np.max(np.abs(diff_mse))

    axes[1, 0].imshow(diff_mse, cmap='bwr', norm=colors.Normalize(-max_diff, max_diff))
    axes[1, 1].imshow(original_img)

    for a in axes.ravel():
        a.axis('off')

    fig.tight_layout()

    return fig


def get_full_names(name_starts):
    full_names = [None, None, None]
    for n in all_images:
        for i, ns in enumerate(name_starts):
            if n.name.find(ns) != -1:
                full_names[i] = n

    return full_names


def get_images(names):
    reconstructions = [np.array(Image.open(n)) for n in names]
    mses = [np.array(Image.open(str(n).replace('.png', '.mse.png'))) for n in names]
    activation_mses = [np.array(Image.open(str(n).replace('.png', '_{}.mse.png'.format(args.activation)))) for n in
                       names]

    return reconstructions, mses, activation_mses


original_image = Image.open(
    uncompressed_dataset / 'val' / input_dir.name.split('_')[0] / '{}.png'.format(input_dir.name))
bpg_examples = get_full_names(['bpg_qp_{}'.format(qp) for qp in args.bpg_qps])
compressor_examples = get_full_names(['{}_alpha_{}'.format(args.compressor, a) for a in args.compressor_alphas])
bpg_reconstructions, bpg_mses, bpg_activation_mses = get_images(bpg_examples)
compressor_reconstructions, compressor_mses, compressor_activation_mses = get_images(compressor_examples)

for i in range(3):
    fig = image_panel(original_image, bpg_reconstructions[i], bpg_mses[i], bpg_activation_mses[i],
                      compressor_reconstructions[i], compressor_mses[i], compressor_activation_mses[i])
    fig.savefig(output_dir / 'panel_{}.pdf'.format(i))
    plt.close(fig)
