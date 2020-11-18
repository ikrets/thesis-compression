import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from experiment import save_experiment_params

parser = argparse.ArgumentParser()
parser.add_argument('--combined_csv', type=str, required=True)
parser.add_argument('--o2o_accuracy_imagenette', type=float, required=True)
parser.add_argument('--o2o_accuracy_imagewoof', type=float, required=True)
parser.add_argument('output_dir', type=str)
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
save_experiment_params(output_dir, args)

sns.set(context='paper', font_scale=1.4)

combined_df = pd.read_csv(args.combined_csv)
combined_df = combined_df[combined_df.dataset.isin(['imagenette', 'imagewoof'])]
combined_df = combined_df[combined_df.compressor.isin(['activation_7_hyperprior', 'bpg'])]
# do not count the header for bpg bitrates
combined_df.loc[combined_df.compressor == 'bpg', ['bpp_1', 'bpp_2']] -= 13 * 8 / 256 / 256
combined_df.sort_values(['bpp_1', 'bpp_2', 'architecture_O', 'architecture_C'])

# 1. Each compressor separately
samearc_grouped = combined_df.groupby(['dataset', 'compressor'])
for (dataset, compressor), df in samearc_grouped:
    o2o_accuracy = vars(args)[f'o2o_accuracy_{dataset}']

    if compressor != 'bpg':
        df['alpha'] = df.compressor_param.apply(lambda s: float(s.split('_')[1]))

    _, axes = plt.subplots(1, 3 if compressor != 'bpg' else 2, figsize=(3.3 * (3 if compressor != 'bpg' else 2), 3.5))

    axes[0].axhline(y=o2o_accuracy, label='O2O accuracy', color='turquoise', linestyle='--')
    sns.lineplot(data=df, x='bpp_2', y='o2c_accuracy', ax=axes[0])
    axes[0].set_ylim(top=1.01 * o2o_accuracy)

    sns.lineplot(data=df, x='bpp_2', y='mse_2', ax=axes[1])

    if compressor != 'bpg':
        sns.lineplot(data=df, x='alpha', y='bpp_2', ax=axes[2])

    target_file = output_dir / 'separate_compressor' / dataset / 'compressor_{}.pdf'.format(compressor)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(target_file)
    plt.close()

relabel = {
    'activation_7_hyperprior': 'Perceptual, block3',
    'bpg': 'BPG baseline'
}
for dataset, df in combined_df.groupby(['dataset']):
    o2o_accuracy = vars(args)[f'o2o_accuracy_{dataset}']
    _, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    g = sns.lineplot(x=df.bpp_2, y=df['o2c_accuracy'] / o2o_accuracy,
                 hue=df.compressor, ax=axes[0], marker='o')
    for t in g.legend().texts:
        text = t.get_text()
        if text in relabel:
            t.set_text(relabel[text])

    axes[0].set_title('o2c accuracy')
    axes[0].set_xlabel('bpp')
    axes[0].set_ylabel('% of O2O accuracy')
    sec_ax = axes[0].secondary_yaxis('right',
                                     functions=(lambda x: x * o2o_accuracy,
                                                lambda y: y / o2o_accuracy))
    sec_ax.set_ylabel('O2C accuracy')

    g = sns.lineplot(x=df.bpp_2, y=df.mse_2, hue=df.compressor, ax=axes[1], marker='o')
    for t in g.legend().texts:
        text = t.get_text()
        if text in relabel:
            t.set_text(relabel[text])
    axes[1].set(xlabel='bpp', ylabel='Reconstruction MSE')

    if dataset == 'imagenette':
        axes[0].set_xbound(0.10, 0.28)
        axes[0].set_ybound(0.75, 1.02)
        axes[1].set_xbound(0.10, 0.28)
        axes[1].set_ybound(0.001, 0.0045)
    if dataset == 'imagewoof':
        axes[0].set_xbound(0.10, 0.28)
        axes[0].set_ybound(0.75, 1.02)
        axes[1].set_xbound(0.10, 0.28)
        axes[1].set_ybound(0.001, 0.0045)


    plt.tight_layout()
    target_file = output_dir / 'compare_compressors' / dataset / 'o2c_accuracy.pdf'
    target_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(target_file)
    plt.close()
