import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from experiment import save_experiment_params

parser = argparse.ArgumentParser()
parser.add_argument('--combined_csv', type=str, required=True)
parser.add_argument('--o2o_accuracy', nargs=2, metavar=('ARCHITECTURE', 'VALUE'), action='append')
parser.add_argument('output_dir', type=str)
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
save_experiment_params(output_dir, args)

sns.set(context='paper', font_scale=1.4)

o2o_values = {architecture: float(value) for architecture, value in args.o2o_accuracy}

combined_df = pd.read_csv(args.combined_csv)
combined_df = combined_df[combined_df.dataset == 'cifar10']
# do not count the header for bpg bitrates
combined_df.loc[combined_df.compressor == 'bpg', ['bpp_1', 'bpp_2']] -= 13 * 8 / 32 / 32
combined_df.sort_values(['bpp_1', 'bpp_2', 'architecture_O', 'architecture_C'])

unspecified_o2o_values = set(combined_df.architecture_O) | set(combined_df.architecture_C)
unspecified_o2o_values.difference_update(o2o_values.keys())
if unspecified_o2o_values:
    print('Please specify O2O accuracy for the following architectures: {}'.format(', '.join(unspecified_o2o_values)))
    exit(1)

samearc_df = combined_df[combined_df.architecture_O == combined_df.architecture_C]

# 1. Each compressor separately
samearc_grouped = samearc_df.groupby(['architecture_O', 'compressor'])
for (architecture, compressor), df in samearc_grouped:
    df = df.loc[(df.bpp_1 > 0.4) & (df.bpp_1 < 2.2)].copy()
    if compressor != 'bpg':
        df['alpha'] = df.compressor_param.apply(lambda s: float(s.split('_')[1]))
    value_vars = ['o2c_accuracy', 'c2o_accuracy', 'c2c_accuracy']
    df = df.melt(id_vars=[c for c in set(df.columns).difference(value_vars)],
                 value_vars=value_vars,
                 value_name='Accuracy value',
                 var_name='Evaluation setup')

    _, axes = plt.subplots(1, 3 if compressor != 'bpg' else 2, figsize=(3.3 * (3 if compressor != 'bpg' else 2), 3.5))

    axes[0].axhline(y=o2o_values[architecture], label='O2O accuracy', color='turquoise', linestyle='--')
    g = sns.lineplot(data=df, x='bpp_1', y='Accuracy value', hue='Evaluation setup', ax=axes[0], marker='o')
    axes[0].set_ylim(bottom=0.63 if architecture == 'resnet18' else 0.68,
                     top=1.01 * o2o_values[architecture])
    axes[0].set_xlim(left=0.5, right=2.2)
    axes[0].set(xlabel='bpp')
    relabel = {'o2c_accuracy': 'O2C',
               'c2o_accuracy': 'C2O',
               'c2c_accuracy': 'C2C'}
    for t in g.legend().texts:
        text = t.get_text()
        if text in relabel:
            t.set_text(relabel[text])

    sns.lineplot(data=df, x='bpp_1', y='mse_1', ax=axes[1], marker='o')
    axes[1].set(xlabel='bpp', ylabel='Reconstruction MSE')
    axes[1].set_xlim(left=0.5, right=2.2)
    axes[1].set_ylim(bottom=0.00025, top=0.006)

    if compressor != 'bpg':
        sns.lineplot(data=df, x='alpha', y='bpp_1', ax=axes[2], marker='o')
        axes[2].set(xlabel=r'$\alpha$', ylabel='bpp')
        axes[2].set_ylim(bottom=0.5, top=2.2)
        formatter = ticker.ScalarFormatter()
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        axes[2].xaxis.set_major_formatter(formatter)

    target_file = output_dir / 'separate_compressor' / architecture / 'compressor_{}.pdf'.format(compressor)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(target_file)
    plt.close()

# 2. All compressors combined, each evaluation type separately
samearc_grouped = samearc_df.groupby('architecture_O')
rename_losses = {
    'act_3_norm': 'Perceptual, block2',
    'act_7_norm': 'Perceptual, block3',
    'act_11_norm': 'Perceptual, block4',
    'block2_conv2': 'Perceptual, block2',
    'block3_conv3': 'Perceptual, block3',
    'block4_conv3': 'Perceptual, block4',
    'block2_conv2 block3_conv3 block4_conv3': 'Perceptual, blocks 2-4',
    'bpg': 'BPG baseline',
    'combined_act_3_7_11': 'Perceptual, blocks 2-4',
    'prediction_crossentropy': 'Prediction cross-entropy',
    'task_crossentropy': 'Task cross-entropy'
}
losses_order = ['Perceptual, block2', 'Perceptual, block3', 'Perceptual, block4',
                'Perceptual, blocks 2-4', 'Prediction cross-entropy', 'Task cross-entropy',
                'BPG baseline']
for accuracy_type in ['o2c_accuracy', 'c2o_accuracy', 'c2c_accuracy']:
    _, axes = plt.subplots(2, 1, figsize=(10, 12))
    for i, (architecture, df) in enumerate(samearc_grouped):
        rename_losses_inv = {v: k for k, v in rename_losses.items() if k in df.compressor.unique()}
        hue_order = [rename_losses_inv[v] for v in losses_order]

        g = sns.lineplot(x=df.bpp_1, y=df[accuracy_type] / o2o_values[architecture],
                     hue=df.compressor, hue_order=hue_order, ax=axes[i], marker='o')
        rename_architecture = {'vgg16': 'VGG16', 'resnet18': 'Resnet18'}
        axes[i].set_title(f'{rename_architecture[architecture]} architecture')
        axes[i].set(xlabel='bpp', ylabel='% of O2O accuracy')
        for t in g.legend().texts:
            text = t.get_text()
            if text in rename_losses:
                t.set_text(rename_losses[text])

        sec_ax = axes[i].secondary_yaxis('right',
                                         functions=(lambda x, architecture=architecture: x * o2o_values[architecture],
                                                    lambda y, architecture=architecture: y / o2o_values[architecture]))
        rename = lambda s: f'{s[:3].upper()} accuracy value'
        sec_ax.set_ylabel(rename(accuracy_type))

        axes[i].set_xbound(0.5, 2.2)
        if accuracy_type == 'o2c_accuracy':
            axes[i].set_ybound(0.67, 1)
        elif accuracy_type == 'c2o_accuracy':
            axes[i].set_ybound(0.92, 1)
        else:
            axes[i].set_ybound(0.87, 1)

    plt.tight_layout()
    target_file = output_dir / 'compare_compressors' / '{}.pdf'.format(accuracy_type)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(target_file)
    plt.close()

# cross-architecture o2c plots
crossarc_df = combined_df[combined_df.architecture_O != combined_df.architecture_C]
crossarc_grouped = crossarc_df.groupby('architecture_O')
best_performing = {
    'vgg16': 'block3_conv3',
    'resnet18': 'act_11_norm'
}
_, axes = plt.subplots(2, 1, figsize=(10, 12))

for i, (architecture, df) in enumerate(crossarc_grouped):
    bpg_df = combined_df[(combined_df.compressor == 'bpg') & (combined_df.architecture_O == architecture)]
    best_performing_df = combined_df[(combined_df.compressor == best_performing[architecture])
                                     & (combined_df.architecture_O == architecture) & (
                                             combined_df.architecture_C == architecture)].copy()
    if architecture == 'vgg16':
        best_performing_compressor = 'Best performing VGG16 O2C (Perceptual, block4)'
    else:
        best_performing_compressor = 'Best performing Resnet18 O2C (Perceptual, block3)'
    best_performing_df['compressor'] = best_performing_compressor

    df = pd.concat([df, bpg_df, best_performing_df])

    rename_losses_inv = {v: k for k, v in rename_losses.items() if k in df.compressor.unique()}
    hue_order = [rename_losses_inv[v] for v in losses_order] + [best_performing_compressor]

    g = sns.lineplot(x=df.bpp_2, y=df['o2c_accuracy'] / o2o_values[architecture],
                 hue=df.compressor, hue_order=hue_order, ax=axes[i],
                 marker='o')

    for t in g.legend().texts:
        text = t.get_text()
        if text in rename_losses:
            t.set_text(rename_losses[text])

    sec_ax = axes[i].secondary_yaxis('right',
                                     functions=(lambda x, a=architecture: x * o2o_values[a],
                                                lambda y, a=architecture: y / o2o_values[a]))
    sec_ax.set_ylabel('O2C accuracy')
    axes[i].set_xlabel('bpp')
    axes[i].set_ylabel('% of O2O accuracy')
    if architecture == 'vgg16':
        axes[i].set_title('VGG16 O model evaluated on Resnet18 C data')
    else:
        axes[i].set_title('Resnet18 O model evaluated on VGG16 C data')
    axes[i].set_xbound(0.5, 2.2)
    axes[i].set_ybound(0.75, 1)

target_file = output_dir / 'cross_architecture' / 'o2c.pdf'
target_file.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(target_file)
plt.close()