import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

sns.set(context='paper')

o2o_values = {architecture: float(value) for architecture, value in args.o2o_accuracy}

combined_df = pd.read_csv(args.combined_csv)
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
    if compressor != 'bpg':
        df['alpha'] = df.compressor_param.apply(lambda s: float(s.split('_')[1]))
    value_vars = ['o2c_accuracy', 'c2o_accuracy', 'c2c_accuracy']
    df = df.melt(id_vars=[c for c in set(df.columns).difference(value_vars)],
                 value_vars=value_vars,
                 value_name='accuracy value',
                 var_name='accuracy type')

    _, axes = plt.subplots(1, 3 if compressor != 'bpg' else 2, figsize=(3.3 * (3 if compressor != 'bpg' else 2), 3.5))

    axes[0].axhline(y=o2o_values[architecture], label='o2o accuracy', color='turquoise', linestyle='--')
    sns.lineplot(data=df, x='bpp_1', y='accuracy value', hue='accuracy type', ax=axes[0])
    axes[0].set_ylim(top=1.01 * o2o_values[architecture])

    sns.lineplot(data=df, x='bpp_1', y='mse_1', ax=axes[1])

    if compressor != 'bpg':
        sns.lineplot(data=df, x='alpha', y='bpp_1', ax=axes[2])

    target_file = output_dir / 'separate_compressor' / architecture / 'compressor_{}.pdf'.format(compressor)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(target_file)
    plt.close()

# 2. All compressors combined, each evaluation type separately
samearc_grouped = samearc_df.groupby('architecture_O')
for accuracy_type in ['o2c_accuracy', 'c2o_accuracy', 'c2c_accuracy']:
    _, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for i, (architecture, df) in enumerate(samearc_grouped):
        sns.lineplot(x=df.bpp_1, y=df[accuracy_type] / o2o_values[architecture],
                     hue=df.compressor, ax=axes[i])
        axes[i].set_title(architecture)
        axes[i].set_ylabel('% of o2o accuracy')
        sec_ax = axes[i].secondary_yaxis('right',
                                         functions=(lambda x, architecture=architecture: x * o2o_values[architecture],
                                                    lambda y, architecture=architecture: y / o2o_values[architecture]))
        sec_ax.set_ylabel('{} value'.format(accuracy_type))
        axes[i].set_xbound(0.5, 2.5)
        axes[i].set_ybound(0.8, 1)

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
_, axes = plt.subplots(1, 2, figsize=(10, 3.5))

for i, (architecture, df) in enumerate(crossarc_grouped):
    bpg_df = combined_df[(combined_df.compressor == 'bpg') & (combined_df.architecture_O == architecture)]
    best_performing_df = combined_df[(combined_df.compressor == best_performing[architecture])
                                     & (combined_df.architecture_O == architecture) & (
                                             combined_df.architecture_C == architecture)].copy()
    best_performing_df['compressor'] = '{} (best performing)'.format(best_performing[architecture])
    df = pd.concat([df, bpg_df, best_performing_df])
    df.loc[df.compressor == 'bpg', 'compressor'] = 'bpg (baseline)'

    sns.lineplot(x=df.bpp_2, y=df['o2c_accuracy'] / o2o_values[architecture],
                 hue=df.compressor, ax=axes[i])
    sec_ax = axes[i].secondary_yaxis('right',
                                     functions=(lambda x, a=architecture: x * o2o_values[a],
                                                lambda y, a=architecture: y / o2o_values[a]))
    sec_ax.set_ylabel('value')
    axes[i].set_ylabel('% of o2o accuracy')
    axes[i].set_title('{} O model evaluated on {} C data'.format(
        architecture, 'vgg16' if architecture == 'resnet18' else 'resnet18'))
    axes[i].set_xbound(0.5, 2.5)
    axes[i].set_ybound(0.75, 1)

target_file = output_dir / 'cross_architecture' / 'o2c.pdf'
target_file.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(target_file)
plt.close()