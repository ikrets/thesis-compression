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
combined_df.sort_values(['bpp_1', 'bpp_2'])

unspecified_o2o_values = set(combined_df.architecture_O) | set(combined_df.architecture_C)
unspecified_o2o_values.difference_update(o2o_values.keys())
if unspecified_o2o_values:
    print('Please specify O2O accuracy for the following architectures: {}'.format(', '.join(unspecified_o2o_values)))
    exit(1)

# 1. Each compressor separately
samearc_df = combined_df[combined_df.architecture_O == combined_df.architecture_C]
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

# cross-architecture plots
crossarc_df = combined_df[combined_df.architecture_O != combined_df.architecture_C]

# sns.set()
# for m in ['o2c_accuracy', 'c2o_accuracy', 'c2c_accuracy']:
#     plt.axhline(y=args.o2o_accuracy, label='o2o_accuracy', color='turquoise', linestyle='--')
#     sns.lineplot(data=combined_data, x='bpp', y=m, hue='compressor_type', marker='.')
#     plt.ylim(0.8, 1.02 * args.o2o_accuracy)
#     plt.title(m)
#     plt.savefig(Path(args.output_dir) / 'by_loss_{}.pdf'.format(m))
#     plt.close()
#
# melted = pd.melt(combined_data, id_vars=['compressor_type', 'compressor_param', 'dataset', 'bpp'],
#                  value_vars=['o2c_accuracy', 'c2o_accuracy', 'c2c_accuracy'], var_name='accuracy_type',
#                  value_name='accuracy_value')
#
# for loss in melted.compressor_type.unique():
#     plt.axhline(y=args.o2o_accuracy, label='o2o_accuracy', color='turquoise', linestyle='--')
#     sns.lineplot(data=melted[melted['compressor_type'] == loss], x='bpp', y='accuracy_value', hue='accuracy_type',
#                  marker='.')
#     if loss != 'bpg':
#         plt.ylim(0.8, 1.02 * args.o2o_accuracy)
#     else:
#         plt.ylim(0.6, 1.02 * args.o2o_accuracy)
#
#     plt.title(loss)
#     plt.savefig(Path(args.output_dir) / 'by_accuracy_type_{}.pdf'.format(loss))
#     plt.close()
