import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from experiment import save_experiment_params

parser = argparse.ArgumentParser()
parser.add_argument('--combined_csv', type=str, required=True)
parser.add_argument('--o2o_accuracy', type=float, required=True)
parser.add_argument('output_dir', type=str)
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
save_experiment_params(output_dir, args)

sns.set(context='paper')

combined_df = pd.read_csv(args.combined_csv)
combined_df = combined_df[combined_df.dataset == 'imagenette']
# do not count the header for bpg bitrates
combined_df.loc[combined_df.compressor == 'bpg', ['bpp_1', 'bpp_2']] -= 13 * 8 / 256 / 256
combined_df.sort_values(['bpp_1', 'bpp_2', 'architecture_O', 'architecture_C'])

# 1. Each compressor separately
samearc_grouped = combined_df.groupby(['compressor'])
for compressor, df in samearc_grouped:
    if compressor != 'bpg':
        df['alpha'] = df.compressor_param.apply(lambda s: float(s.split('_')[1]))

    _, axes = plt.subplots(1, 3 if compressor != 'bpg' else 2, figsize=(3.3 * (3 if compressor != 'bpg' else 2), 3.5))

    axes[0].axhline(y=args.o2o_accuracy, label='o2o accuracy', color='turquoise', linestyle='--')
    sns.lineplot(data=df, x='bpp_2', y='o2c_accuracy', ax=axes[0])
    axes[0].set_ylim(top=1.01 * args.o2o_accuracy)

    sns.lineplot(data=df, x='bpp_2', y='mse_2', ax=axes[1])

    if compressor != 'bpg':
        sns.lineplot(data=df, x='alpha', y='bpp_2', ax=axes[2])

    target_file = output_dir / 'separate_compressor' / 'compressor_{}.pdf'.format(compressor)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(target_file)
    plt.close()

_, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].axhline(y=1, label='o2o accuracy', color='turquoise', linestyle='--')
sns.lineplot(x=combined_df.bpp_2, y=combined_df['o2c_accuracy'] / args.o2o_accuracy,
             hue=combined_df.compressor, ax=axes[0])
axes[0].set_title('o2c accuracy')
axes[0].set_ylabel('% of o2o accuracy')
sec_ax = axes[0].secondary_yaxis('right',
                                 functions=(lambda x: x * args.o2o_accuracy,
                                            lambda y: y / args.o2o_accuracy))
sec_ax.set_ylabel('o2c accuracy value')

axes[0].set_xbound(0.14, 0.3)
axes[0].set_ybound(0.9, 1.02)

sns.lineplot(x=combined_df.bpp_2, y=combined_df.mse_2, hue=combined_df.compressor, ax=axes[1])
axes[1].set_xbound(0.14, 0.3)
axes[1].set_ybound(0.001, 0.004)

plt.tight_layout()
target_file = output_dir / 'compare_compressors' / 'o2c_accuracy.pdf'
target_file.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(target_file)
plt.close()
