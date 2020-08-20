import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from experiment import save_experiment_params

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', type=str, required=True)
parser.add_argument('--bpg_evaluation_csv', type=str, required=True)
parser.add_argument('output_dir', type=str)
args = parser.parse_args()

Path(args.output_dir).mkdir(parents=True, exist_ok=True)
save_experiment_params(args.output_dir, args)

bpg = pd.read_csv(args.bpg_evaluation_csv)
# do not count the header
bpg.loc[:, 'bpp'] -= 13 * 8 / 32 / 32
bpg.loc[:, 'downstream_loss'] = 'bpg'

combined_data = []
for evaluation_result in Path(args.experiment_dir).glob('**/evaluation_*/results.csv'):
    downstream_loss, alpha, dataset, _ = evaluation_result.parts[-4:]
    alpha = float(alpha.split('_')[1])
    dataset = dataset.split('_')[1]
    values = pd.read_csv(evaluation_result)

    combined_data.append({
        'downstream_loss': downstream_loss,
        'alpha': alpha,
        'dataset': dataset,
    })
    combined_data[-1].update(values.loc[0, ['bpp', 'o2c_accuracy', 'c2o_accuracy', 'c2c_accuracy']])

combined_data = pd.DataFrame(combined_data)
combined_data = pd.concat([combined_data, bpg], axis='rows')
combined_data = combined_data.sort_values('bpp')

sns.set()
for m in ['o2c_accuracy', 'c2o_accuracy', 'c2c_accuracy']:
    plt.axhline(y=0.944, label='o2o_accuracy', color='turquoise', linestyle='--')
    sns.lineplot(data=combined_data, x='bpp', y=m, hue='downstream_loss', marker='.')
    plt.ylim(0.8, 0.96)
    plt.title(m)
    plt.savefig(Path(args.output_dir) / 'by_loss_{}.pdf'.format(m))
    plt.close()

melted = pd.melt(combined_data, id_vars=['downstream_loss', 'alpha', 'dataset', 'bpp'],
                 value_vars=['o2c_accuracy', 'c2o_accuracy', 'c2c_accuracy'], var_name='accuracy_type',
                 value_name='accuracy_value')

for loss in melted.downstream_loss.unique():
    plt.axhline(y=0.944, label='o2o_accuracy', color='turquoise', linestyle='--')
    sns.lineplot(data=melted[melted['downstream_loss'] == loss], x='bpp', y='accuracy_value', hue='accuracy_type',
                 marker='.')
    if loss != 'bpg':
        plt.ylim(0.8, 0.96)
    else:
        plt.ylim(0.6, 0.96)

    plt.title(loss)
    plt.savefig(Path(args.output_dir) / 'by_accuracy_type_{}.pdf'.format(loss))
    plt.close()
