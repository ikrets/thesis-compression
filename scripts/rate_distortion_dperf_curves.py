import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from experiment import save_experiment_params

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', type=str, required=True)
parser.add_argument('--o2o_accuracy', type=float, required=True)
parser.add_argument('output_dir', type=str)
args = parser.parse_args()

Path(args.output_dir).mkdir(parents=True, exist_ok=True)
save_experiment_params(args.output_dir, args)

combined_data = []
for evaluation_result in Path(args.experiment_dir).glob('**/results.csv'):
    experiment_type, compressor_type, compressor_param, _ = evaluation_result.parts[-4:]
    dataset, model = experiment_type.split('_')
    values = pd.read_csv(evaluation_result)

    # do not count the header
    if compressor_type == 'bpg':
        values['bpp'] -= 13 * 8 / 32 / 32

    combined_data.append({
        'compressor_type': compressor_type,
        'compressor_param': compressor_param,
        'dataset': dataset,
        'model': model
    })
    combined_data[-1].update(values.loc[0, ['bpp', 'o2c_accuracy', 'c2o_accuracy', 'c2c_accuracy']])

combined_data = pd.DataFrame(combined_data)
combined_data = combined_data.sort_values('bpp')

sns.set()
for m in ['o2c_accuracy', 'c2o_accuracy', 'c2c_accuracy']:
    plt.axhline(y=args.o2o_accuracy, label='o2o_accuracy', color='turquoise', linestyle='--')
    sns.lineplot(data=combined_data, x='bpp', y=m, hue='compressor_type', marker='.')
    plt.ylim(0.8, 1.02 * args.o2o_accuracy)
    plt.title(m)
    plt.savefig(Path(args.output_dir) / 'by_loss_{}.pdf'.format(m))
    plt.close()

melted = pd.melt(combined_data, id_vars=['compressor_type', 'compressor_param', 'dataset', 'bpp'],
                 value_vars=['o2c_accuracy', 'c2o_accuracy', 'c2c_accuracy'], var_name='accuracy_type',
                 value_name='accuracy_value')

for loss in melted.compressor_type.unique():
    plt.axhline(y=args.o2o_accuracy, label='o2o_accuracy', color='turquoise', linestyle='--')
    sns.lineplot(data=melted[melted['compressor_type'] == loss], x='bpp', y='accuracy_value', hue='accuracy_type',
                 marker='.')
    if loss != 'bpg':
        plt.ylim(0.8, 1.02 * args.o2o_accuracy)
    else:
        plt.ylim(0.6, 1.02 * args.o2o_accuracy)

    plt.title(loss)
    plt.savefig(Path(args.output_dir) / 'by_accuracy_type_{}.pdf'.format(loss))
    plt.close()
