import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from experiment import save_experiment_params

parser = argparse.ArgumentParser()
parser.add_argument('experiment_dir', type=str)
parser.add_argument('output_dir', type=str)
args = parser.parse_args()

Path(args.output_dir).mkdir(parents=True, exist_ok=True)
save_experiment_params(args.output_dir, args)

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

combined_data = pd.DataFrame(combined_data).sort_values('bpp')
sns.set()
for m in ['o2c_accuracy', 'c2o_accuracy', 'c2c_accuracy']:
    sns.lineplot(data=combined_data, x='bpp', y=m, hue='downstream_loss')
    plt.title(m)
    plt.savefig(Path(args.output_dir) / 'by_loss_{}.png'.format(m))

for loss in combined_data.downstream_loss.unique():
    # pivoted =
    sns.lineplot(data=combined_data, x='bpp', y=loss, hue='')
