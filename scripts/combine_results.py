import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from experiment import save_experiment_params

parser = argparse.ArgumentParser()
parser.add_argument('evaluation_dir', type=str)
parser.add_argument('output_dir', type=str)
args = parser.parse_args()

all_results = list(Path(args.evaluation_dir).glob('*/*/*/results.csv'))
combined_results = []

for result in tqdm(all_results, desc='result files'):
    result_row = {}

    experiment_type, result_row['compressor'], result_row['compressor_param'] = result.parent.parts[-3:]
    experiment_name_parts = experiment_type.split('_')
    result_row['dataset'] = experiment_name_parts[0]

    if len(experiment_name_parts) == 2:
        result_row['architecture_O'] = experiment_name_parts[1]
        result_row['architecture_C'] = experiment_name_parts[1]
    elif len(experiment_name_parts) == 3:
        if experiment_name_parts[1][0] != 'O' or experiment_name_parts[2][0] != 'C':
            raise RuntimeError('Unexpected experiment type: {}'.format(experiment_type))

        result_row['architecture_O'] = experiment_name_parts[1][1:]
        result_row['architecture_C'] = experiment_name_parts[2][1:]
    else:
        raise RuntimeError('Unexpected experiment type: {}'.format(experiment_type))

    result_df = pd.read_csv(result)

    if 'dataset' in result_df.columns:
        result_df.rename({c: c + '_1' for c in result_df.columns if 'accuracy' not in c},
                         axis='columns', inplace=True)
        for c in result_df.columns:
            if 'accuracy' not in c:
                result_df[c[:-2] + '_2'] = result_df[c]

    for c in result_df.columns:
        result_row[c] = result_df.loc[0, c]
    combined_results.append(result_row)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
save_experiment_params(output_dir, args)
combined_df = pd.DataFrame(combined_results)
combined_df.to_csv(output_dir / 'combined_results.csv')
