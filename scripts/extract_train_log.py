import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import argparse

from experiment import save_experiment_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_experiment_params(output_dir, args)
    sns.set(context='paper', font_scale=2)

    event_files = list(Path(args.experiment_dir).glob('*.tfevents.*'))
    if len(event_files) != 1:
        print(f'{len(event_files)} event files found instead of 1!')
        exit(1)

    df = []

    for e in tf.train.summary_iterator(str(event_files[0])):
        for v in e.summary.value:
            if v.tag == 'epoch_categorical_accuracy':
                df.append({'Epoch': e.step,
                           'Accuracy': v.simple_value,
                           'Split': 'train'})
            if v.tag == 'epoch_val_categorical_accuracy':
                df.append({'Epoch': e.step,
                           'Accuracy': v.simple_value,
                           'Split': 'val'})

    df = pd.DataFrame(df)
    sns.lineplot(data=df, x='Epoch', y='Accuracy', hue='Split')
    ax = plt.gca()
    ax.set_ybound(0.7, 1.002)

    plt.tight_layout()
    plt.savefig(output_dir / 'train_plot.pdf')
