import tensorflow as tf
import optuna
import argparse
import math
from pathlib import Path
from coolname import generate_slug
import datasets.cifar10
from models.compressor import Compressor


def objective(trial, args):
    hparams = {
        'filters': trial.suggest_int('filters', 32, 192),
        'steps': trial.suggest_int('steps', 2, 5),
        'kernel_size': trial.suggest_categorical('kernel_size', [3, 5]),
        'lambda': 0.01,
        'downstream_loss_type': 'none',
        'scale_hyperprior': False
    }

    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    main_lr = trial.suggest_loguniform('main_lr', 1e-5, 1e-2)
    aux_lr = trial.suggest_loguniform('aux_lr', 1e-5, 1e-2)
    epochs = 201
    val_period = 5
    checkpoint_period = 20

    def make_schedule(base_lr, warmup, boundaries, reduction):
        def schedule(epoch):
            if epoch < warmup:
                return base_lr * (epoch + 1) / warmup

            index = next(i for i, b in enumerate(boundaries) if b > epoch)
            return base_lr * (reduction ** index)

        return schedule

    main_schedule = make_schedule(main_lr, boundaries=[140, 160, 180, 190], reduction=0.5, warmup=10)
    aux_schedule = make_schedule(aux_lr, boundaries=[140, 160, 180, 190], reduction=0.5, warmup=10)

    tf.reset_default_graph()

    model = tf.keras.models.load_model(args.model)
    if args.model_weights:
        model.load_weights(args.model_weights)

    compressor = Compressor(hparams, model=model)

    dataset = Path(args.dataset)

    with tf.device('/cpu:0'):
        train_filenames = list((dataset / 'train').glob('**/*.png'))
        train_dataset = datasets.cifar10.pipeline(filenames=train_filenames,
                                                  flip=True,
                                                  crop=False,
                                                  batch_size=batch_size,
                                                  num_parallel_calls=8)
        train_steps = math.ceil(len(train_filenames) / batch_size)
        val_filenames = list((dataset / 'test').glob('**/*.png'))
        val_dataset = datasets.cifar10.pipeline(filenames=val_filenames,
                                                flip=False,
                                                crop=False,
                                                batch_size=batch_size,
                                                num_parallel_calls=8)
        val_steps = math.ceil(len(val_filenames) / batch_size)

    final_val_losses = compressor.fit(train_dataset=train_dataset,
                                      train_steps=train_steps,
                                      val_dataset=val_dataset,
                                      val_steps=val_steps,
                                      epochs=epochs,
                                      main_lr_schedule=main_schedule,
                                      aux_lr_schedule=aux_schedule,
                                      val_period=val_period,
                                      checkpoint_period=checkpoint_period,
                                      experiments_dir=args.experiments_dir,
                                      trial_id=generate_slug()
                                      )

    trial.set_user_attr('bpp', final_val_losses['bpp'])
    trial.set_user_attr('psnr', final_val_losses['metric_psnr'])

    return final_val_losses['total']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--experiments_dir', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model_weights', type=str)
    args = parser.parse_args()

    Path(args.experiments_dir).mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name='find compression hyperparams',
                                storage=f'sqlite:///{args.experiments_dir}/study.db')
    obj = lambda trial: objective(trial, args)

    study.set_user_attr('args', vars(args))
    study.optimize(obj, n_trials=100)
