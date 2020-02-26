import subprocess
import numpy as np
import shlex

while True:
    num_filters = 64
    depth = 3
    film_depth = 3
    film_width = 128
    # batchsize = np.random.choice([16, 32, 64])
    batchsize = 16
    epochs = 101
    min_max_bpp = (0.4, 2.7)

    # downstream_loss_type = np.random.choice(['activation_difference', 'task_performance'])
    downstream_loss_type = 'task_performance'
    burnin_epochs = 80
    alpha_burnin = True
    last_frozen_layer = np.random.choice(['dense', 'flatten'])

    if downstream_loss_type == 'activation_difference':
        downstream_loss_layer = f'activation_{np.random.choice([3, 7, 9])}'
        loss_type_specific = f'--downstream_layer {downstream_loss_layer}'

        while True:
            lambda_mult = np.random.uniform(.1, 10)
            alpha_mult = np.random.uniform(.1, 10)

            lambda_low = 7e-5 * lambda_mult
            lambda_high = 0.0025 * lambda_mult
            alpha_low = 0.0003 * alpha_mult
            alpha_high = 0.015 * alpha_mult

            if lambda_low < lambda_high and alpha_low < alpha_high:
                break

    if downstream_loss_type == 'task_performance':
        loss_type_specific = '--model_performance_loss categorical_crossentropy'

        alpha_mult = np.random.uniform(.1, 10)
        alpha_low_high_mult = 10 ** np.random.uniform(1, 2)

        lambda_low = 7e-5
        lambda_high = 0.0025
        alpha_low = 0.00001 * alpha_mult
        alpha_high = alpha_low_high_mult * alpha_low

    sample_function = np.random.choice(['uniform', 'loguniform'])

    lr = 1e-4
    opt = 'adam'

    args = f'python train_film_compressor_comparison.py ' \
           '--correct_bgr ' \
           f'--num_filters {num_filters} --depth {depth} --num_postproc 0 --film_depth {film_depth} ' \
           f'--film_width {film_width} --film_activation relu --epochs {epochs} --dataset datasets/cifar-10 ' \
           f'--batchsize {batchsize} --eval_batchsize 256 ' \
           f'--lambda_range {lambda_low} {lambda_high} ' \
           f'--alpha_range {alpha_low} {alpha_high} ' \
           f'{"--alpha_burnin" if alpha_burnin else ""} ' \
           f'--main_lr {lr} --aux_lr 1e-3 --optimizer {opt} ' \
           f'--sample_function {sample_function} ' \
           '--eval_points 7 ' \
           '--downstream_model experiments/cifar_10_normal_training/sophisticated-poetic-warthog-of-variation/model.hdf5 ' \
           '--downstream_model_weights experiments/cifar_10_normal_training/sophisticated-poetic-warthog-of-variation/final_weights.hdf5 ' \
           f'--last_frozen_layer {last_frozen_layer} --burnin_epochs {burnin_epochs} ' \
           '--checkpoint_period 10 --train_summary_period 1 --val_summary_period 10 ' \
           f'{downstream_loss_type} {loss_type_specific} ' \
           'experiments/compressor/cifar10-yoto-cat-crossent-burnin'

    args = shlex.split(args)
    subprocess.run(args)
