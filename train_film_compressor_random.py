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

    # downstream_loss_type = np.random.choice(['activation_difference', 'task_performance'])
    downstream_loss_type = 'task_performance'
    last_frozen_layer = 'dense'

    if downstream_loss_type == 'activation_difference':
        downstream_loss_layer = np.random.choice([f'activation_{i}' for i in range(1, 9)])
        loss_type_specific = f'--downstream_loss_layer {downstream_loss_layer}'

        lambda_mult = np.random.uniform(.1, 1)
        alpha_mult = np.random.uniform(.1, 1)

        lambda_low = 7e-5 * lambda_mult
        lambda_high = 0.0025 * lambda_mult
        alpha_low = 0.0003 * alpha_mult
        alpha_high = 0.015 * alpha_mult

    if downstream_loss_type == 'task_performance':
        loss_type_specific = '--model_performance_loss categorical_crossentropy'

        alpha_mult = np.random.uniform(.1, 10)
        alpha_low_high_mult = np.random.uniform(10, 100)

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
           f'--film_width {film_width} --film_activation relu --epochs 51 --dataset datasets/cifar-10 ' \
           f'--batchsize {batchsize} --eval_batchsize 256 ' \
           f'--lambda_range {lambda_low} {lambda_high} ' \
           f'--alpha_range {alpha_low} {alpha_high} ' \
           f'--main_lr {lr} --aux_lr 1e-3 --optimizer {opt} ' \
           f'--sample_function {sample_function} ' \
           '--eval_points 7 ' \
           '--downstream_model experiments/cifar_10_normal_training/sophisticated-poetic-warthog-of-variation/model.hdf5 ' \
           '--downstream_model_weights experiments/cifar_10_normal_training/sophisticated-poetic-warthog-of-variation/final_weights.hdf5 ' \
           f'--last_frozen_layer {last_frozen_layer} ' \
           '--checkpoint_period 10 --train_summary_period 1 --val_summary_period 10 ' \
           f'{downstream_loss_type} {loss_type_specific} ' \
           'experiments/compressor/cifar10-yoto-performance-initial-search'

    args = shlex.split(args)
    subprocess.run(args)
