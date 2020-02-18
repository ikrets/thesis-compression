import subprocess
import numpy as np
import shlex

while True:
    num_filters = np.random.choice([64, 128])
    depth = np.random.choice([3, 4])
    film_depth = np.random.choice([2, 3, 4])
    film_width = np.random.choice([128, 196])
    batchsize = np.random.choice([16, 32, 64])
    lambda_low = np.random.choice([0.00005, 0.0001, 0.002, 0.005])
    lambda_high = np.random.choice([0.0025, 0.005, 0.01, 0.1])
    alpha_low = np.random.choice([0.01, 0.1, 0.3, 0.5])
    alpha_high = np.random.choice([0.29, 0.45, 0.7, 0.9, 0.95])

    if alpha_low >= alpha_high or lambda_low >= lambda_high:
        continue

    lr = 1e-4
    opt = 'adam'

    args = f'python train_film_compressor_comparison.py ' \
           f'--num_filters {num_filters} --depth {depth} --num_postproc 0 --film_depth {film_depth} ' \
           f'--film_width {film_width} --film_activation relu --epochs 101 --dataset datasets/cifar-10 ' \
           f'--batchsize {batchsize} --eval_batchsize 128 ' \
           f'--lambda_range {lambda_low} {lambda_high} ' \
           f'--alpha_range {alpha_low} {alpha_high} ' \
           f'--main_lr {lr} --aux_lr 1e-3 --optimizer {opt} ' \
           '--eval_points 5 ' \
           '--downstream_model experiments/cifar_10_normal_training/sophisticated-poetic-warthog-of-variation/model.hdf5 ' \
           '--downstream_model_weights experiments/cifar_10_normal_training/sophisticated-poetic-warthog-of-variation/final_weights.hdf5 ' \
           '--downstream_loss_layer activation_5 --checkpoint_period 10 --train_summary_period 1 --val_summary_period 10 ' \
           'experiments/compressor/cifar10-yoto-alpha-lambda'

    args = shlex.split(args)
    subprocess.run(args)

