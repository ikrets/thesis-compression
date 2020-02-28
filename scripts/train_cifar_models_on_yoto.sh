#!/bin/bash

export PYTHONPATH=.
for dataset in $(ls -d $1/*/*/); do
  python train_on_cifar10.py --dataset $dataset --experiment_dir experiments/cifar_10_yoto_c2x --base_lr 1e-1 --base_wd 5e-4 --batch_size 128
done