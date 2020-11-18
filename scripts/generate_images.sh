#!/bin/bash

S3_PATH=/media/ilya/TOSHIBAEXT/thesis-compression-s3

export PYTHONPATH=.

image_names="n01440764_10380.png n02102040_142.png n02979186_6390.png n03000684_1031.png \
  n03028079_4612.png n03394916_4301.png n03417042_2230.png n03425413_5362.png n03445777_1132.png \
  n03888257_17872.png"

python scripts/images_and_distortions.py \
  --image_names $image_names \
  --original_model $S3_PATH/experiments/imagenette_classifiers/imagenette2_filtered/final_model.hdf5 \
  --original_model_readout activation_15 \
  --original_dataset $S3_PATH/datasets/imagenette2_filtered \
  --bpg_datasets $S3_PATH/datasets/imagenette2_filtered_bpg \
  --compressed_datasets $S3_PATH/experiments/imagenette_r18 \
  --output_dir $S3_PATH/evaluation/images_and_distortions