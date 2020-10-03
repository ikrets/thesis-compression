#!/bin/bash

EXPERIMENT_DIR=$HOME/thesis-compression-s3/experiments
BPG_DATASETS=$HOME/thesis-compression-data/datasets/cifar-10-bpg
EVALUATION_DIR=$HOME/thesis-compression-s3/evaluation

export TF_CPP_MIN_LOG_LEVEL=2

# cifar-vgg16, BPG compression
for bpg_c_model in $EXPERIMENT_DIR/cifar10_vgg16/bpg/*/; do
  output_dir="$EVALUATION_DIR/cifar10_vgg16/bpg/$(basename $bpg_c_model)"
  echo "Processing $output_dir."

  if [ -f $output_dir/results.csv ]; then
    echo "Skipping: results.csv exists."
    continue
  fi

  python evaluator.py --batch_size 2048 \
    --uncompressed_dataset data/datasets/cifar-10 \
    --downstream_O_model data/trained_models/cifar-10/vgg16/uncompressed/final_model.hdf5 \
    --compressed_dataset_1 "$BPG_DATASETS/$(basename $bpg_c_model)" \
    --compressed_dataset_1_type files \
    --compressed_dataset_2 "$BPG_DATASETS/$(basename $bpg_c_model)" \
    --compressed_dataset_2_type files \
    --downstream_C_model $bpg_c_model/final_model.hdf5 \
    --output_dir $output_dir
done

# cifar-vgg16, learned compressor
for learned_c_model in $EXPERIMENT_DIR/cifar10_vgg16/*/*/; do
  if [ "$(basename "$(dirname "$learned_c_model")")" == "bpg" ]; then
    continue
  fi

  output_dir="$EVALUATION_DIR/cifar10_vgg16/$(basename "$(dirname "$learned_c_model")")/$(basename "$learned_c_model")"
  echo "Processing $output_dir."

  if [ -f "$output_dir/results.csv" ]; then
    echo "Skipping: results.csv exists."
    continue
  fi

  python evaluator.py --batch_size 2048 \
    --uncompressed_dataset data/datasets/cifar-10 \
    --downstream_O_model data/trained_models/cifar-10/vgg16/uncompressed/final_model.hdf5 \
    --compressed_dataset_1 "$learned_c_model/compressed_cifar10/compressed.tfrecord" \
    --compressed_dataset_1_type tfrecords \
    --compressed_dataset_2 "$learned_c_model/compressed_cifar10/compressed.tfrecord" \
    --compressed_dataset_2_type tfrecords \
    --downstream_C_model "$learned_c_model/C_model/final_model.hdf5" \
    --output_dir "$output_dir"
done

# resnet18, BPG
for learned_c_model in $EXPERIMENT_DIR/cifar10_resnet18/bpg/*/; do
  output_dir="$EVALUATION_DIR/cifar10_resnet18/bpg/$(basename $learned_c_model)"
  echo "Processing $output_dir."

  if [ -f $output_dir/results.csv ]; then
    echo "Skipping: results.csv exists."
    continue
  fi

  if [ -f $learned_c_model/c_model_bgr ]; then
    maybe_correct_c_model="--downstream_C_model_correct_bgr"
  else
    maybe_correct_c_model=""
  fi

  python evaluator.py --batch_size 2048 \
    --uncompressed_dataset data/datasets/cifar-10 \
    --downstream_O_model data/trained_models/cifar-10/resnet18/uncompressed_bgr/model.hdf5 \
    --downstream_O_model_weights data/trained_models/cifar-10/resnet18/uncompressed_bgr/final_weights.hdf5 \
    --downstream_O_model_correct_bgr \
    $maybe_correct_c_model \
    --compressed_dataset_1 "$BPG_DATASETS/$(basename $learned_c_model)" \
    --compressed_dataset_1_type files \
    --compressed_dataset_2 "$BPG_DATASETS/$(basename $learned_c_model)" \
    --compressed_dataset_2_type files \
    --downstream_C_model $learned_c_model/final_model.hdf5 \
    --output_dir $output_dir
done

# resnet18, learned compression
for learned_c_model in $EXPERIMENT_DIR/cifar10_resnet18/*/*/; do
  if [ "$(basename "$(dirname "$learned_c_model")")" == "bpg" ]; then
    continue
  fi

  output_dir="$EVALUATION_DIR/cifar10_resnet18/$(basename $(dirname $learned_c_model))/$(basename $learned_c_model)"
  echo "Processing $output_dir."

  if [[ $(basename $learned_c_model) =~ "epoch_check" ]]; then
    echo "Skipping: this is a probe."
    continue
  fi

  if [ -f $output_dir/results.csv ]; then
    echo "Skipping: results.csv exists."
    continue
  fi

  python evaluator.py --batch_size 2048 \
    --uncompressed_dataset data/datasets/cifar-10 \
    --downstream_O_model data/trained_models/cifar-10/resnet18/uncompressed_bgr/model.hdf5 \
    --downstream_O_model_weights data/trained_models/cifar-10/resnet18/uncompressed_bgr/final_weights.hdf5 \
    --downstream_O_model_correct_bgr \
    --compressed_dataset_1 $learned_c_model/compressed_cifar10/compressed.tfrecord \
    --compressed_dataset_1_type tfrecords \
    --compressed_dataset_2 $learned_c_model/compressed_cifar10/compressed.tfrecord \
    --compressed_dataset_2_type tfrecords \
    --downstream_C_model $learned_c_model/C_model/final_model.hdf5 \
    --output_dir $output_dir
done

# cross-architecture O2C C2O vgg16 to resnet18
for vgg_c in $EXPERIMENT_DIR/cifar10_vgg16/*/*/; do
  if [ "$(basename "$(dirname "$vgg_c")")" == "bpg" ]; then
    continue
  fi

  vgg_loss=$(basename "$(dirname "$vgg_c")")
  vgg_loss_param=$(basename "$vgg_c")
  output_dir="$EVALUATION_DIR/cifar10_Oresnet18_Cvgg16/$vgg_loss/$vgg_loss_param"
  echo "Processing $output_dir."

  if [ -f "$output_dir/results.csv" ]; then
    echo "Skipping: results.csv exists."
    continue
  fi

  python evaluator.py --batch_size 2048 \
    --uncompressed_dataset data/datasets/cifar-10 \
    --downstream_O_model data/trained_models/cifar-10/resnet18/uncompressed_bgr/model.hdf5 \
    --downstream_O_model_weights data/trained_models/cifar-10/resnet18/uncompressed_bgr/final_weights.hdf5 \
    --downstream_O_model_correct_bgr \
    --compressed_dataset_1 "$vgg_c/compressed_cifar10/compressed.tfrecord" \
    --compressed_dataset_1_type tfrecords \
    --compressed_dataset_2 "$vgg_c/compressed_cifar10/compressed.tfrecord" \
    --compressed_dataset_2_type tfrecords \
    --skip_C2C \
    --downstream_C_model "$vgg_c/C_model/final_model.hdf5" \
    --output_dir "$output_dir"
done

# cross-architecture O2C C2O resnet18 to vgg16
for resnet_c in $EXPERIMENT_DIR/cifar10_resnet18/*/*/; do
  if [ "$(basename "$(dirname "$resnet_c")")" == "bpg" ]; then
    continue
  fi

  resnet_loss=$(basename "$(dirname "$resnet_c")")
  resnet_loss_param=$(basename "$resnet_c")
  output_dir="$EVALUATION_DIR/cifar10_Ovgg16_Cresnet18/$resnet_loss/$resnet_loss_param"
  echo "Processing $output_dir."

  if [[ "$resnet_loss_param" =~ "epoch_check" ]]; then
    echo "Skipping: this is a probe."
    continue
  fi

  if [ -f "$output_dir/results.csv" ]; then
    echo "Skipping: results.csv exists."
    continue
  fi

  python evaluator.py --batch_size 2048 \
    --uncompressed_dataset data/datasets/cifar-10 \
    --downstream_O_model data/trained_models/cifar-10/vgg16/uncompressed/final_model.hdf5 \
    --compressed_dataset_1 "$resnet_c/compressed_cifar10/compressed.tfrecord" \
    --compressed_dataset_1_type tfrecords \
    --compressed_dataset_2 "$resnet_c/compressed_cifar10/compressed.tfrecord" \
    --compressed_dataset_2_type tfrecords \
    --skip_C2C \
    --downstream_C_model "$resnet_c/C_model/final_model.hdf5" \
    --output_dir "$output_dir"
done
