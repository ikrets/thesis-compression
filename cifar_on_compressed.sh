EXPERIMENT_DIR=/home/ilya/thesis-compression/experiments/cifar_10_normal_training
BASE_LR=1e-1
BASE_WD=5e-4
BATCH_SIZE=128

python train_on_cifar10.py --dataset /home/ilya/thesis-compression/datasets/cifar-10-bpg/qp40_cfmt444 \
  --experiment_dir $EXPERIMENT_DIR --base_lr $BASE_LR --base_wd $BASE_WD --batch_size $BATCH_SIZE

python train_on_cifar10.py --dataset /home/ilya/thesis-compression/datasets/cifar-10-bpg/qp35_cfmt444 \
  --experiment_dir $EXPERIMENT_DIR --base_lr $BASE_LR --base_wd $BASE_WD --batch_size $BATCH_SIZE

python train_on_cifar10.py --dataset /home/ilya/thesis-compression/datasets/cifar-10-bpg/qp30_cfmt444 \
  --experiment_dir $EXPERIMENT_DIR --base_lr $BASE_LR --base_wd $BASE_WD --batch_size $BATCH_SIZE
