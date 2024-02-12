#!/bin/bash

#SBATCH --job-name=terabyte_train
#SBATCH -A bcev-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
<<<<<<< HEAD
#SBATCH --cpus-per-task=16
=======
#SBATCH --cpus-per-task=64
>>>>>>> e1f2fe7c869780c43a33bb98881b8a151cf1305d
#SBATCH --mem=240g
#SBATCH -t 12:00:00
#SBATCH --output=delta_kaggle_%j.log


<<<<<<< HEAD
module load anaconda3_gpu
=======
module purge # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
>>>>>>> e1f2fe7c869780c43a33bb98881b8a151cf1305d

module load anaconda3_gpu
module list  # job documentation and metadata
echo "job is starting on `hostname`"
cd /u/haofeng1/dlrm_comp/
source ~/.bashrc
<<<<<<< HEAD
# conda activate dlrm
=======
>>>>>>> e1f2fe7c869780c43a33bb98881b8a151cf1305d

# set environment varibales

export MASTER_PORT=27149
<<<<<<< HEAD
export WORLD_SIZE=4
=======
>>>>>>> e1f2fe7c869780c43a33bb98881b8a151cf1305d
export DLRM_ALLTOALL_IMPL="alltoall"
echo "WORLD_SIZE="$WORLD_SIZE
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_option

<<<<<<< HEAD
dlrm_pt_bin="python dlrm_s_pytorch.py"
=======
dlrm_pt_bin="python ext_dist_test.py"
>>>>>>> e1f2fe7c869780c43a33bb98881b8a151cf1305d
raw_data="/projects/bcev/haofeng1/Kaggle/raw/train.txt"
processed_data="/projects/bcev/haofeng1/Kaggle/processed/kaggleAdDisplayChallenge_processed.npz"
echo "run pytorch ..."

$dlrm_pt_bin --arch-sparse-feature-size=32 --arch-mlp-bot="13-512-256-64-32" --arch-mlp-top="512-256-1" \
--data-generation=dataset \
--data-set=kaggle \
--processed-data-file=$processed_data \
--raw-data-file=$raw_data \
--loss-function=bce \
--round-targets=True \
--learning-rate=0.1 \
--nepochs=1 \
--mini-batch-size=128 \
--print-freq=1024 \
--print-time \
--test-freq=1024 \
--test-mini-batch-size=16384 \
--test-num-workers=16 \
--use-gpu \
<<<<<<< HEAD
--enable-compress \

$dlrm_extra_option 2>&1 | tee run_terabyte_pt.log
=======
>>>>>>> e1f2fe7c869780c43a33bb98881b8a151cf1305d

echo "done"
