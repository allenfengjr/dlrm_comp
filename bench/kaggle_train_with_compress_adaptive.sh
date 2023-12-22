#!/bin/bash

#SBATCH --job-name=kaggle
#SBATCH -A r00114
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --output=adaptive_%j.log 
#SBATCH --mem=200G
echo "Baseline"
module load nvidia
export LD_LIBRARY_PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/comm_libs/nccl/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/comm_libs/openmpi4/openmpi-4.0.5/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/N/u/haofeng/BigRed200/SZ3_build/lib64/:$LD_LIBRARY_PATH
export PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/comm_libs/openmpi4/openmpi-4.0.5/bin/:$PATH
module load cudatoolkit
#echo $LD_LIBRARY_PATH
cd /N/u/haofeng/BigRed200/dlrm
source ~/.bashrc
conda activate dlrm

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#WARNING: must have compiled PyTorch and caffe2

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_option
export MASTER_PORT=27149
export WORLD_SIZE=1
export DLRM_ALLTOALL_IMPL="alltoall"
echo "WORLD_SIZE="$WORLD_SIZE
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# Define error bound
# Indices of the tables with custom error bounds
export TIGHTEN_EB_TABLES="2 3 9 11 15 20 23 25"
export LOOSEN_EB_TABLES="8 16 19 21 22 24"
# Custom error bound for the tables defined above
export TIGHTEN_EB_VALUE="0.005"
export LOOSEN_EB_VALUE="0.03"
# Base error bound for all other tables
export BASE_ERROR_BOUND="0.02"

# Early Stage: 1024 * 64(total 306969 mini-batch as 128 batch size)
export EARLY_STAGE=65536

# Compress/Uncompress every 4096 mini-batch
export CYCLE_LEN_COMP=4096
export CYCLE_LEN_NO_COMP=4096
export DECAY_FUNC="log"

dlrm_pt_bin="python dlrm_s_with_compress_adaptive.py"
dlrm_c2_bin="python dlrm_s_caffe2.py"
raw_data="/N/scratch/haofeng/Kaggle/raw/train.txt"
processed_data="/N/scratch/haofeng/Kaggle/processed/kaggleAdDisplayChallenge_processed.npz"
echo "run pytorch ..."
# WARNING: the following parameters will be set based on the data set
# --arch-embedding-size=... (sparse feature sizes)
# --arch-mlp-bot=... (the input to the first layer of bottom mlp)
mpirun $WORLD_SIZE $dlrm_pt_bin --arch-sparse-feature-size=32 --arch-mlp-bot="13-512-256-64-32" --arch-mlp-top="512-256-1" \
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
--enable-compress

#$dlrm_extra_option 2>&1 | tee run_terabyte_pt.log

echo "done"
