#!/bin/bash

#SBATCH --job-name=auto
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=96:00:00
#SBATCH --output=bigred_%j.log 
#SBATCH --mem=200G

module load nvidia
export LD_LIBRARY_PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/comm_libs/nccl/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/comm_libs/openmpi4/openmpi-4.0.5/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/cuda/lib64:$LD_LIBRARY_PATH
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

dlrm_pt_bin="python dlrm_s_pytorch.py"
dlrm_c2_bin="python dlrm_s_caffe2.py"
raw_data="/N/scratch/haofeng/TB/raw/day"
processed_data="/N/scratch/haofeng/TB/processed/terabyte_processed.npz"
echo "run pytorch ..."
# WARNING: the following parameters will be set based on the data set
# --arch-embedding-size=... (sparse feature sizes)
# --arch-mlp-bot=... (the input to the first layer of bottom mlp)
$dlrm_pt_bin --arch-sparse-feature-size=64 --arch-mlp-bot="13-512-256-64" --arch-mlp-top="512-512-256-1" --max-ind-range=10000000 \
--data-generation=dataset \
--data-set=terabyte \
--processed-data-file=$processed_data \
--raw-data-file=$raw_data
--loss-function=bce \
--round-targets=True \
--learning-rate=0.1 \
--nepochs=1 \
--mini-batch-size=8192 \
--print-freq=8192 \
--print-time \
--test-mini-batch-size=16384 \
--test-num-workers=16 \

#$dlrm_extra_option 2>&1 | tee run_terabyte_pt.log

echo "done"
