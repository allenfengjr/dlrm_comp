#!/bin/bash

#SBATCH --job-name=Terebytes
#SBATCH -A r00114
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --output=TB_debug_multi_%j.log 
#SBATCH --mem=240G

cat "Note for bash: just test Terebyte Dataset"

module load nvidia
export LD_LIBRARY_PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/comm_libs/nccl/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/comm_libs/openmpi4/openmpi-4.0.5/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/N/u/haofeng/BigRed200/SZ3_build/lib64/:$LD_LIBRARY_PATH
export PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/comm_libs/openmpi4/openmpi-4.0.5/bin/:$PATH
module load cudatoolkit
cd /N/u/haofeng/BigRed200/dlrm
source ~/.bashrc
conda activate dlrm

# set environment varibales


if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_option

dlrm_pt_bin="python dlrm_s_pytorch.py"
raw_data="/N/scratch/haofeng/criteo_TB/raw/day"
processed_data="/N/scratch/haofeng/criteo_TB/50M_processed/terabyte_processed.npz"

echo "run pytorch ..."
# WARNING: the following parameters will be set based on the data set
# --arch-embedding-size=... (sparse feature sizes)
# --arch-mlp-bot=... (the input to the first layer of bottom mlp)
# mpirun -np $WORLD_SIZE
$dlrm_pt_bin --arch-sparse-feature-size=64 --arch-mlp-bot="13-512-256-64" --arch-mlp-top="512-512-256-1" \
--max-ind-range=50000000 \
--data-generation=dataset \
--data-set=terabyte \
--processed-data-file=$processed_data \
--raw-data-file=$raw_data \
--loss-function=bce \
--round-targets=True \
--learning-rate=0.1 \
--nepochs=1 \
--mini-batch-size=2048 \
--print-freq=1024 \
--print-time \
--test-freq=1024 \
--test-mini-batch-size=10240 \
--memory-map \
--data-sub-sample-rate=0.0 \
--use-gpu \
--save-model="/N/scratch/haofeng/TB_unlimit_original_model.pt"

#$dlrm_extra_option 2>&1 | tee run_terabyte_pt.log

echo "done"
