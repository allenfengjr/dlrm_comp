#!/bin/bash

#SBATCH --job-name=Terebytes
#SBATCH -A r00114
#SBATCH -p gpu-debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH --output=TB_10M_comp_%j.log 
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

export MASTER_PORT=27149
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

dlrm_pt_bin="python dlrm_s_with_compress_quan.py"
raw_data="/N/slate/haofeng/10M_processed/day"
processed_data="/N/slate/haofeng/10M_processed/terabyte_processed.npz"

# set compression variables
export SZ_PATH="/u/haofeng1/SZ3/lib64/"
export TIGHTEN_EB_TABLES="2 3 9 11 15 20 23 25"
export LOOSEN_EB_TABLES="8 16 19 21 22 24"
# Custom error bound for the tables defined above
export TIGHTEN_EB_VALUE="0.01"
export LOOSEN_EB_VALUE="0.05"
# Base error bound for all other tables
export BASE_ERROR_BOUND="0.03"

export EB_CONSTANT=2

# Early Stage: terabytes 65536*16
export EARLY_STAGE=1048576
echo "ALL STEP CASE"
# Compress/Uncompress every 4096 mini-batch
export CYCLE_LEN_COMP=4096
export CYCLE_LEN_NO_COMP=4096
export DECAY_FUNC="step"

echo "run pytorch ..."

$dlrm_pt_bin --arch-sparse-feature-size=64 --arch-mlp-bot="13-512-256-64" --arch-mlp-top="512-512-256-1" \
--max-ind-range=10000000 \
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
--data-sub-sample-rate=0.875 \
--use-gpu \

$dlrm_extra_option 2>&1 | tee run_terabyte_pt.log

echo "done"
