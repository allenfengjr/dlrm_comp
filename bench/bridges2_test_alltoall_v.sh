#!/bin/bash

#SBATCH --job-name=auto
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH --output=bridge_a2av_%j.log 
#SBATCH --gpus=v100-16:8


module load cuda

cd /jet/home/haofeng1/dlrm_comp
source ~/.bashrc
conda activate dlrm

# set environment varibales

export MASTER_PORT=27149
export WORLD_SIZE=8
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

dlrm_pt_bin="python ext_dist_test.py"
raw_data="/N/scratch/haofeng/Kaggle/raw/train.txt"
processed_data="/N/scratch/haofeng/Kaggle/processed/kaggleAdDisplayChallenge_processed.npz"
echo "run pytorch ..."

mpirun -np $WORLD_SIZE $dlrm_pt_bin

$dlrm_extra_option 2>&1 | tee run_terabyte_pt.log

echo "done"