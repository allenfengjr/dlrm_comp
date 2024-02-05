#!/bin/bash

#SBATCH --job-name=terabyte_train
#SBATCH -A bcev-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=240g
#SBATCH -t 12:00:00
#SBATCH --output=delta_tb_%j.log


module purge # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)

module load anaconda3_gpu
module list  # job documentation and metadata
echo "job is starting on `hostname`"
cd /u/haofeng1/dlrm_comp/
source ~/.bashrc

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
raw_data="/projects/bcev/haofeng1/Kaggle/raw/train.txt"
processed_data="/projects/bcev/haofeng1/Kaggle/processed/kaggleAdDisplayChallenge_processed.npz"
echo "run pytorch ..."

mpirun -np $WORLD_SIZE $dlrm_pt_bin

$dlrm_extra_option 2>&1 | tee run_terabyte_pt.log

echo "done"
