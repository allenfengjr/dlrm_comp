#!/bin/bash

#SBATCH --job-name=auto
#SBATCH -A r00114
#SBATCH -p gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH --output=all_to_all_comp%j.log 
#SBATCH --mem=200G

module load nvidia
export LD_LIBRARY_PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/comm_libs/nccl/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/comm_libs/openmpi4/openmpi-4.0.5/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/N/soft/sles15/nvidia/21.5/Linux_x86_64/21.5/comm_libs/openmpi4/openmpi-4.0.5/bin/:$PATH
export LD_LIBRARY_PATH=/N/u/haofeng/BigRed200/fz:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/N/u/haofeng/BigRed200/SZ3_build/lib64/:$LD_LIBRARY_PATH

module load cudatoolkit
cd /N/u/haofeng/BigRed200/dlrm
source ~/.bashrc
conda activate new_dlrm

# set environment varibales

export MASTER_PORT=27149
export WORLD_SIZE=2
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

dlrm_pt_bin="python ext_dist_only_communication.py"
raw_data="/N/scratch/haofeng/Kaggle/raw/train.txt"
processed_data="/N/scratch/haofeng/Kaggle/processed/kaggleAdDisplayChallenge_processed.npz"
echo "run pytorch ..."

mpirun -np $WORLD_SIZE $dlrm_pt_bin

$dlrm_extra_option 2>&1 | tee run_terabyte_pt.log

echo "done"
