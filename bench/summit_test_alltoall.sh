#!/bin/bash

#!/bin/bash
#BSUB -nnodes 2
#BSUB -W 01:30
#BSUB -P csc337
#BSUB -alloc_flags "smt4 nvme"
#BSUB -J bert 
#BSUB -o %J.output_grads_sum.txt
#BSUB -e %J.err_grads_sum.txt
#BSUB -q batch

DLRM_ROOT=/ccs/home/dtao/haofeng/dlrm

# Load modules
module load open-ce/1.5.2-py38-0
module load gcc
conda activate bert-pytorch
# Determine number of nodes
nprocspn=6
nnodes=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
nprocs=$(( ${nnodes} * ${nprocspn} ))

echo Number of nodes ${nnodes}
echo nprocspn ${nprocspn}
#Baixi: important for enabling 6 gpus per compute node
unset CUDA_VISIBLE_DEVICES
#CONFIG=/home/sunbaixi/Second_Order/BERT-PyTorch/config/bert_pretraining_phase2_config.json

cd ${DLRM_ROOT}
nodes=($(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch))
head=${nodes[0]}

jsrun --smpiargs="-disable_gpu_hooks" -n ${nnodes} -g 6 -c 42 -a ${nprocspn} -r 1 \
        -E DLRM_ROOT=$DLRM_ROOT \
        -E OMP_NUM_THREADS=8 \
        -E MASTER_PORT=27149 \
        -E WORLD_SIZE=4 \
        -E DLRM_ALLTOALL_IMPL="alltoall" \
        -E MASTER_ADDR=$head \
        --bind=proportional-packed:7 \
        --launch_distribution=packed \
        python ext_dist_only_communication.py

echo "done"
