#!/bin/bash

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi

export DUMP_PATH="/root/dlrm_comp/SC_ADAE_EMB"


export TIGHTEN_EB_TABLES="2 3 9 11 15 20 23 25"
export LOOSEN_EB_TABLES="8 16 19 21 22 24"
# Custom error bound for the tables defined above
export TIGHTEN_EB_VALUE="0.01"
export LOOSEN_EB_VALUE="0.05"
# Base error bound for all other tables
export BASE_ERROR_BOUND="0.03"

export EB_CONSTANT=2

# Early Stage
export EARLY_STAGE=65536
echo "ALL STEP CASE"
# Compress/Uncompress every 4096 mini-batch
export CYCLE_LEN_COMP=4096
export CYCLE_LEN_NO_COMP=4096
export DECAY_FUNC="step"

dlrm_pt_bin="python /root/dlrm_comp/dlrm_s_with_compress_quan.py"

# RAW DATA PATH
# example path
# raw_data="./Kaggle/raw/train.txt"
# processed_data="./Kaggle/raw/kaggleAdDisplayChallenge_processed.npz"

# NOTE: Please put Criteo Kaggle dataset(including train.txt, test.txt) under `Kaggle/raw/` directory.

raw_data="path_to_dataset/train.txt"
processed_data=""

echo "run pytorch ..."
# WARNING: the following parameters will be set based on the data set
# --arch-embedding-size=... (sparse feature sizes)
# --arch-mlp-bot=... (the input to the first layer of bottom mlp)
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
--enable-compress \
--use-gpu | tee output.log

# Distributed running should use mpirun/torchrun command
# mpirun -np $WORLD_SIZE $dlrm_pt_bin --arch-sparse-feature-size=32 --arch-mlp-bot="13-512-256-64-32" --arch-mlp-top="512-256-1" \
# --data-generation=dataset \
# --data-set=kaggle \
# --processed-data-file=$processed_data \
# --raw-data-file=$raw_data \
# --loss-function=bce \
# --round-targets=True \
# --learning-rate=0.1 \
# --nepochs=1 \
# --mini-batch-size=128 \
# --print-freq=1024 \
# --print-time \
# --test-freq=1024 \
# --test-mini-batch-size=16384 \
# --test-num-workers=16 \
# --use-gpu \

echo "done"
