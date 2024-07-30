#!/bin/bash

# conda activate dlrm
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_option
export DUMP_PATH="/root/dlrm_comp/SC_ADAE_EMB"


dlrm_pt_bin="python /root/dlrm_comp/dlrm_s_with_compress_quan.py"

# RAW DATA PATH
# example path
# raw_data="./10M_processed/day"
# processed_data="./10M_processed/terabyte_processed.npz"

# NOTE: Please put Criteo Terabyte dataset(day_0, day_1,...,day_23) under a directory.

raw_data="path_to_dataset/day"
processed_data="path_to_dataset/terabyte_processed.npz"

# set compression variables
export SZ_PATH="/u/haofeng1/SZ3/lib64/"
export TIGHTEN_EB_TABLES="0 9 10 19 20 21 22"
export LOOSEN_EB_TABLES="5 8 12 15 16 17 18 24 25"
# Custom error bound for the tables defined above
export TIGHTEN_EB_VALUE="0.01"
export LOOSEN_EB_VALUE="0.05"
# Base error bound for all other tables
export BASE_ERROR_BOUND="0.03"

export EB_CONSTANT=2

# Early Stage: terabytes 65536
export EARLY_STAGE=65536
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
--use-gpu | tee output.log

$dlrm_extra_option 2>&1 | tee run_terabyte_pt.log

echo "done"
