#!/bin/bash


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


# Define error bound
# Indices of the tables with custom error bounds
export CUSTOM_EB_TABLES="2 3 9 11 15 20 23 25"
# Custom error bound for the tables defined above
export CUSTOM_EB_VALUE="0.005"
# Base error bound for all other tables
export BASE_ERROR_BOUND="0.05"
# Early Stage: 1024 * 64(total 306969 mini-batch as 128 batch size)
export EARLY_STAGE=65536

# Compress/Uncompress every 4096 mini-batch
export CYCLE_LEN_COMP=4096
export CYCLE_LEN_NO_COMP=4096
export DECAY_FUNC="log"

dlrm_pt_bin="python /home/haofeng/dlrm_comp/dlrm_s_with_compress_adaptive.py"
dlrm_c2_bin="python dlrm_s_caffe2.py"

raw_data="/home/haofeng/datasets/Kaggle/raw/train.txt"
processed_data="/home/haofeng/datasets/Kaggle/raw/kaggleAdDisplayChallenge_processed.npz"
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
--use-gpu \
--enable-compress

#$dlrm_extra_option 2>&1 | tee run_terabyte_pt.log

echo "done"
