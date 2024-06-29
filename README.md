# Part 1: Lossy Compression in Training

## Step 1. Train DLRM with adaptive compression
Running scripts are in `SC_ADAE_script`. Run `bash kaggle_run.sh` and `bash tb_run.sh` to train DLRM with adaptive compression on two dataset respectively.

Here are some parameters you can tune for training.

```bash
# set tighten/loosen embedding table id
export TIGHTEN_EB_TABLES="0 9 10 19 20 21 22"
export LOOSEN_EB_TABLES="5 8 12 15 16 17 18 24 25"
# set tighten/loosen embedding table error bound
export TIGHTEN_EB_VALUE="0.01"
export LOOSEN_EB_VALUE="0.05"
# set base error bound for all other tables
export BASE_ERROR_BOUND="0.03"
# set decay magnitude
export EB_CONSTANT=2

# set decay stage end iteration
export EARLY_STAGE=65536
# set decay function
export DECAY_FUNC="step"
```

## Step 2. Parse log and 
To parse the output logs and generate the figure. Run `python accuracy_parser.py` to generate the parsed accuracy logs. Run `python accuracy_curve.py` to generate accuracy curve figures.

# Part 2: Compression Benchmark

**Input: Dumped EMB data** \
**Output: Compression Ratio, De/Compression Throughput**

## Step 1. Do Quantization

To simulate lossy compression, please first apply quantization. Run `python quantization.py EMB_file_path decay_stage`, to generate quantization code of inputs. Modify `EMB_file_path` as the embedding data directory. Modify `decay_stage` as the decay stop point. Function `build_error_bound()` define the build strategy of table-wise and iteration-wise adjustment, `step-wise decay` by default.

Change `intType` variable to choose use `int8` or `int16` as quantization code datatype.

## Step 1. Choose Lossless Encoder

Run `python lossless_encoder.py` to generate compression ratio, compression throughput and decompression throughput. For more details about executable binary usage, please refer to documents of above repos and `EXECUTABLES` variable in `lossless_encoder.py`.

To install and use LZ4 and ANS encoder, refer to [nvcomp](https://developer.nvidia.com/nvcomp).

Usage example `benchmark_ans_chunked -f {filename}`.

To install vector-based LZ encoder, refer to [gpulz](https://github.com/hipdac-lab/ICS23-GPULZ). Please switch to `vector_matching` branch to use the proper vector-based LZ encoder.

Usage example `gpulz -i {fliename}`.

To install Huffman encoder, refer to [cusz](https://github.com/szcompressor/cuSZ/). After compiling, the execution binary `bin_hf` is under `example` folder.

Usage example `bin_hf {filename} x y z booklen`. 

You can also use huffman_wopred to compress raw embedding data. Run `sz3 -f -i inputFile -z outputFile -Dimension d_x d_y d_z -M errorBoundMode errorBound`.


## Step 3. Parse Log and draw

To extract the log file, run `python huffman_parser.py`, `python nvcomp_parser.py`, and `python gpulz_parser.py` for different lossless encoders' log. These script will read log file as input and print compression ratio, compression throughput, and decompression throughput. Modify `file_path` as input log path.

