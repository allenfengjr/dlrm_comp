# Compression Benchmark

**Input: Dumped EMB data**

## Step 1. Do Quantization

To simulate lossy compression, please first apply quantization. Run `quantization.py`, to generate quantization code of inputs. Please modify `EMB_file_path` and filename format to fit requirement.

## Step 1.1(Optional) Data Padding

To simulate compression on larger batch_size, you can run `padding_files.py` to padding EMB data with its own copy.

## Step 2. Choose Lossless Encoder

There are four encoder we use, LZ4 encoder, ANS encoder, GPULZ encoder, and Huffman encoder.

To install and use LZ4 and ANS encoder, refer to [nvcomp](https://developer.nvidia.com/nvcomp).

To install GPULZ encoder, refer to [gpulz](https://github.com/hipdac-lab/ICS23-GPULZ).

To install Huffman encoder, refer to [cusz]

## Step 3. Parse Log

To extract the log file, use `parser.py`, it will 

## Step 4. Visualization

