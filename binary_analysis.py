'''
Usage:
Example python binary_analysis.py --filepath [Binary File Path] \
--savepath [Saved Image Path] \
--table_idx [The table index of embedding vectors] \
--iteration [The iteration of embedding vectors] \
--error_bound [The relative error bound] \
--shape [(batch_size, emb_len)]
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filepath", type=str, help="the path of binary file")
parser.add_argument("--savepath", type=str, help="the path of saved figures")
parser.add_argument("--table_idx", type=int, help="the index of embedding table")
parser.add_argument("--iteration", type=int, help="the number of iteration")
parser.add_argument("--error_bound", type=float, help="the error bound for pre-quantization and compression")
parser.add_argument("--shape", type=int, nargs=2, help="the shape of the embedding vectors, for example (batch_size, emb_len)")
parser.add_argument("--print_frequency", action="store_true", default=False, help="Print top20 values after quantization")

# Read the binary file as a numpy array
def read_binary_file(filename, data_type):
    return np.fromfile(filename, dtype=data_type)

def int8_to_bits(value):
    """Convert an int8 to its 8-bit representation."""
    return [(value >> i) & 1 for i in range(7, -1, -1)]

def draw_bitmaps(data, table, iter):
    """
    Draw 16 figures from a (128, 16) numpy array.
    Each figure is of shape (128, 8) representing the bit values.
    """
    rows, cols = data.shape
    
    for col in range(cols):
        bitmaps = np.zeros((rows, 8), dtype=int)
        for row in range(rows):
            bitmaps[row, :] = int8_to_bits(data[row, col])

        plt.figure()
        plt.imshow(bitmaps, cmap='gray', aspect='auto',interpolation='none')
        plt.colorbar()
        plt.title(f"Bit Representation for Column {col} table: {table} iter: {iter}")
        plt.xlabel("Bit Index")
        plt.ylabel("Row")
        plt.show()
        plt.savefig(f"bit_table_{table}_iter_{iter}_{col}.png")
        plt.cla()

def quantization(original_arr, eb):
    eb = (original_arr.max() - original_arr.min()) * eb
    quantization_arr = np.round(original_arr* (1/(eb*2))).astype(np.int8)
    #quantization_arr.tofile(filename)
    return quantization_arr

# Draw histogram
def draw_histogram(data, table, iter):
    plt.hist(data, bins=128, range=(-64,63), facecolor='blue', alpha=0.7)
    plt.title(f"Histogram_table_{table}_iter_{iter}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    plt.savefig(f"hist_table_{table}_iter_{iter}.png")
    plt.cla()

def draw_heatmap(data, table, iter):
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Heatmap table_{table}_iter_{iter}")
    plt.show()
    plt.savefig(f"heat_table_{table}_iter_{iter}.png")
    plt.cla()

def draw_values_with_grid(data, table, iter):
    """
    Draw the values of the data array in a grid.
    """
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    
    # Display grid
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)

    # Remove the major tick labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Display data values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, str(data[i, j]), ha='center', va='center', color='black')

    plt.savefig(f"values_table_{table}_iter_{iter}.png")
    plt.cla()

# Print the top n most frequent integers without using Counter
def print_top_frequent(data, n=20):
    value_counts = {}
    for val in data:
        if val in value_counts:
            value_counts[val] += 1
        else:
            value_counts[val] = 1
            
    most_common = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:n]
    
    print(f"Top {n} most frequent integers:")
    for val, freq in most_common:
        print(f"Value: {val}, Frequency: {freq}")

if __name__ == "__main__":
    args = parser.parse_args()
    filepath = args.filepath
    savepath = args.savepath
    table = args.table_idx
    iter = args.iteration
    batch_size, emb_len = args.shape
    eb = args.error_bound
    print(batch_size)
    data = np.fromfile(filename, dtype=np.float32)
    data = quantization(data, eb)
    draw_histogram(data, table, iter)
    subset = data[:128*emb_len]  # Select the first 128 rows
    subset = subset.reshape(128,emb_len)
    draw_bitmaps(subset, table, iter)
    draw_heatmap(subset, table, iter)
    subset = data[:8*emb_len]
    subset = subset.reshape(8,emb_len)
    draw_values_with_grid(subset, table, iter)
    if args.print_freq:
        print_top_frequent(data, 20)

