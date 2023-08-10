import os
import numpy as np
import matplotlib.pyplot as plt

def read_numpy_file(root_dir, compressor, r_tolerance, iter, table):
    filename = f"sampleDelta_compressor_{compressor}_eb_{r_tolerance}_iter_{iter}_table_{table}.npy"
    filepath = os.path.join(root_dir, filename)
    data = np.load(filepath)
    return data

def read_bin_file(root_dir, device, table_order, batch):
    filename = f"embedding_output_vector_{device}_{table_order}_batch_{batch}.bin"
    filepath = os.path.join(root_dir, filename)
    data = np.fromfile(filepath, dtype = np.float16)
    return data

def calculate_l2_norm(data):
    norm = np.linalg.norm(data)
    return norm

def draw_histogram(data, filename):
    l2_norm = calculate_l2_norm(data)
    plt.hist(data.flatten(), bins=50)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(filename)
    if l2_norm is not None:
        plt.text(0.7, 0.9, f"L2 norm: {l2_norm:.4f}", transform=plt.gca().transAxes)    
    plt.savefig(filename)
    plt.close()
# Example usage
# data = read_numpy_file("/N/scratch/haofeng/TB_emb", "ZFP_compressor", 0.7, 211968, 5)
for device in range(8):
    for table_order in range(3):
        for batch in range(50):
            data = read_bin_file("/N/scratch/haofeng/TB_emb", device, table_order, batch)
            norm = calculate_l2_norm(data)
            print(f"L2 norm of data: {norm}")
            pic_name = f"./histogram/emb_device_{device}_table_{table_order}_batch_{batch}.png"
            draw_histogram(data, pic_name)
