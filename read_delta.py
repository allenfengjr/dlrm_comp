import os
import numpy as np
import matplotlib.pyplot as plt

def read_numpy_file(root_dir, compressor, r_tolerance, iter, table):
    filename = f"sampleDelta_compressor_{compressor}_eb_{r_tolerance}_iter_{iter}_table_{table}.npy"
    filepath = os.path.join(root_dir, filename)
    data = np.load(filepath)
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
    plt.show()
    plt.savefig(filename)

# Example usage
data = read_numpy_file("/N/scratch/haofeng/embedding_outputs", "ZFP_compressor", 0.7, 211968, 5)
norm = calculate_l2_norm(data)
print(f"L2 norm of data: {norm}")
draw_histogram(data, './ZFP_histogram_0.7.png')
