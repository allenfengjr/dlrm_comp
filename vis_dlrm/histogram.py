import numpy as np
import matplotlib.pyplot as plt

# Read the binary file as a numpy array (uint8)
def read_binary_file(filename):
    return np.fromfile(filename, dtype=np.uint8)

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
    plt.hist(data, bins=64, range=(-32,31), facecolor='blue', alpha=0.7)
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

def main():
    table = 11
    iter = 34
    filename = f"/home/shared/embs_aug15/table{table}_iter_{iter}.bin"
    data = np.fromfile(filename, dtype=np.float32)
    data = quantization(data, 1e-2)
    draw_histogram(data, table, iter)
    subset = data[:128*16]  # Select the first 128 rows
    subset = subset.reshape(128,16)
    draw_bitmaps(subset, table, iter)
    draw_heatmap(subset, table, iter)
    subset = data[:8*16]
    subset = subset.reshape(8,16)
    draw_values_with_grid(subset, table, iter)
    print_top_frequent(data, 20)

if __name__ == "__main__":
    main()