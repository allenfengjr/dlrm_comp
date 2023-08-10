import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# quantization
def float16ToBinary(f):
    i16 = np.float16(f).view('H')
    return format(i16, '016b')

def save_image(bitmaps, fileName, index):
    plt.imshow(bitmaps[index], cmap="gray")
    figure_name = fileName[31:-4] + "_bit_" + str(index)
    plt.show()
    plt.savefig(figure_name, dpi=300)

def trans2heatmap(fileName, M, N):
    with open(fileName, mode='rb') as file:
        numpy_data = np.fromfile(file, dtype=np.float16)
    
    numpy_data = numpy_data.reshape((M, N))
    
    bitmaps = []
    for bit_index in range(16):
        bit_map = np.zeros((M, N), dtype=int)
        for i in range(M):
            for j in range(N):
                binary_str = float16ToBinary(numpy_data[i, j])
                bit_map[i, j] = int(binary_str[bit_index])
        bitmaps.append(bit_map)
    
    for i in range(16):
        save_image(bitmaps, fileName, i)

M, N = 128, 16

# Uncomment the following lines to process all .f32 or .dat files in the directory
# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     if os.path.isfile(f) and (f.endswith(".f32") or f.endswith(".dat")):
#         print(f)
#         trans2heatmap(f, M, N)

device, table_order, batch = 1, 1, 49
filename = f"embedding_output_vector_{device}_{table_order}_batch_{batch}.bin"
filepath = os.path.join("/N/scratch/haofeng/TB_emb", filename)
trans2heatmap(filepath, M, N)
