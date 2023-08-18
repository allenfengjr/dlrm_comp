import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

dtype = np.dtype('B')

def decimalToBinary(n):
    bnr = bin(n).replace('0b','')
    x = bnr[::-1] #this reverses an array
    while len(x) < 8:
        x += '0'
    bnr = x[::-1]
    return bnr

def save_image(df, fileName, index):
    plt.imshow(df, cmap ="gray")
    # plt.colorbar()
    figure_name = fileName[31:-4] + "_binary_" + str(index)
    plt.savefig(figure_name, dpi=300)

def quantization(original_np_array, r_error_bound):
    a_error_bound = (original_np_array.max() - original_np_array.min()) * r_error_bound
    quantized_array = np.empty(original_np_array.shape)
    for idx, val in np.ndenumerate(original_np_array):
        quantized_array[ind] = round(val/2*a_error_bound)
        print(val/2*a_error_bound)
    return quantized_array.astype(np.int16) # maybe int32?

def trans2heatmap(fileName, n):
    # fileName = "/data/lab/tao/boyuan/1800x3600/SOLIN_1_1800_3600.f32"
    # n = 1800 * 3600
    with open(fileName, mode='rb') as file:
        numpy_data = np.fromfile(file,dtype)
    
    cnt = 0
    data_map = [[] for i in range(128)]
    for i in range(128):
        for j in range(16):
            for k in range(2):
                tmp = decimalToBinary(int(numpy_data[cnt]))
                for char in tmp:
                    data_map[i].append(int(char))
                cnt += 1
    dmp = []
    for i in range(16):
        dmp.append([[data_map[j][i*16+k] for k in range(16)] for j in range(128)])
        df = pd.DataFrame(dmp[i])
        save_image(df, fileName, i)
    

directory = "/data/lab/tao/boyuan/1800x3600/"
n = 128 * 16
 
# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     # checking if it is a file
#     if os.path.isfile(f):
#         if f.endswith(".f32") or f.endswith(".dat"):
#             print(f)
#             trans2heatmap(f, n)
device, table_order, batch = 1, 1, 49
filename = f"embedding_output_vector_{device}_{table_order}_batch_{batch}.bin"
filepath = os.path.join("/N/scratch/haofeng/TB_emb", filename)
trans2heatmap(filepath, n)
# trans2heatmap("/data/lab/tao/boyuan/1800x3600/FLNTC_1_1800_3600.f32", n)