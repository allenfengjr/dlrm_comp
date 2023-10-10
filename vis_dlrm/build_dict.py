import numpy as np
import pickle

def quantization(original_arr, eb):
    eb = (original_arr.max() - original_arr.min()) * eb
    print("absolute error bound, ", eb)
    quantization_arr = np.round(original_arr* (1/(eb*2))).astype(np.int32)
    quantization_arr += abs(quantization_arr.min())+1
    synthetic_outlier = quantization_arr[np.where(quantization_arr>=65536)].astype(np.float32)
    return quantization_arr


def binary_to_array(filename, dimension=16):
    # Read the binary data
    data = np.fromfile(filename, dtype=np.float32)  # Change dtype if your data type is different
    
    # Make sure the data can be reshaped to the desired shape
    if data.size % dimension != 0:
        raise ValueError(f"Data cannot be reshaped into (-1, {dimension})")

    # Reshape the data
    reshaped_data = data.reshape(-1, dimension)
    
    return reshaped_data

def array_to_dict_and_keys(arr):
    # Use a dictionary to map unique rows (converted to tuples) to unique keys
    # and also maintain a list of these keys in their original order
    unique_rows_dict = {}
    keys = []
    next_key = 0

    for row in arr:
        row_tuple = tuple(row)
        if row_tuple not in unique_rows_dict:
            unique_rows_dict[row_tuple] = next_key
            next_key += 1
        keys.append(unique_rows_dict[row_tuple])

    return unique_rows_dict, keys

def save_to_file(data, filename):
    if isinstance(data, dict):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    else:  # If it's the list of keys
        np.array(data).tofile(filename)

table = 2
epoch = 10
iter = 0
sparse_len = 16
filename = f"/home/haofeng/TB_emb_8m/embedding_output_vector_table_{table}_epoch_{epoch}_iter_{iter}.bin"
array_data = binary_to_array(filename, sparse_len)

# add quantization
array_data = quantization(array_data, 0.002)
unique_dict, keys_list = array_to_dict_and_keys(array_data)

print(f"Table is, {table}")
print(f" the size of dict is, {len(unique_dict.keys())}")
print(f" the size of list is, {len(keys_list)}")

# Save dictionary and list to separate files
save_to_file(unique_dict, f"/home/haofeng/TB_emb_8m/Dict_table_{table}_epoch_{epoch}_iter_{iter}.pkl")
save_to_file(keys_list, f"/home/haofeng/TB_emb_8m/keyList_table_{table}_epoch_{epoch}_iter_{iter}.bin")
