import numpy as np
import pickle
from tabulate import tabulate

def quantization(original_arr, eb):
    eb = (original_arr.max() - original_arr.min()) * eb
    quantization_arr = np.round(original_arr* (1/(eb*2))).astype(np.int32)
    quantization_arr += abs(quantization_arr.min())+1
    synthetic_outlier = quantization_arr[np.where(quantization_arr>=65536)].astype(np.float32)
    return quantization_arr

def binary_to_array(filename, dimension=16):
    data = np.fromfile(filename, dtype=np.float32)
    if data.size % dimension != 0:
        raise ValueError(f"Data cannot be reshaped into (-1, {dimension})")
    reshaped_data = data.reshape(-1, dimension)
    return reshaped_data

def array_to_dict_and_keys(arr):
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
    else:
        np.array(data).tofile(filename)

# New error bounds
error_bounds = [1e-2, 5e-3, 2e-2]

iter = 23
sparse_len = 32

# Initialize a list to hold results for printing
results = []

# Loop through each table, and each error bound, and perform your operation
for eb in error_bounds:
    results_for_eb = []
    for table in range(26):
        filename = f"/home/haofeng/SC_emb_data/SC_Kaggle_emb/EMB_{table}_iter_{iter}.bin"
        array_data = binary_to_array(filename, sparse_len)

        # Get the size of the set of the original data before quantization
        original_set_size = len(np.unique(array_data, axis=0))
        
        # Add quantization
        array_data_q = quantization(array_data, eb)
        unique_dict, keys_list = array_to_dict_and_keys(array_data_q)

        # Save dictionary and list to separate files
        save_to_file(unique_dict, f"/home/haofeng/SC_emb_data/SC_Kaggle_emb/Dict_table_{table}_iter_{iter}.pkl")
        save_to_file(keys_list, f"/home/haofeng/SC_emb_data/SC_Kaggle_emb/keyList_table_{table}_iter_{iter}.bin")

        # Append the results for printing
        ratio = len(unique_dict.keys()) / original_set_size
        results_for_eb.append([table, eb, original_set_size, len(unique_dict.keys()), len(keys_list), ratio])

    # Rank tables based on the ratio (dict size/ original dict size)
    results_for_eb.sort(key=lambda x: x[-1])  # Sort by the ratio in ascending order
    
    # Add ranked results for the current error bound to the final results
    results.extend(results_for_eb)

# Print the results
# print(tabulate(results, headers=['Table', 'Error Bound', 'Original Set Size', 'Dict Size', 'List Size', 'Ratio'], tablefmt='grid'))

# Rank tables by dict size/original dict size ratio for each error bound and print the ranked result.
for eb in error_bounds:
    sorted_results = sorted([res for res in results if res[1] == eb], key=lambda x: x[-1])
    print(f"\nRanked results for error bound {eb}:")
    print(tabulate(sorted_results, headers=['Table', 'Error Bound', 'Original Set Size', 'Dict Size', 'List Size', 'Ratio'], tablefmt='grid'))