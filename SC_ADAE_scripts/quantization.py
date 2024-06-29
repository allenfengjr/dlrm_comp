import os
import sys
import numpy as np
import torch
inttype = np.int16

# Terabyte

# Table-wise and Iteration-wise error bound
def build_error_bound():
    TIGHTEN_EB_TABLES = {0, 9, 10, 19, 20, 21, 22}
    LOOSEN_EB_TABLES = {5, 8, 12, 15, 16, 17, 18, 24, 25}
    TIGHTEN_EB_VALUE = 0.01
    LOOSEN_EB_VALUE = 0.03
    BASE_ERROR_BOUND = 0.02
    NUM_TABLES = 26
    NUM_ITERATIONS = 23
    # Generating EB list with error bounds
    EB = []
    for table in range(NUM_TABLES):
        if table in TIGHTEN_EB_TABLES:
            base_eb = TIGHTEN_EB_VALUE
        elif table in LOOSEN_EB_TABLES:
            base_eb = LOOSEN_EB_VALUE
        else:
            base_eb = BASE_ERROR_BOUND
        table_eb = []
        # default decay function is 'step-wise decay'
        decay_stage = sys.argv[3]
        for iter in range(NUM_ITERATIONS):
            if iter < decay_stage:
                eb = (base_eb * 2) - (iter * (base_eb / decay_stage))
            else:
                eb = base_eb
        EB.append(table_eb)
    return EB

def quantization(original_arr, eb, filename):
    eb = (original_arr.max() - original_arr.min()) * eb
    print("absolute error bound, ", eb)
    quantization_arr = np.round(original_arr* (1/(eb*2))).astype(inttype)
    quantization_arr += abs(quantization_arr.min())+1
    synthetic_outlier = quantization_arr[np.where(quantization_arr>=65536)].astype(np.float32)
    # synthetic_outlier.tofile(f"{filename}.outlier") # no outlier with my error bound

    print("Range: {}  {}".format(quantization_arr.min(),quantization_arr.max()))
    quantization_arr.tofile(f"{filename}.quan")
    return quantization_arr

def fbgemm_quantization(original_arr, filename):
    send_tensor = torch.tensor(original_arr).cuda()
    send_tensor = torch.ops.fbgemm.FloatToHFP8Quantized(original_arr, 4, 1, 10.0)
    # receive_tensor = torch.ops.fbgemm.HFP8QuantizedToFloat(receive_tensor, 4, 1)
    send_tensor = send_tensor.cpu().numpy()
    send_tensor.tofile(f"{filename}.fbgemm_quan")
    return None

def pytorch_quantization(original_arr, filename):
    send_tensor = torch.tensor(original_arr)
    min_value = send_tensor.min().item()
    max_value = send_tensor.max().item()
    scale = (max_value-min_value)/255
    zero_point = 0

    # Quantize the tensor
    quantized_tensor = torch.quantize_per_tensor(send_tensor, scale=scale, zero_point=zero_point, dtype=torch.qint8)
    # Dequantize the tensor
    #dequantized_tensor = quantized_tensor.dequantize()
    quantized_nparray = quantized_tensor.int_repr().numpy()
    quantized_nparray.tofile(f"{filename}.pytorch_quan")
    return quantized_tensor


EMB_file_path = sys.argv[2] # "/N/u/haofeng/BigRed200/SC_TB_emb"
for tb in range(26):
    for iter in range(1,23):
        filename = f"{EMB_file_path}/EMB_{tb}_iter_{iter}.bin"
        data = np.fromfile(filename, dtype=np.float32)
        EB = build_error_bound()
        eb = EB[tb][iter] # get table-wise and iter-wise eb
        quantization(data, eb, filename)