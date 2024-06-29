import os
import numpy as np
import torch

DIR = "/home/haofeng/SC_emb_data/SC_TB_emb"
PATTERN = "EMB_{table}_iter_{iter}.bin"

def quantization(original_arr, eb, filename):
    eb = (original_arr.max() - original_arr.min()) * eb
    print("absolute error bound, ", eb)
    quantization_arr = np.round(original_arr* (1/(eb*2))).astype(np.int32)
    #Baixi
    quantization_arr += abs(quantization_arr.min())+1
    synthetic_outlier = quantization_arr[np.where(quantization_arr>=65536)].astype(np.float32)
    synthetic_outlier.tofile(f"{filename}.outlier")

    print("Range: {}  {}".format(quantization_arr.min(),quantization_arr.max()))
    quantization_arr.tofile(f"{filename}.quan")
    return quantization_arr

def fbgemm_quantization(original_arr, filename):
    send_tensor = torch.tensor(original_arr).cuda()
    send_tensor = torch.ops.fbgemm.FloatToHFP8Quantized(original_arr, 4, 1, 10.0)
    # receive_tensor = torch.ops.fbgemm.HFP8QuantizedToFloat(receive_tensor, 4, 1)
    send_tensor = send_tensor.cpu().numpy()
    send_tensor.tofile(f"quan_{filename}")
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


# Constants for error bounds
TIGHTEN_EB_TABLES = {0, 9, 10, 19, 20, 21, 22}
LOOSEN_EB_TABLES = {5, 8, 12, 15, 16, 17, 18, 24, 25}
TIGHTEN_EB_VALUE = 0.02
LOOSEN_EB_VALUE = 0.02
BASE_ERROR_BOUND = 0.02
NUM_TABLES = 26
NUM_ITERATIONS = 24

# Generating EB list with error bounds
EB = []
# for table in range(NUM_TABLES):
#     if table in TIGHTEN_EB_TABLES:
#         base_eb = TIGHTEN_EB_VALUE
#     elif table in LOOSEN_EB_TABLES:
#         base_eb = LOOSEN_EB_VALUE
#     else:
#         base_eb = BASE_ERROR_BOUND

#     table_eb = []
#     for iter in range(NUM_ITERATIONS):
#         if iter < 8:
#             eb = (base_eb * 2) - (iter * (base_eb / 8))
#         else:
#             eb = base_eb
#         table_eb.append(round(eb, 5))  # Rounding to 5 decimal places
#     EB.append(table_eb)

# constant
for table in range(NUM_TABLES):
    if table in TIGHTEN_EB_TABLES:
        base_eb = TIGHTEN_EB_VALUE
    elif table in LOOSEN_EB_TABLES:
        base_eb = LOOSEN_EB_VALUE
    else:
        base_eb = BASE_ERROR_BOUND

    table_eb = []
    for iter in range(NUM_ITERATIONS):
        if iter < 8:
            eb = (base_eb)
        else:
            eb = base_eb
        table_eb.append(round(eb, 5))  # Rounding to 5 decimal places
    EB.append(table_eb)
print(EB)
for table in range(len(EB)):  # Loop through tables based on the EB list
    for iter in range(1,len(EB[table])):  # Loop through iterations for each table
        # Construct the filename
        filename = PATTERN.format(table=table, iter=iter)
        filename = os.path.join(DIR, filename)
        error_bound = EB[table][iter]
        if os.path.isfile(filename):
            error_bound = EB[table][iter]
            data = np.fromfile(filename, dtype=np.float32)
            quantization(data, error_bound, filename)