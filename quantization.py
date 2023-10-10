import os
import numpy as np
import torch

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

for tb in range(26):
    filename = f"/N/scratch/haofeng/TB_emb_32/embedding_output_vector_table_{tb}_epoch_10_iter_0.bin"
    data = np.fromfile(filename, dtype=np.float32)
    fbgemm_quantization(data, filename)
