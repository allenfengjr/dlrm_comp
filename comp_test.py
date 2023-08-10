import extend_distributed_new as ext_dist
import torch
import numpy as np
import os
import time
import ctypes
from ctypes import *

so_path = '/N/u/haofeng/BigRed200/fz/fz-gpu.so'
pfz = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)

#import torch.profiler
if __name__ == "__main__":
    # init a communication size matrix
    rank, local_rank = -1, -1
    ext_dist.init_distributed(rank=rank, use_gpu=True, backend="nccl")
    ext_dist.print_all("my rank is, ",ext_dist.my_rank)
    device = torch.device('cuda', ext_dist.my_local_rank)
    device_id = ext_dist.my_local_rank
    datatype = torch.float32
    
    # generate sending data
    tensor_size = 8192*8
    emb_len = 32
    table_num = [2]*ext_dist.my_size
    # table_num[0] = 3
    global_table_num = sum(table_num)
    r_tolerance = 1e-2
    send_tensor_list = [torch.ones((tensor_size * emb_len), dtype=datatype).cuda() for _ in range(table_num[ext_dist.my_rank])]
    ext_dist.print_all("python send tensor list first element, ", send_tensor_list[0][0])
    ext_dist.print_all("send tensor size, ", send_tensor_list[0].shape)
    # Part 1: compression
    # outputs buffer
    # compressed_tensor_gpu = torch.empty(int(tensor_size*emb_len*table_num[ext_dist.my_rank]*4), dtype = torch.uint8).cuda()
    compressed_tensor_gpu = torch.tensor([3 for i in range(tensor_size * emb_len * 4 * table_num[ext_dist.my_rank])], dtype = torch.uint8).cuda()

    ext_dist.print_all("compressed tensor size", compressed_tensor_gpu.shape)
    # compressed data size tensor
    comp_size_tensor = torch.empty(int(table_num[ext_dist.my_rank] * 4), dtype = torch.int32).cuda()
    
    '''
    compression_func_call(
        original_tensor_list = send_tensor_list, # len(send_tensor_list) = table_num[ext_dist.my_rank]
        device_id = device, # if you need device id instead of torch.device, please use ext_dist.my_local_rank
        input_size = int(tensor_size * emb_len * table_num[ext_dist.my_rank]), # python will not handle int32/int64
        num_table = int(table_num[ext_dist]), # may use dist.n_emb_per_rank list in DLRM
        world_size = int(ext_dist.my_size),
        dimension = [tensor_size, emb_len], # in DLRM code, it is tensor.shape => a tuple
        output_tensor = output_tensor, # I use 1d tensor as buffer
        compressed_size_tensor = comp_size_tensor
    )
    '''
    # launch compression
    send_tensor_list_ptr = [cast(_.data_ptr(), POINTER(c_float)) for _ in send_tensor_list]
    input_list_c = (POINTER(c_float) * table_num[ext_dist.my_rank])(*send_tensor_list_ptr)
    input_list_prt = pointer(input_list_c)
    dimension_info = (c_int*3)(*[int(tensor_size/ext_dist.my_size), emb_len, 1])
    dimension_info_ptr = pointer(dimension_info)
    compressed_ptr = cast(compressed_tensor_gpu.data_ptr(), POINTER(c_uint8))
    compressed_size_ptr = cast(comp_size_tensor.data_ptr(), POINTER(c_uint8))
    input_size_c = c_int(int(tensor_size * emb_len))
    input_list_size = c_int(int(table_num[ext_dist.my_rank]))
    world_size = c_int(int(ext_dist.my_size))
    error_bound = c_float(0.1)
    compressed_size_c = (c_int * (ext_dist.my_size * table_num[ext_dist.my_rank]))()
    compressed_size_ptr = pointer(compressed_size_c)
    print("input size is")
    pfz.pfzCompress(
        input_list_prt,
        c_int(device_id),
        input_size_c,
        input_list_size,
        world_size,
        error_bound,
        dimension_info_ptr,
        compressed_ptr,
        compressed_size_ptr
    )
    for i in range(ext_dist.my_size * table_num[ext_dist.my_rank]):
        comp_size_tensor[i] = compressed_size_c[i]
    ext_dist.print_all("compressed size, ", comp_size_tensor)
    ext_dist.print_all("compressed pointer, ", compressed_size_c[2])
    # Part 2: Alltoall_v communication
    # meta data communication
    meta_send_cnt = [table_num[ext_dist.my_rank]] * ext_dist.my_size
    meta_a2a_v_req = ext_dist.alltoall_v(
        comp_size_tensor,
        send_cnt=meta_send_cnt,
        receive_cnt=table_num
    )
    size_each_table = meta_a2a_v_req.wait()
    size_each_table = size_each_table.tolist()
    ext_dist.print_all("meta alltoall finish")
    # calculate the send_cnt and receice_cnt for data alltoall
    send_cnt = [sum(comp_size_tensor[i:i+table_num[ext_dist.my_rank]]) for i in range(0, len(comp_size_tensor), table_num[ext_dist.my_rank])]
    receive_cnt = [sum(size_each_table[start:start+size])
                    for start, size in zip([sum(table_num[:i])
                        for i in range(len(table_num)+1)], table_num)
                ]
    a2q_v_req = ext_dist.alltoall_v(output_tensor, send_cnt = send_cnt, receive_cnt = receive_cnt)
    receive_tensor = a2q_v_req.wait()

    # Part 3: Decompression
    # create decompressed tensor list buffer
    decompressed_tensor_list = [
        torch.cuda.FloatTensor((int(tensor_size/ext_dist.my_size), emb_len))
        for _ in range(sum(table_num))
    ]
    '''
    decompression_func_call(
        compressed_tensor = receive_tensor, # received tensor list, 2d tensors
        offset_tensor = torch.tensor(size_each_table).cuda(), # this is a count tensor, may need another prefix sum caculation
        offset_size = ext_dist.my_size,
        device_id = device, # same as compression_func_call
        output_tensor_list = decompressed_tensor_list
    )
    '''
    compressed_size = [s for s in size_each_table]
    offset_list = list(accumulate(compressed_size))
    offset_list = [0, ] + offset_list[:-1]
    offset_list_c = (c_int*global_table_num)(*offset_list)
    print(offset_list)
    offset_list_ptr = pointer(offset_list_c)

    offset_list_size = c_int(global_table_num)

    decompressed_tensor_ptr_list = [cast(decompressed_tensor_list[i].data_ptr(), POINTER(c_float)) for i in range(ext_dist.my_size)]
    decompressed_tensor_ptr_list_c = (POINTER(c_float) * global_table_num)(*decompressed_tensor_ptr_list)
    decompressed_tensor_ptr_c = pointer(decompressed_tensor_ptr_list_c)
    compressed_ptr = cast(receive_tensor.data_ptr(), POINTER(c_uint8))

    # launch decompression
    pfz.pfzDecompress(
        compressed_ptr,
        offset_list_ptr,
        offset_list_size,
        c_int(device_id),
        decompressed_tensor_ptr_c
    )
