import extend_distributed as ext_dist
import torch
import numpy as np
import os
import time
#import torch.profiler
if __name__ == "__main__":
    # init a communication size matrix
    rank, local_rank = -1, -1
    ext_dist.init_distributed(rank=rank, use_gpu=True, backend="nccl")
    ext_dist.print_all("my rank is, ",ext_dist.my_rank)
    device = torch.device('cuda', ext_dist.my_local_rank)
    datatype = torch.float32
    
    # generate sending data
    tensor_size = 8192*100*ext_dist.my_size
    emb_len = 32
    table_num = [2]*ext_dist.my_size
    table_num[0] = 3
    r_tolerance = 1e-2
    send_tensor_list = [torch.randn((tensor_size, emb_len), dtype=datatype).to(device) for _ in range(table_num[ext_dist.my_rank])]
    
    # Part 1: compression
    # outputs buffer
    output_tensor = torch.cuda.FloatTensor(int(tensor_size*emb_len*table_num[ext_dist.my_rank]))

    # compressed data size tensor
    comp_size_tensor = torch.cuda.ByteTensor(int(table_num[ext_dist.my_rank] * 4))

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
    decompression_func_call(
        compressed_tensor = receive_tensor, # received tensor list, 2d tensors
        offset_tensor = torch.tensor(size_each_table).cuda(), # this is a count tensor, may need another prefix sum caculation
        offset_size = ext_dist.my_size,
        device_id = device, # same as compression_func_call
        output_tensor_list = decompressed_tensor_list
    )

