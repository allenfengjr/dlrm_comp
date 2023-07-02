import extend_distributed as ext_dist
import torch
import numpy as np
import os
import zfpy
import ctypes
from ctypes import *
from random import random
from compressor import *
import time
import torch.profiler

'''
In this Python Script, we just
'''

if __name__ == "__main__":
    # init a communication size matrix
    rank, local_rank = -1, -1    
    ext_dist.init_distributed(rank=rank, use_gpu=True, backend="nccl")
    ext_dist.print_all("my rank is, ",ext_dist.my_rank)
    device = torch.device('cuda', ext_dist.my_local_rank)
    datatype = torch.float32
    # hardcode data size and number of table
    tensor_size = 32*8192*100*ext_dist.my_size
    table_num = [2]*ext_dist.my_size
    table_num[0] = 3
    tolerance = 2e-1

    # Here we use FZ-GPU to get a fake compression overhead
    # generate alltoall_v data
    original_data = torch.randn((table_num[ext_dist.my_rank], tensor_size), dtype=datatype)
    split_tensors = torch.chunk(original_data, ext_dist.my_size, dim=1)
    max_iteration_num = 10
    iter = 0
    with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=2,
                    warmup=2,
                    active=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./Log_A2A_comp_result', worker_name='alltoall_v'),
                record_shapes=True,
                profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
                with_stack=True
        ) as p:
        while iter < max_iteration_num:
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            print("with compression: ", current_time)
            for s in split_tensors:
                s.numpy().tofile("need_compression_tensor.bin")
                #s = torch.tensor(np.fromfile("./fail_tensor.bin",np.float32), dtype=torch.float32).to(device)
                #s_o = s.new_empty(s.shape).to(device)
                s = s.to(device)
                s_o = torch.empty((s.shape[0], s.shape[1]),dtype=torch.float32)
                s_o = s_o.to(device)
                ext_dist.print_all(s.shape, s_o.shape)
                run_pfz(s, s_o, s.shape[0], s.shape[1], 1, 1e-3)
            # ext_dist.barrier()# test for FZ-GPU sync
            # end timer compression
            compressed_data = []
            data_size = []
            for r in split_tensors:
                # data that need to send to each rank
                for t in r:
                    s = t.detach().cpu().numpy()
                    a_tolerance = (s.max()-s.min()) * tolerance
                    c_t = zfpy.compress_numpy(s,tolerance = a_tolerance)
                    # print("compression ratio", (s.size*s.itemsize)/len(c_t))
                    compressed_data.append(c_t)
                    data_size.append(len(c_t))

            # generate metadata we need
            t_data_size = torch.tensor(data_size).to(device)

            meta_send_cnt = [table_num[ext_dist.my_rank]] * ext_dist.my_size
            meta_a2a_v_req = ext_dist.alltoall_v(
                t_data_size,
                send_cnt=meta_send_cnt,
                receive_cnt=table_num
            )
            size_each_table = meta_a2a_v_req.wait()
            size_each_table = size_each_table.tolist()# in profile result, tolist() spend a lot of time, why?

            # Here we spend a lot of time using torch.cat method, I believe transfer the list to tensor directly will have better performance
            #send_tensor = torch.cat([torch.tensor(list(b),dtype=torch.uint8) for b in compressed_data]).to(device)
            # second method, only works when compressed_data[i] is equal length
            #compressed_data_tensor = torch.tensor(compressed_data, dtype=torch.uint8)
            #send_tensor = compressed_data_tensor.flatten().to(device)
            compressed_data = b''.join(compressed_data)
            send_tensor = torch.frombuffer(compressed_data, dtype=torch.uint8).to(device)
            send_cnt = [sum(data_size[i:i+table_num[ext_dist.my_rank]]) for i in range(0, len(data_size), table_num[ext_dist.my_rank])]
            receive_cnt = [sum(size_each_table[start:start+size])
                            for start, size in zip([sum(table_num[:i])
                                for i in range(len(table_num)+1)], table_num)
                        ]
            # Now metadata is transferd, we have a list which length is #table, this is for spliting the receice tensors
            # We need to sum part of the list to get a new length which equal to worldsize    
            # In first step, I will compress
            sub_size = 1 #send_tensor.shape[1]*send_tensor.shape[2]
            send_cnt = ext_dist.multiply_list_by_number(send_cnt, sub_size)
            receive_cnt = ext_dist.multiply_list_by_number(receive_cnt, sub_size)
            
            # all to all
            a2q_v_req = ext_dist.alltoall_v(send_tensor, send_cnt = send_cnt, receive_cnt = receive_cnt)
            receive_tensor = a2q_v_req.wait()
            
            # reshape and decompress the data
            compressed_tensor = torch.split(receive_tensor, size_each_table)
            decompressed_numpy = []
            for i in compressed_tensor:
                tmp = i.cpu().numpy().tobytes()
                decompressed_numpy.append(zfpy.decompress_numpy(tmp))
            if iter < 6:
                p.step()
            iter+=1
    print("success")
