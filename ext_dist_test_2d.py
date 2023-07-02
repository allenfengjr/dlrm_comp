import extend_distributed as ext_dist
import torch
import numpy as np
import os
import time
import torch.profiler

enable_profiling = os.environ.get("enable_profiling", None)
if __name__ == "__main__":
    # init a communication size matrix
    rank, local_rank = -1, -1
    ext_dist.init_distributed(rank=rank, use_gpu=True, backend="nccl")
    ext_dist.print_all("my rank is, ",ext_dist.my_rank)
    # generate alltoall_v data
    device = torch.device('cuda', ext_dist.my_local_rank)
    datatype = torch.float32
    # generate sending data and receiving buffer
    tensor_size = 8192*100*ext_dist.my_size
    emb_len = 32
    table_num = [2]*ext_dist.my_size
    table_num[0] = 3
    tolerance = 1e-1

    # generate alltoall_v data
    send_tensor = torch.randn((table_num[ext_dist.my_rank]*tensor_size, emb_len), dtype=datatype).to(device)
    
    #a2q_v_req = ext_dist.alltoall(send_tensor)
    # actually the table size right now
    send_cnt = [int(table_num[ext_dist.my_rank])]*ext_dist.my_size
    receive_cnt = table_num
    # ext_dist.print_all(send_tensor)
    is_dense = not send_tensor.is_sparse
    is_cuda = send_tensor.is_cuda

    # Print the result
    # ext_dist.print_all("Tensor is dense is, ", is_dense)
    # ext_dist.print_all("Tensor is cuda is, ", is_cuda)

    # I do sub_size here to times a constant number for all elements in a List
    # Not sure it should be the first dimension number or total size number of cnt list
    sub_size = int(tensor_size/ext_dist.my_size)
    send_cnt = ext_dist.multiply_list_by_number(send_cnt, sub_size)
    receive_cnt = ext_dist.multiply_list_by_number(receive_cnt, sub_size)
    print(send_cnt)
    print(receive_cnt)
    # all to all
    iter = 0
    with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=2,
                    warmup=2,
                    active=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./Log_A2A_2D_32_result', worker_name='alltoall_v'),
                record_shapes=True,
                profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
                with_stack=True
        ) as p:
        while iter<10:
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            print(current_time)
            a2q_v_req = ext_dist.alltoall_v(send_tensor, send_cnt = send_cnt, receive_cnt = receive_cnt)
            receive_tensor = a2q_v_req.wait()
            if iter < 6 and enable_profiling == "true":
                p.step()
            iter+=1
    print("sucess")
                                    
