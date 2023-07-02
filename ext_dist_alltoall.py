import extend_distributed as ext_dist
import torch
import numpy as np
import os
import time
import torch.profiler
if __name__ == "__main__":
    # init a communication size matrix
    rank, local_rank = -1, -1
    ext_dist.init_distributed(rank=rank, use_gpu=True, backend="nccl")
    ext_dist.print_all("my rank is, ",ext_dist.my_rank)
    # generate alltoall_v data
    device = torch.device('cuda', ext_dist.my_local_rank)
    # generate sending data and receiving buffer
    tensor_size = 8192*ext_dist.my_size
    table_num = [2]*ext_dist.my_size
    table_num[0] = 3
    tolerance = 1e-1

    # generate alltoall_v data
    tmp_tensor = torch.randn((tensor_size, 32))
    send_tensor = []
    for i in range(sum(table_num)):
        tmp_tensor = tmp_tensor.to(device)
        send_tensor.append(tmp_tensor)
    # ext_dist.print_all(send_tensor)
    # Print the result
    # ext_dist.print_all("Tensor is dense is, ", is_dense)
    # ext_dist.print_all("Tensor is cuda is, ", is_cuda)
    # all to all
    iter = 0
    with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=2,
                    warmup=2,
                    active=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./Log_A2A_original_result', worker_name='alltoall_v'),
                record_shapes=True,
                profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
                with_stack=True
        ) as p:
        while iter<30:
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            print(current_time)
            a2a_req = ext_dist.alltoall(send_tensor,table_num)
            receive_tensor = a2a_req.wait()
            if iter < 5:
                p.step()            
            iter+=1
            ext_dist.barrier()
    print("sucess")
                                    