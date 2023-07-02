import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def run(rank, size):
    """ Distributed function to be implemented later. """
    group = dist.new_group(range(size))  # creates a new distributed group
    tensor = torch.rand((10,))  # each process creates its random tensor
    tensor_list = [torch.zeros(10,) for _ in range(size)]  # prepare the list for storing results
    
    # perform all_to_all communication
    dist.all_to_all_single(tensor_list, tensor, group=group)
    
    print('Rank ', rank, ' has data ', tensor_list)

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    size = 4  # number of processes
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
