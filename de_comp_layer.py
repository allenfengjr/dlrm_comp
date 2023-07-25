'''
This is PyTorch Module that for compression and decompression.

In this two modules, we define the forward and backward function,



'''
import torch
import torch.nn

class Compression(nn.Module):
    def __init__(self, input_dim, world_size, num_table, num_table_on_device):
        super(Compression, self).__init__()
        # Define parameters 
        # num_table_on_device should get from ext_dist or
        self.compressor = "SZ"
        self.eb = 1e-2
        self.num_table = num_table
        self.num_table_on_device = num_table_on_device
        self.world_size = world_size
        
    
    def forward(self, x, device_id):
        if isinstance(x, list):
            tensor_shape = x[0].shape
            compressed_tensor = torch.cuda.FloatTensor(
                int(self.num_table_on_device*tensor_shape[0]*tensor_shape[1])
            )
            comp_tensor_size = torch.cuda.IntTensor(
                int(self.world_size * self.num_table_on_device)
            )
            # list of device index or device index?
            compression_func_call(
                x, 
                device_id,
                int(tensor_shape[0] * tensor_shape[1]),
                self.world_size,
                [self.eb] * self.num_table_on_device,
                tensor_shape,
                compressed_tensor,
                comp_tensor_size
            )
            chunk_shape = tensor_shape
            chunk_shape[0]/=self.world_size
            return compressed_tensor, comp_tensor_size, chunk_shape
        raise ValueError("Unsupported input type. Please provide data in a supported format.")
    def backward(self, grad_output):
        # grad_output is grads from all ranks via backward func in Alltoall_v
        # directly passback, only reformat to `ly` shape

        return None

class Decompression(nn.Module):
    def __init__(self, input_dim, world_size, num_table, num_table_on_device, chunk_shape):
        super(Compression, self).__init__()
        # Define parameters 
        self.compressor = "SZ"
        self.eb = 1e-2
        self.num_table = num_table
        self.num_table_on_device = num_table_on_device
        self.world_size = world_size
        self.original_shape
    
    def forward(self, x, device_id, offset_tensor):
        if isinstance(x, torch.tensor):
            decompressed_tensor = [ torch.cuda.FloatTensor(chunk_shape) for _ in range(self.num_table)]
            decompression_func_call(
                x,
                device_id,
                offset_tensor,
                decompressed_tensor
            )
            return decompressed_tensor
        raise ValueError("Unsupported input type. Please provide data in a supported format.")
    def backward(self, grad_output)
        return None