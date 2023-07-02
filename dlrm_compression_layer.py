import torch.nn.Function as Function
class dlrm_compression(Function):
    def __init__(self, compressor, layout, error_bound):
        super(compression, self).__init__()
        self.compressor = compressor
        self.layout = layout
        self.error_bound = error_bound
        return None

    @staticmethod
    def forward(ctx, data):
        '''
        This should be same as dlrm_compressor, just change some interface. 
        '''
        self.compressor(data, self.layout)
        return data_compressed
    
    @staticmethod
    def backward(ctx, grad_output):
        '''
        I assume that before and after compression, data should be approximately equal
        x' = compression(x)
        dx'/dx = 1
        '''
        return grad_output

class dlrm_quantization(Function):
    def __init__(self, bits):
        return None

    @staticmethod
    def forward(data):
        return data
    
    @staticmethod
    def backward(data):
        '''
        I am not sure what to do in quantization backward, should I just set it also 1?
        '''