import torch

# Assuming x is your tensor on the GPU and requires_grad is True
x = torch.rand(5, 5, device='cpu', requires_grad=True)
print(x.data)

# The error bound (eb)
eb = 0.1

# Quantization and dequantization directly on the tensor data
def quantize_and_dequantize_data(x, eb):
    # Ensure calculations are performed on the same device as x (GPU in this case)
    eb = torch.tensor(eb, device=x.device)
    
    # Quantization
    x.data = (x.data / (2*eb)).round() * (2*eb)  # Directly modify the tensor data

# Apply the quantization and dequantization
quantize_and_dequantize_data(x, eb)

# x.data is now modified in-place without affecting the gradient computation
print(x.data)
