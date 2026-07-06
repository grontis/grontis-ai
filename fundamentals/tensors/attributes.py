import torch

demoTensor = torch.randn(3, 4)
print(f"Tensor: {demoTensor}")

print(f".shape: {demoTensor.shape}")
print(f".dtype: {demoTensor.dtype}")
print(f".device: {demoTensor.device}")

print(f".itemsize: {demoTensor.itemsize}")

