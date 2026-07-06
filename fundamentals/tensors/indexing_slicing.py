import torch

demoTensor = torch.tensor([[1, 2, 3]])
print(f"demoTensor: {demoTensor}")

print(f"first element: {demoTensor[0]}, type: {demoTensor[0].type()}")

print(f"last element: {demoTensor[-1]}")

demoTensor[0][0] = 99
print(f"first element after update: {demoTensor[0]}")

print(f"t[: , 1]: {demoTensor[:, 0]}")

print(f"t[t > 3]: {demoTensor[demoTensor > 3]}")