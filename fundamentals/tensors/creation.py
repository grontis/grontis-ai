import torch

zerosTensor = torch.zeros(2, 3)
print(zerosTensor)

onesTensor = torch.ones(2, 4)
print(onesTensor)

randnTensor = torch.randn(2, 2)
print(randnTensor)

arangeTensor = torch.arange(2, 5)
print(arangeTensor)

torchTensor = torch.tensor([9, 4, 5])
print(torchTensor)

copiedTensor = torchTensor.detach().clone()
print(copiedTensor)