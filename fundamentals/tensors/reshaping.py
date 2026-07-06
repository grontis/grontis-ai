import torch

demoTensor = torch.randn(3, 4)
print(f"Tensor: {demoTensor}")

#view: returned tensor will share the underlying data with original tensor
viewedTensor = demoTensor.view(4, 3)
print(f"viewedTensor: {viewedTensor}")

#reshape: depending on the input, may or may not return a view or copy of original tensor
reshapedTensor = demoTensor.reshape(2, 6)
print(f"reshapedTensor: {reshapedTensor}")

unsqueezedTensor = torch.unsqueeze(demoTensor, 2)
print(f"unsqueezedTensor: {unsqueezedTensor}")

resqueezedTensor = torch.squeeze(unsqueezedTensor)
print(f"resqueezedTensor: {resqueezedTensor}")

transposedTensor = demoTensor.transpose(1, 0)
print(f"transposedTensor: {transposedTensor}")
