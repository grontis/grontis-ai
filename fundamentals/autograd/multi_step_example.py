import torch

x = torch.tensor(2.0, requires_grad=True)

a = x * 3        # a = 3x
b = a + 1        # b = 3x + 1
c = b ** 2       # c = (3x + 1)²

c.backward()
print(x.grad)    # 42.0