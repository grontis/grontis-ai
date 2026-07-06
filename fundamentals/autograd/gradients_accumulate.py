import torch

x = torch.tensor(3.0, requires_grad=True)

y = x ** 2
y.backward()
print(x.grad)

#to clear we would use:
#x.grad = None
#or
#x.grad.zero_()

#NOTE: same computation here without clearing, gets added onto old gradient!
y = x ** 2
y.backward()
print(x.grad)

