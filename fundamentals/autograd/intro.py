import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2

print(y)                 # tensor(9., grad_fn=<PowBackward0>)
print(y.grad_fn)         # <PowBackward0 object ...>
print(y.requires_grad)   # True — it inherited tracking from x
print(x.grad)            # None — gradients don't exist until .backward()

y.backward()
print(x.grad)            # tensor(6.)