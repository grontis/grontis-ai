import torch
import torch.nn as nn

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- data: the same noisy line y = 2x + 1 ---
x = torch.linspace(-3, 3, 100).unsqueeze(1)        # (100, 1)
y = 2 * x + 1 + 0.5 * torch.randn(100, 1)

# --- the model as a proper nn.Module subclass ---
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)              # one weight, one bias

    def forward(self, x):
        return self.linear(x)

model = LinearRegression().to(device)              # .to() moves ALL params at once
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

x, y = x.to(device), y.to(device)

# --- the exact same 1.3 skeleton ---
for epoch in range(200):
    preds = model(x)                # 1. forward  (calls forward() via __call__)
    loss = loss_fn(preds, y)        # 2. measure
    optimizer.zero_grad()           # 3. clear  (every param)
    loss.backward()                 # 4. compute grads
    optimizer.step()                # 5. update (every param)
 
    if epoch % 40 == 0:
        print(f"epoch {epoch:3d} | loss {loss.item():.4f}")

w, b = model.linear.weight.item(), model.linear.bias.item()
print(f"\nLearned: y = {w:.2f}x + {b:.2f}   (target: y = 2x + 1)")