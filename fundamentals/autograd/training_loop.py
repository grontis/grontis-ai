import torch
import torch.nn as nn

torch.manual_seed(42)

print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.linspace(-3, 3, 100).unsqueeze(1)
y = 2 * x + 1 + 0.5 * torch.randn(100, 1)

# --- the three "batteries-included" pieces ---
model = nn.Linear(1, 1).to(device)                 # holds a weight + bias for us
loss_fn = nn.MSELoss()                             # mean squared error, prebuilt
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)  # knows how to update

x, y = x.to(device), y.to(device)

# --- the skeleton ---
for epoch in range(200):
    preds = model(x)                # 1. forward
    loss = loss_fn(preds, y)        # 2. measure

    optimizer.zero_grad()           # 3. clear
    loss.backward()                 # 4. compute grads
    optimizer.step()                # 5. update

    if epoch % 40 == 0:
        print(f"epoch {epoch:3d} | loss {loss.item():.4f}")

# pull the learned numbers out of the model
w, b = model.weight.item(), model.bias.item()
print(f"\nLearned: y = {w:.2f}x + {b:.2f}   (target: y = 2x + 1)")