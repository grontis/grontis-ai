#real data sets are so big they are too big to pass all at once, so they are split into batches
#use a DataLoader

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
x = torch.linspace(-3, 3, 100).unsqueeze(1)
y = 2 * x + 1 + 0.5 * torch.randn(100, 1)

dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

for epoch in range(50):                       # outer: passes over the whole dataset
    for batch_x, batch_y in dataloader:       # inner: one batch at a time
        preds = model(batch_x)
        loss = loss_fn(preds, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"epoch {epoch:2d} | last batch loss {loss.item():.4f}")

