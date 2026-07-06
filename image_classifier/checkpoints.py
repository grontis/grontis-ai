import torch
import os
os.makedirs("checkpoints", exist_ok=True)

# Save when you beat best test accuracy
torch.save(model.state_dict(), "checkpoints/best.pt")

# Load later (into a freshly constructed model of the SAME class) ---
model = CNN().to(device)
model.load_state_dict(torch.load("checkpoints/best.pt", map_location=device))
model.eval()   # switch to eval mode before using it for inference