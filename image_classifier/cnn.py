import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device)

# --Data--
train_data = datasets.FashionMNIST(
    "data", 
    train=True, 
    download=True, 
    transform=ToTensor())

test_data = datasets.FashionMNIST(
    "data", 
    train=False, 
    download=True, 
    transform=ToTensor())

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# as mentioned in the document, this Model definition is the only thing that changes
# core loop of train and eval is the same, but a different model definition is used
# between this and the MLP.py model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    
model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        preds = model(images)             # 1. forward -> (B,10) logits
        loss = loss_fn(preds, labels)     # 2. measure

        optimizer.zero_grad()             # 3. clear old grads
        loss.backward()                   # 4. compute grads
        optimizer.step()                  # 5. update weights

        running_loss += loss.item() * images.size(0)
        correct += (preds.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, loss_fn, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = loss_fn(preds, labels)

            running_loss += loss.item() * images.size(0)
            correct += (preds.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

for epoch in range(5):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    te_loss, te_acc = evaluate(model, test_loader, loss_fn, device)
    print(f"epoch {epoch+1} | train acc {tr_acc:.3f} | test acc {te_acc:.3f}")