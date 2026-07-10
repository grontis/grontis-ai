import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import os, time

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

cfg = dict(
    batch_size=128,
    epochs=50,
    lr=1e-3,
    weight_decay=5e-2,
    run_name="runs/06_full_amp"
)

normalize = transforms.Normalize(
    (0.4914, 0.4822, 0.4465),
    (0.2470, 0.2435, 0.2616))

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

train_data = datasets.CIFAR10(
    "data",
    train=True,
    download=True,
    transform=train_transform
)

test_data = datasets.CIFAR10(
    "data",
    train=False,
    download=True,
    transform=test_transform
)

train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

class DeepCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4 , 512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))
    
model = DeepCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg["lr"],
    weight_decay=cfg["weight_decay"])

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
scaler = torch.amp.GradScaler("cuda")

def train_one_epoch(model, loader):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        with torch.amp.autocast("cuda"):
            preds = model(images)
            loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        correct += (preds.argmax(1) == labels).sum().item()
        total += labels.size(0)
    
    return running_loss / total, correct / total

def evaluate(model, loader):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = loss_fn(preds, labels)
            running_loss += loss.item * images.size(0)
            correct += (preds.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


if __name__ == "__main__":
    writer = SummaryWriter(cfg["run_name"])
    os.makedirs("checkpoints", exist_ok=True)
    best_acc = 0.0

    for epoch in range(cfg["epochs"]):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, train_loader)
        te_loss, te_acc = evaluate(model, test_loader)
        scheduler.step()

        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("loss/test", te_loss, epoch)
        writer.add_scalar("acc/train", tr_acc, epoch)
        writer.add_scalar("acc/test", te_acc, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), "checkpoints/cifar_best.pt")

        print(f"epoch {epoch+1:3d} | train {tr_acc:.3f} | test {te_acc:.3f} "
              f"| lr {scheduler.get_last_lr()[0]:.5f} | {time.perf_counter()-t0:.1f}s"
              + ("  <- new best" if te_acc == best_acc else ""))

    writer.close()
    print(f"best test accuracy: {best_acc:.3f}")