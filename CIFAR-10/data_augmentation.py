from torchvision import transforms
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader

normalize = transforms.Normalize(
    mean=(0.4914, 0.4822, 0.4465),   # per-channel mean (R, G, B)
    std=(0.2470, 0.2435, 0.2616),    # per-channel std
)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4), #pad to 40x40 after cropping random 32x32 window
    transforms.RandomHorizontalFlip(), # 50% chance of mirror flip
    transforms.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([
    #No augmentation = deterministic
    transforms.ToTensor(),
    normalize
])

train_data = datasets.CIFAR10(
    "data",
    train=True,
    download=True,
    transform=train_transform,
)

test_data = datasets.CIFAR10(
    "data",
    train=False,
    download=True,
    transform=test_transform
)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)