import torch
from torchvision import datasets
from torchvision.transforms import ToTensor


train_data = datasets.FashionMNIST(
    root="data",            # where to store files on disk
    train=True,             # the 60k training split
    download=True,          # fetch if not already present
    transform=ToTensor(),   # how to convert each raw image (see below)
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,            #The 10k test split
    download=True,
    transform=ToTensor(),
)

print(len(train_data), len(test_data))

img, label = train_data[0] #accessing via index gives one (image, label) pair
print(img.shape, img.dtype)
print(label)

img, _ = train_data[0]
print(img.min().item(), img.max().item())   # ~0.0 ... ~1.0, never 255