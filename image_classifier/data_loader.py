from torch.utils.data import DataLoader
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

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)