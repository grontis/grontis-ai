from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

print(len(train_data), len(test_data))
img, label = train_data[0]
print(img.shape, img.dtype)
print(train_data.classes)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 6, figsize=(12, 2))
for i, ax in enumerate(axes):
    img, label = train_data[i]
    ax.imshow(img.permute(1, 2, 0))
    ax.set_title(train_data.classes[label])
    ax.axis("off")

plt.show()