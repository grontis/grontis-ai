from torchvision import transforms

normalize = transforms.Normalize(
    mean=(0.4914, 0.4822, 0.4465),   # per-channel mean (R, G, B)
    std=(0.2470, 0.2435, 0.2616),    # per-channel std
)