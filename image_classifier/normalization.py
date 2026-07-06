from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),      # -> [0,1], shape (1, 28, 28)
    transforms.Normalize(mean=(0.2860,), std=(0.3530,)),    # -> zero mean
])