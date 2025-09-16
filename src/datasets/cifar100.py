import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def loadData(batch=128, valid=5000, workers=2, seed=42):
    transform = transforms.ToTensor()


    trainset = datasets.CIFAR100(root="data", train=True,  download=True, transform=transform)
    testset  = datasets.CIFAR100(root="data", train=False, download=True, transform=transform)


    gen = torch.Generator().manual_seed(seed)
    train_len = len(trainset) - valid
    train_subset, valid_subset = random_split(trainset, [train_len, valid], generator=gen)

    train_loader = DataLoader(train_subset, batch_size=batch, shuffle=True, num_workers=workers)
    val_loader = DataLoader(valid_subset, batch_size=batch, shuffle=False, num_workers=workers)
    test_loader = DataLoader(testset, batch_size=batch, shuffle=False, num_workers=workers)
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = loadData()
    images, labels = next(iter(train_loader))
