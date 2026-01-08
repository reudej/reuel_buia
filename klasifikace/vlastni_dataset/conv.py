import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms

transform = transforms.ToTensor()

ds1 = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

images = torch.stack([x for x, _ in ds1])
labels = torch.tensor([y for _, y in ds1])

torch.save({
    "images": images,
    "labels": labels
}, "cifar10_train.pt")

ds2 = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

images = torch.stack([x for x, _ in ds2])
labels = torch.tensor([y for _, y in ds2])


torch.save({
    "images": images,
    "labels": labels
}, "cifar10_test.pt")
