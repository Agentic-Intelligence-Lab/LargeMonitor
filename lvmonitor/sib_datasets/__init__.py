"""Si-blurry-style dataset registry without baked-in train augmentations."""

from torchvision.datasets import CIFAR10, CIFAR100

from .CUB200 import CUB200
from .CORe50 import CORe50
from .Imagenet_R import Imagenet_R
from .Imagenet_sketch import Imagenet_Sketch
from .TinyImageNet import TinyImageNet

__all__ = [
    "CUB200",
    "CORe50",
    "Imagenet_R",
    "Imagenet_Sketch",
    "TinyImageNet",
    "CIFAR10",
    "CIFAR100",
    "get_dataset",
]

# dataset class, mean, std, num_classes
datasets = {
    "cifar10": (CIFAR10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 10),
    "cifar100": (CIFAR100, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 100),
    "tinyimagenet": (TinyImageNet, (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262), 200),
    "cub200": (CUB200, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 200),
    "imagenet-r": (Imagenet_R, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 200),
    "core50": (CORe50, (0.6003, 0.5684, 0.5414), (0.1785, 0.1912, 0.2008), 50),
    "imagenet_sketch": (Imagenet_Sketch, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 1000),
}


def get_dataset(name: str):
    if name not in datasets:
        raise KeyError(f"Unknown dataset {name!r}. Choose from {list(datasets)}")
    return datasets[name]
