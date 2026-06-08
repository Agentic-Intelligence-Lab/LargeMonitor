from typing import Callable, Optional
import os

import torch
from torchvision import datasets
from torchvision.datasets.utils import download_url


class Imagenet_R(torch.utils.data.Dataset):
    """ImageNet-R loader aligned with Disjoint layout (imagenet-r/train|test)."""

    def __init__(
        self,
        root: str,
        train: bool,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
        self.filename = "imagenet-r.tar"
        self.fpath = os.path.join(self.root, "imagenet-r")

        if not os.path.exists(self.fpath):
            if not download:
                raise RuntimeError("Dataset not found. You can use download=True to download it")
            print("Downloading from " + self.url)
            download_url(self.url, self.root, filename=self.filename)
        if not os.path.exists(self.fpath):
            import tarfile

            tar = tarfile.open(os.path.join(self.root, self.filename), "r")
            tar.extractall(self.root)
            tar.close()

        split_dir = "train" if train else "test"
        fpath = os.path.join(self.fpath, split_dir)
        if not os.path.isdir(fpath):
            raise RuntimeError(
                f"Expected ImageNet-R at {fpath} (200 class folders). "
                f"Set --data-dir to the parent of imagenet-r/ (e.g. {self.root})."
            )

        self.data = datasets.ImageFolder(fpath, transform=transform, target_transform=target_transform)
        self.classes = self.data.classes
        self.class_to_idx = self.data.class_to_idx
        self.targets = self.data.targets
        self.samples = self.data.samples

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)
