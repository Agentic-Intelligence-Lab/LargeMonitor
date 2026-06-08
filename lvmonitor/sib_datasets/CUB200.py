from typing import Callable, Optional
import os
import tarfile
import zipfile

import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url


class CUB200(Dataset):
    def __init__(
        self,
        root: str,
        train: bool,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__()

        self.root = os.path.expanduser(root)
        self.url = "https://data.deepai.org/CUB200(2011).zip"
        self.filename = "CUB200(2011).zip"

        fpath = os.path.join(self.root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
                raise RuntimeError("Dataset not found. You can use download=True to download it")
            print("Downloading from " + self.url)
            download_url(self.url, self.root, filename=self.filename)
        if not os.path.exists(os.path.join(self.root, "CUB_200_2011")):
            with zipfile.ZipFile(fpath, "r") as zip_ref:
                zip_ref.extractall(self.root)
            with tarfile.open(os.path.join(self.root, "CUB_200_2011.tgz"), "r") as tar_ref:
                tar_ref.extractall(self.root)

        image_root = self.root + "/CUB200-2011/images"
        full_set = ImageFolder(image_root, transform=transform, target_transform=target_transform)
        len_train = int(len(full_set) * 0.8)
        len_val = len(full_set) - len_train
        train_subset, test_subset = random_split(
            full_set,
            [len_train, len_val],
            generator=torch.Generator().manual_seed(42),
        )
        self.dataset = train_subset if train else test_subset
        self.classes = full_set.classes
        self.targets = [full_set.targets[i] for i in self.dataset.indices]

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return len(self.dataset)
