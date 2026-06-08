from typing import Callable, Optional
import os
import zipfile

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url


class TinyImageNet(ImageFolder):
    def __init__(
        self,
        root: str,
        train: bool,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        self.root = os.path.expanduser(root)
        # self.url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        # self.filename = "tiny-imagenet-200.zip"

        # fpath = os.path.join(self.root, self.filename)
        # if not os.path.isfile(fpath):
        #     if not download:
        #         raise RuntimeError("Dataset not found. You can use download=True to download it")
        #     print("Downloading from " + self.url)
        #     download_url(self.url, self.root, filename=self.filename)
        # if not os.path.exists(os.path.join(self.root, "tiny-imagenet-200")):
        #     with zipfile.ZipFile(fpath, "r") as zip_ref:
        #         zip_ref.extractall(self.root)

        self.path = self.root + "/tiny-imagenet-200/"
        if train:
            super().__init__(
                self.path + "train",
                transform=transform,
                target_transform=target_transform,
            )
            self.classes = []
            with open(self.path + "wnids.txt", "r", encoding="utf-8") as f:
                for line_id in f.readlines():
                    self.classes.append(line_id.split("\n")[0])
            self.class_to_idx = {clss: idx for idx, clss in enumerate(self.classes)}
            self.targets = []
            for idx, (path, _) in enumerate(self.samples):
                self.samples[idx] = (path, self.class_to_idx[path.split("/")[-3]])
                self.targets.append(self.class_to_idx[path.split("/")[-3]])
        else:
            super().__init__(
                self.path + "val",
                transform=transform,
                target_transform=target_transform,
            )
            self.classes = []
            with open(self.path + "wnids.txt", "r", encoding="utf-8") as f:
                for line_id in f.readlines():
                    self.classes.append(line_id.split("\n")[0])
            self.class_to_idx = {clss: idx for idx, clss in enumerate(self.classes)}
            self.targets = []
            with open(self.path + "val/val_annotations.txt", "r", encoding="utf-8") as f:
                file_to_idx = {
                    line.split("\t")[0]: self.class_to_idx[line.split("\t")[1]]
                    for line in f.readlines()
                }
                for idx, (path, _) in enumerate(self.samples):
                    self.samples[idx] = (path, file_to_idx[path.split("/")[-1]])
                    self.targets.append(file_to_idx[path.split("/")[-1]])
