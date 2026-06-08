import glob
import os
from shutil import move, rmtree

import torch
import tqdm
from torchvision import datasets


class CORe50(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.url = "http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip"
        self.filename = "core50_128x128.zip"
        self.fpath = os.path.join(self.root, "core50_128x128")

        self.train_session_list = ["s1", "s2", "s4", "s5", "s6", "s8", "s9", "s11"]
        self.test_session_list = ["s3", "s7", "s10"]
        self.label = [f"o{i}" for i in range(1, 51)]
        self.targets = []

        if not os.path.exists(self.fpath + "/train") and not os.path.exists(self.fpath + "/test"):
            self.split()

        if self.train:
            fpath = self.fpath + "/train"
            self.dataset = [
                datasets.ImageFolder(f"{fpath}/{s}", transform=transform)
                for s in self.train_session_list
            ]
            for d in self.dataset:
                self.targets.extend(d.targets)
        else:
            fpath = self.fpath + "/test"
            self.dataset = datasets.ImageFolder(fpath, transform=transform)
            self.targets = self.dataset.targets

        self.classes = [str(i) for i in range(50)]

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return len(self.dataset)

    def split(self):
        train_folder = self.fpath + "/train"
        test_folder = self.fpath + "/test"

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        for s in tqdm.tqdm(self.train_session_list, desc="Preprocessing"):
            src = os.path.join(self.fpath, s)
            if os.path.exists(os.path.join(train_folder, s)):
                continue
            move(src, train_folder)

        for s in tqdm.tqdm(self.test_session_list, desc="Preprocessing"):
            for label in self.label:
                dst = os.path.join(test_folder, label)
                if not os.path.exists(dst):
                    os.mkdir(os.path.join(test_folder, label))

                for src in glob.glob(os.path.join(self.fpath, s, label, "*.png")):
                    move(src, dst)
            rmtree(os.path.join(self.fpath, s))
