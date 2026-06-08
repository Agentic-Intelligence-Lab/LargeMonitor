from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import pandas as pd

NUM_TASKS = 10
NUM_CLASSES = 100


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x.flatten(1)


def load_cifar100(root: str | Path = "./data"):
    transform = transforms.ToTensor()
    return datasets.CIFAR100(root=str(root), train=True, download=True, transform=transform)


def create_split_cifar100_tasks(dataset, num_tasks: int = NUM_TASKS):
    """Split-CIFAR100: 10 classes per task, same as Disjoint/datasets.py."""
    assert NUM_CLASSES % num_tasks == 0
    classes_per_task = NUM_CLASSES // num_tasks
    targets = np.array(dataset.targets)
    task_indices = []
    for t in range(num_tasks):
        lo = t * classes_per_task
        hi = lo + classes_per_task
        idx = np.where((targets >= lo) & (targets < hi))[0]
        task_indices.append(idx)
    return [Subset(dataset, idx) for idx in task_indices]


def task_label(task_id: int, class_names: list[str], num_tasks: int = NUM_TASKS) -> str:
    classes_per_task = NUM_CLASSES // num_tasks
    lo = task_id * classes_per_task
    hi = lo + classes_per_task
    names = ", ".join(class_names[lo:hi])
    return f"Task {task_id}: classes {lo}-{hi - 1} ({names})"


@torch.no_grad()
def extract_features(encoder, img_tensor, device):
    x = img_tensor.unsqueeze(0).to(device)
    return encoder(x).squeeze(0).cpu()


def compute_mean_cosine(batch_feat, buffer_feat):
    if buffer_feat is None or len(buffer_feat) == 0:
        return 0.0
    sim = F.cosine_similarity(batch_feat.unsqueeze(1), buffer_feat.unsqueeze(0), dim=-1)
    return sim.mean().item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SimpleCNN(in_channels=3).to(device).eval()

    full_dataset = load_cifar100(root="./data/")
    tasks = create_split_cifar100_tasks(full_dataset)
    task_names = [task_label(i, full_dataset.classes) for i in range(NUM_TASKS)]

    batch_size = 1024
    buffer_size = 10000
    feature_buffer = None
    records = []

    for task_id in range(NUM_TASKS):
        loader = DataLoader(tasks[task_id], batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"\n=== Processing {task_names[task_id]} ===")

        for batch_idx, (images, _) in enumerate(loader):
            feats = []
            for img in images:
                feats.append(extract_features(encoder, img, device))
            feats = torch.stack(feats)

            mean_sim = compute_mean_cosine(feats, feature_buffer)
            print(f"Batch {batch_idx:3d} | Mean Cosine Sim = {mean_sim:.4f}")
            records.append({
                "task_id": task_id,
                "task": task_names[task_id],
                "batch_idx": batch_idx,
                "mean_cosine_sim": mean_sim,
            })

            if feature_buffer is None:
                feature_buffer = feats.clone()
            else:
                feature_buffer = torch.cat([feature_buffer, feats], dim=0)
            if len(feature_buffer) > buffer_size:
                feature_buffer = feature_buffer[-buffer_size:]

    out_path = f"cifar100_cnn_kaiming_buffer_{buffer_size}_cosine.csv"
    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(df)


if __name__ == "__main__":
    main()
