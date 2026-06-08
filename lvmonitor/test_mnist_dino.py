from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from transformers import AutoModel, AutoImageProcessor
import pandas as pd

def load_mnist(root: str | Path = "./data"):
    transform = transforms.ToTensor()
    return datasets.MNIST(root=str(root), train=True, download=True, transform=transform)

def create_split_mnist_tasks(dataset):
    targets = dataset.targets.numpy()
    task_indices = []
    for i in range(0, 10, 2):
        idx = np.where((targets == i) | (targets == i+1))[0]
        task_indices.append(idx)
    task_datasets = [Subset(dataset, idx) for idx in task_indices]
    return task_datasets

@torch.inference_mode()
def extract_features(model, processor, img_tensor, device):
    img_np = img_tensor.squeeze(0).cpu().numpy()
    gray = (img_np * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    feat = out.last_hidden_state[:, 0].squeeze(0)
    return feat

def compute_mean_cosine(batch_feat, buffer_feat):
    if buffer_feat is None or len(buffer_feat) == 0:
        return 0.0
    sim = F.cosine_similarity(batch_feat.unsqueeze(1), buffer_feat.unsqueeze(0), dim=-1)
    return sim.mean().item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./dinov3-vits16-pretrain-lvd1689m"

    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()

    full_dataset = load_mnist(root="./data/")
    tasks = create_split_mnist_tasks(full_dataset)

    batch_size = 1024
    buffer_size = 10000
    feature_buffer = None

    task_order = [0, 1, 2, 3, 4]
    task_names = ["Task 0:0,1", "Task 1:2,3", "Task 2:4,5", "Task 3:6,7", "Task 4:8,9"]
    records = []

    for task_id in task_order:
        loader = DataLoader(tasks[task_id], batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"\n=== Processing {task_names[task_id]} ===")

        for batch_idx, (images, _) in enumerate(loader):
            batch_feats = []
            for img in images:
                feat = extract_features(model, processor, img, device)
                batch_feats.append(feat)
            batch_feats = torch.stack(batch_feats)

            mean_sim = compute_mean_cosine(batch_feats, feature_buffer)
            task_name = task_names[task_id]
            print(f"{task_names[task_id]} | Batch {batch_idx} | Mean Cosine Sim = {mean_sim:.4f}")

            records.append({
                "task": task_name,
                "batch_idx": batch_idx,
                "mean_cosine_sim": mean_sim
            })

            if feature_buffer is None:
                feature_buffer = batch_feats.clone()
            else:
                feature_buffer = torch.cat([feature_buffer, batch_feats], dim=0)

            if len(feature_buffer) > buffer_size:
                feature_buffer = feature_buffer[-buffer_size:]
    
    df = pd.DataFrame(records)
    df.to_csv("mnist_buffer_cosine.csv", index=False)
    print("\nData saved to mnist_buffer_cosine.csv")
    print(df)

if __name__ == "__main__":
    main()