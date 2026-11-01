# =========================
# FILE: data_loader.py (UPGRADED)
# =========================

import torch
import numpy as np
import cv2
from collections import defaultdict

from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split


# =========================
# CLAHE
# =========================

class CLAHETransform:
    def __call__(self, img):
        img = np.array(img)

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(2.0, (8, 8))
        cl = clahe.apply(l)

        merged = cv2.merge((cl, a, b))
        img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

        return Image.fromarray(img)


# =========================
# UNDERWATER AUGMENT
# =========================

class UnderwaterAugment:
    def __call__(self, img):
        img = np.array(img).astype(np.float32)

        if np.random.rand() < 0.5:
            img[..., 0] *= np.random.uniform(0.7, 1.0)
            img[..., 1] *= np.random.uniform(0.9, 1.1)
            img[..., 2] *= np.random.uniform(1.0, 1.3)

        if np.random.rand() < 0.3:
            haze = np.random.uniform(0.6, 0.9)
            img = img * haze + (1 - haze) * 255

        return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


# =========================
# TRANSFORMS (FIXED ORDER)
# =========================

train_transform = transforms.Compose([
    CLAHETransform(),
    UnderwaterAugment(),

    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),

    transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
    transforms.GaussianBlur(3),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.4729, 0.6096, 0.5808],
        std=[0.2020, 0.2574, 0.2559]
    )
])


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    CLAHETransform(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4729, 0.6096, 0.5808],
        std=[0.2020, 0.2574, 0.2559]
    )
])


# =========================
# DATASET
# =========================

dataset_path = "fish4knowledge"
full_dataset = datasets.ImageFolder(dataset_path)

print("Total images:", len(full_dataset))


# =========================
# VIDEO SPLIT + SUBSAMPLING
# =========================

def get_video_id(path):
    return path.split("/")[-1].split("_")[1]

video_groups = defaultdict(list)

for idx, (path, label) in enumerate(full_dataset.samples):
    vid = get_video_id(path)
    video_groups[vid].append(idx)

# 🔥 Frame subsampling
for vid in video_groups:
    video_groups[vid] = video_groups[vid][::3]

video_ids = list(video_groups.keys())

train_vids, temp_vids = train_test_split(video_ids, test_size=0.2, random_state=42)
val_vids, test_vids = train_test_split(temp_vids, test_size=0.5, random_state=42)

train_idx = [i for v in train_vids for i in video_groups[v]]
val_idx   = [i for v in val_vids   for i in video_groups[v]]
test_idx  = [i for v in test_vids  for i in video_groups[v]]


# =========================
# DATASETS
# =========================

train_dataset = Subset(
    datasets.ImageFolder(dataset_path, transform=train_transform),
    train_idx
)

val_dataset = Subset(
    datasets.ImageFolder(dataset_path, transform=test_transform),
    val_idx
)

test_dataset = Subset(
    datasets.ImageFolder(dataset_path, transform=test_transform),
    test_idx
)


# =========================
# WEIGHTED SAMPLING (FIXED)
# =========================

train_labels = [full_dataset.samples[i][1] for i in train_idx]

class_counts = np.bincount(train_labels)
class_weights = 1. / np.sqrt(class_counts + 1e-6)

train_weights = [class_weights[label] for label in train_labels]

sampler = WeightedRandomSampler(
    weights=train_weights,
    num_samples=len(train_weights),
    replacement=True
)


# =========================
# LOADERS
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True

)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)