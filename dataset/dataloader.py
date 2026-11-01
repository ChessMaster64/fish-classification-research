from collections import defaultdict

import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from sklearn.model_selection import train_test_split

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import WeightedRandomSampler


# ==========================================================
# CLAHE
# ==========================================================

class CLAHETransform:

    def __call__(self, img):

        img = np.array(img)

        lab = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2LAB,
        )

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        cl = clahe.apply(l)

        img = cv2.merge((cl, a, b))

        img = cv2.cvtColor(
            img,
            cv2.COLOR_LAB2RGB,
        )

        return Image.fromarray(img)


# ==========================================================
# Underwater Augmentation
# ==========================================================

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

        img = np.clip(
            img,
            0,
            255,
        )

        return Image.fromarray(
            img.astype(np.uint8)
        )


# ==========================================================
# Transforms
# ==========================================================

def get_train_transform():

    return transforms.Compose([

        CLAHETransform(),

        UnderwaterAugment(),

        transforms.RandomResizedCrop(
            224,
            scale=(0.6, 1.0),
        ),

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(15),

        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05,
        ),

        transforms.GaussianBlur(3),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.4729, 0.6096, 0.5808],
            std=[0.2020, 0.2574, 0.2559],
        ),

    ])


def get_test_transform():

    return transforms.Compose([

        transforms.Resize((224, 224)),

        CLAHETransform(),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.4729, 0.6096, 0.5808],
            std=[0.2020, 0.2574, 0.2559],
        ),

    ])


# ==========================================================
# Video ID
# ==========================================================

def get_video_id(path):

    return path.split("/")[-1].split("_")[1]


# ==========================================================
# Main
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_DATASET = PROJECT_ROOT / "fish4knowledge"

def create_dataloaders(
    dataset_path=DEFAULT_DATASET,

    batch_size=32,

    frame_stride=3,

    train_ratio=0.8,

    val_ratio=0.1,

    test_ratio=0.1,

    num_workers=8,

    seed=42,

):

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    dataset_path = str(dataset_path)

    full_dataset = datasets.ImageFolder(
        dataset_path
    )

    video_groups = defaultdict(list)

    for idx, (path, label) in enumerate(full_dataset.samples):

        video_groups[
            get_video_id(path)
        ].append(idx)

    for vid in video_groups:

        video_groups[vid] = video_groups[vid][::frame_stride]

    video_ids = list(video_groups.keys())

    train_vids, temp_vids = train_test_split(

        video_ids,

        test_size=(1 - train_ratio),

        random_state=seed,

    )

    val_fraction = val_ratio / (val_ratio + test_ratio)

    val_vids, test_vids = train_test_split(

        temp_vids,

        train_size=val_fraction,

        random_state=seed,

    )

    train_idx = [
        i
        for v in train_vids
        for i in video_groups[v]
    ]

    val_idx = [
        i
        for v in val_vids
        for i in video_groups[v]
    ]

    test_idx = [
        i
        for v in test_vids
        for i in video_groups[v]
    ]

    train_dataset = Subset(

        datasets.ImageFolder(
            dataset_path,
            transform=get_train_transform(),
        ),

        train_idx,

    )

    val_dataset = Subset(

        datasets.ImageFolder(
            dataset_path,
            transform=get_test_transform(),
        ),

        val_idx,

    )

    test_dataset = Subset(

        datasets.ImageFolder(
            dataset_path,
            transform=get_test_transform(),
        ),

        test_idx,

    )

    train_labels = [

        full_dataset.samples[i][1]

        for i in train_idx

    ]

    class_counts = np.bincount(train_labels)

    class_weights = 1.0 / np.sqrt(
        class_counts + 1e-6
    )

    sample_weights = [

        class_weights[label]

        for label in train_labels

    ]

    sampler = WeightedRandomSampler(

        sample_weights,

        len(sample_weights),

        replacement=True,

    )

    persistent = num_workers > 0

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        sampler=sampler,

        num_workers=num_workers,

        pin_memory=True,

        persistent_workers=persistent,

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=True,

        persistent_workers=persistent,

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=True,

        persistent_workers=persistent,

    )

    return (

        train_loader,

        val_loader,

        test_loader,

        full_dataset.classes,

    )