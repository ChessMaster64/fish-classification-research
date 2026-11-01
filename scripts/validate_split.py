"""Validate video-level split counts and leakage for the configured dataset."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.dataloader import get_video_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    image_folder = datasets.ImageFolder(args.dataset)
    video_groups = defaultdict(list)
    for index, (path, _) in enumerate(image_folder.samples):
        video_groups[get_video_id(path)].append(index)
    for video_id in video_groups:
        video_groups[video_id] = video_groups[video_id][::3]

    video_ids = list(video_groups)
    train_videos, temporary_videos = train_test_split(
        video_ids, test_size=0.2, random_state=42
    )
    validation_videos, test_videos = train_test_split(
        temporary_videos, test_size=0.5, random_state=42
    )
    partitions = {
        "train": train_videos,
        "validation": validation_videos,
        "test": test_videos,
    }
    counts = {
        name: sum(len(video_groups[video]) for video in videos)
        for name, videos in partitions.items()
    }
    overlaps = {
        f"{left}_{right}": bool(set(partitions[left]) & set(partitions[right]))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    result = {
        "classes": len(image_folder.classes),
        "videos": len(video_ids),
        "sampled_images": sum(counts.values()),
        "counts": counts,
        "video_overlaps": overlaps,
        "seed": 42,
        "frame_stride": 3,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
