"""Evaluate a dense or BlockTT checkpoint on the held-out test split."""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compression.replace import replace_linear_layers
from config import BlockTTConfig
from dataset.dataloader import create_dataloaders
from evaluation.metrics import paper_classification_metrics
from models.hybrid_model import DeepInterleavedResNet50ViTGNN
from training.checkpoint_compat import load_model_checkpoint


def count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_model(kind):
    model = DeepInterleavedResNet50ViTGNN()
    if kind == "blocktt":
        model = replace_linear_layers(model, BlockTTConfig(max_rank=16), verbose=False)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("dense", "blocktt"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = build_model(args.model).to(device)
    checkpoint_metadata = load_model_checkpoint(model, args.checkpoint, device=device)
    _, _, test_loader, class_names = create_dataloaders(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu().tolist()
            y_true.extend(labels.tolist())
            y_pred.extend(predictions)

    labels = list(range(len(class_names)))
    metrics = paper_classification_metrics(y_true, y_pred, labels=labels)
    parameters = count_trainable_parameters(model)
    report = {
        "model": args.model,
        "checkpoint": checkpoint_metadata,
        "dataset": str(args.dataset),
        "classes": class_names,
        "samples": len(y_true),
        "metrics": metrics,
        "trainable_parameters": parameters,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        ).tolist(),
    }
    if args.model == "blocktt":
        dense_model = build_model("dense")
        dense_parameters = count_trainable_parameters(dense_model)
        report["dense_trainable_parameters"] = dense_parameters
        report["parameter_reduction_percent"] = (
            (dense_parameters - parameters) / dense_parameters * 100
        )
        report["compression_ratio"] = dense_parameters / parameters

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    predictions_path = args.output.with_name(args.output.stem + "_predictions.csv")
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("true_label", "predicted_label"))
        writer.writerows(zip(y_true, y_pred))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
