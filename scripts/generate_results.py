"""Generate comparison tables and figures from evaluation JSON records."""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "comparison.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "model",
                "accuracy",
                "precision",
                "recall",
                "weighted_f1",
                "specificity",
                "mcc",
                "trainable_parameters",
                "parameter_reduction_percent",
                "compression_ratio",
            )
        )
        for record in records:
            metrics = record["metrics"]
            writer.writerow(
                (
                    record["model"],
                    metrics["accuracy"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                    metrics["specificity"],
                    metrics["mcc"],
                    record["trainable_parameters"],
                    record.get("parameter_reduction_percent"),
                    record.get("compression_ratio"),
                )
            )

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to generate result figures") from error

    names = [record["model"] for record in records]
    for metric, filename, ylabel in (
        ("mcc", "mcc_comparison.png", "MCC"),
        ("f1", "weighted_f1_comparison.png", "Weighted F1 (%)"),
    ):
        values = [record["metrics"][metric] for record in records]
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(names, values)
        axis.set_ylabel(ylabel)
        axis.set_title(ylabel + " comparison")
        figure.tight_layout()
        figure.savefig(args.output_dir / filename, dpi=300)
        plt.close(figure)


if __name__ == "__main__":
    main()
