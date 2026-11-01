"""Construct dense/BlockTT models and write parameter statistics."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compression.replace import replace_linear_layers
from config import BlockTTConfig
from models.hybrid_model import DeepInterleavedResNet50ViTGNN


def count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dense = DeepInterleavedResNet50ViTGNN()
    blocktt = replace_linear_layers(dense, BlockTTConfig(max_rank=16), verbose=False)
    dense_parameters = count_trainable_parameters(dense)
    blocktt_parameters = count_trainable_parameters(blocktt)
    result = {
        "dense_trainable_parameters": dense_parameters,
        "blocktt_trainable_parameters": blocktt_parameters,
        "parameter_reduction_percent": (dense_parameters - blocktt_parameters) / dense_parameters * 100,
        "compression_ratio": dense_parameters / blocktt_parameters,
        "max_rank": 16,
        "row_blocks": 2,
        "col_blocks": 2,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
