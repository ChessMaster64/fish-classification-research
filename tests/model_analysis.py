"""
tests/model_analysis.py

Model analysis utility for the BlockTT compressed hybrid model.

Reports:
1. Total trainable parameters
2. Parameter distribution by major component
3. Convolution breakdown
4. Linear / BlockTT layer inventory
"""

import torch
import torch.nn as nn

import config

from models.hybrid_model import (
    DeepInterleavedResNet50ViTGNN,
)

from compression.replace import (
    replace_linear_layers,
)

from layers.block_tt_linear import BlockTTLinear


# ============================================================
# Utility Functions
# ============================================================

def count_params(module):

    return sum(
        p.numel()
        for p in module.parameters()
        if p.requires_grad
    )


def format_number(x):

    return f"{x:,}"


def percentage(value, total):

    return 100 * value / total


# ============================================================
# Build Model
# ============================================================

print("\nLoading BlockTT model...\n")

model = DeepInterleavedResNet50ViTGNN()

model = replace_linear_layers(
    model,
    config.BlockTTConfig(max_rank=16),
    verbose=False,
)

model.eval()

TOTAL_PARAMS = count_params(model)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("=" * 75)
print("MODEL SUMMARY")
print("=" * 75)

print(
    f"Total Trainable Parameters : "
    f"{format_number(TOTAL_PARAMS)}"
)


# ============================================================
# PARAMETER BREAKDOWN
# ============================================================

print("\n")
print("=" * 75)
print("PARAMETER BREAKDOWN")
print("=" * 75)

components = {
    "Stem": model.stem,
    "Layer1": model.layer1,
    "Layer2": model.layer2,
    "Layer3": model.layer3,
    "Layer4": model.layer4,
    "Transformer 1": model.transformer1,
    "Transformer 2": model.transformer2,
    "Transformer 3": model.transformer3,
    "GNN 1": model.gnn1,
    "GNN 2": model.gnn2,
    "Projection In": model.proj_in,
    "Projection Out": model.proj_out,
    "Classifier": model.head,
}

print(
    f"{'Component':<25}"
    f"{'Parameters':>18}"
    f"{'Percent':>12}"
)

print("-" * 60)

for name, module in components.items():

    params = count_params(module)

    print(
        f"{name:<25}"
        f"{format_number(params):>18}"
        f"{percentage(params, TOTAL_PARAMS):>11.2f}%"
    )

# ============================================================
# CONVOLUTION BREAKDOWN
# ============================================================

print("\n")
print("=" * 75)
print("CONVOLUTION BREAKDOWN")
print("=" * 75)

conv_stats = {

    "7x7": {
        "count": 0,
        "params": 0,
    },

    "3x3": {
        "count": 0,
        "params": 0,
    },

    "1x1": {
        "count": 0,
        "params": 0,
    },

    "Other": {
        "count": 0,
        "params": 0,
    },

}

downsample_count = 0
downsample_params = 0

for name, module in model.named_modules():

    if isinstance(module, nn.Conv2d):

        params = count_params(module)

        if module.kernel_size == (7, 7):

            key = "7x7"

        elif module.kernel_size == (3, 3):

            key = "3x3"

        elif module.kernel_size == (1, 1):

            key = "1x1"

        else:

            key = "Other"

        conv_stats[key]["count"] += 1
        conv_stats[key]["params"] += params

        if "downsample" in name:

            downsample_count += 1
            downsample_params += params

print(
    f"{'Kernel':<15}"
    f"{'Count':>10}"
    f"{'Parameters':>18}"
    f"{'Percent':>12}"
)

print("-" * 60)

for key in ["7x7", "3x3", "1x1", "Other"]:

    params = conv_stats[key]["params"]

    print(
        f"{key:<15}"
        f"{conv_stats[key]['count']:>10}"
        f"{format_number(params):>18}"
        f"{percentage(params, TOTAL_PARAMS):>11.2f}%"
    )

print("-" * 60)

print(
    f"{'Downsample':<15}"
    f"{downsample_count:>10}"
    f"{format_number(downsample_params):>18}"
    f"{percentage(downsample_params, TOTAL_PARAMS):>11.2f}%"
)

# ============================================================
# LINEAR LAYER ANALYSIS
# ============================================================

print("\n")
print("=" * 75)
print("LINEAR LAYER ANALYSIS")
print("=" * 75)

dense_count = 0
dense_params = 0

blocktt_count = 0
blocktt_params = 0

print("\nDense Linear Layers")
print("-" * 75)

for name, module in model.named_modules():

    if isinstance(module, nn.Linear):

        params = count_params(module)

        dense_count += 1
        dense_params += params

        print(
            f"{name:<55}"
            f"{format_number(params):>18}"
        )


print("\n")
print("=" * 75)
print("BLOCKTT LAYERS")
print("=" * 75)

for name, module in model.named_modules():

    if isinstance(module, BlockTTLinear):

        params = count_params(module)

        blocktt_count += 1
        blocktt_params += params

        print(
            f"{name:<55}"
            f"{format_number(params):>18}"
        )


# ============================================================
# BLOCKTT INTERNAL TT LAYERS
# ============================================================

print("\n")
print("=" * 75)
print("TT LAYERS INSIDE BLOCKTT")
print("=" * 75)

tt_linear_count = 0
tt_linear_params = 0

for name, module in model.named_modules():

    if module.__class__.__name__ == "TTLinear":

        params = count_params(module)

        tt_linear_count += 1
        tt_linear_params += params

        print(
            f"{name:<55}"
            f"{format_number(params):>18}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("SUMMARY")
print("=" * 75)

print(f"Total Parameters        : {format_number(TOTAL_PARAMS)}")
print()

print(f"Dense Linear Layers     : {dense_count}")
print(f"Dense Linear Parameters : {format_number(dense_params)}")
print()

print(f"BlockTT Layers          : {blocktt_count}")
print(f"BlockTT Parameters      : {format_number(blocktt_params)}")
print()

print(f"TTLinear Modules        : {tt_linear_count}")
print(f"TTLinear Parameters     : {format_number(tt_linear_params)}")

print("\nDone.")

print("\n")
print("=" * 75)
print("TOP CONVOLUTION LAYERS")
print("=" * 75)

conv_layers = []

for name, module in model.named_modules():

    if isinstance(module, nn.Conv2d):

        params = count_params(module)

        conv_layers.append(
            (
                params,
                name,
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
            )
        )

conv_layers.sort(reverse=True)

print(
    f"{'Layer':<45}"
    f"{'Parameters':>12}"
    f"{'Kernel':>12}"
    f"{'Channels':>18}"
)

print("-" * 90)

for params, name, cin, cout, kernel, stride in conv_layers[:20]:

    print(
        f"{name:<45}"
        f"{format_number(params):>12}"
        f"{str(kernel):>12}"
        f"{f'{cin}->{cout}':>18}"
    )