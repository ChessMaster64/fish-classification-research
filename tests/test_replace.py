import torch

import config

from models.hybrid_model import DeepInterleavedResNet50ViTGNN
from compression.replace import replace_linear_layers


device = config.DEVICE

model = DeepInterleavedResNet50ViTGNN()

print("\nDense Parameters")

dense_params = sum(
    p.numel()
    for p in model.parameters()
)

print(dense_params)

cfg = config.BlockTTConfig(
    max_rank=16,
)

compressed = replace_linear_layers(
    model,
    cfg,
)

print("\nCompressed Parameters")

compressed_params = sum(
    p.numel()
    for p in compressed.parameters()
)

print(compressed_params)

x = torch.randn(
    2,
    3,
    224,
    224,
)

with torch.no_grad():

    y = compressed(x)

print()

print("Output Shape:", y.shape)