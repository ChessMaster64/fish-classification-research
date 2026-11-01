import torch

import config

from models.hybrid_model import DeepInterleavedResNet50ViTGNN


device = config.DEVICE

model = DeepInterleavedResNet50ViTGNN().to(device)

x = torch.randn(
    2,
    3,
    224,
    224,
    device=device,
)

with torch.no_grad():
    y = model(x)

print()

print("Input :", x.shape)

print("Output:", y.shape)

params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print()

print(f"Parameters: {params:,}")

assert y.shape == (2, config.NUM_CLASSES)

print()

print("✓ Hybrid Model Passed")