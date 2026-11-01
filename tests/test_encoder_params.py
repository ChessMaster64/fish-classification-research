import torch

from models.resnet50_encoder import ResNet50Encoder

model = ResNet50Encoder()

params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"Trainable Parameters: {params:,}")