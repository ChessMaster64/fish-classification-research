import torch

from models.hybrid_model import HybridModel


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = HybridModel().to(device)

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

print()

print("Output dtype :", y.dtype)

print("Output device:", y.device)