import torch

from models.hybrid_encoder import HybridEncoder


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = HybridEncoder().to(device)

x = torch.randn(
    2,
    196,
    384,
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