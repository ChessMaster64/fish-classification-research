import torch

from models.transformer import TransformerEncoder


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = TransformerEncoder(
    embed_dim=384,
    num_heads=8,
    depth=3,
).to(device)


x = torch.randn(
    4,
    196,
    384,
    device=device,
)

y = model(x)

print()

print("Input :", x.shape)

print("Output:", y.shape)

assert x.shape == y.shape

print()

print("✓ Transformer Passed")