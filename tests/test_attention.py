import torch

from models.multihead_attention import MultiHeadAttention

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = MultiHeadAttention().to(device)

x = torch.randn(
    2,
    196,
    384,
).to(device)

with torch.no_grad():
    y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)

params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"Parameters: {params:,}")