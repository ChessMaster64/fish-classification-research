import torch

from models.transformer_encoder_layer import TransformerEncoderLayer

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = TransformerEncoderLayer().to(device)

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