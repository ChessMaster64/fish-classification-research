import torch

from compression.compress_transformer_encoder import compress_transformer_encoder
from compression.config import BlockTTConfig
from models.transformer_encoder import TransformerEncoder

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = TransformerEncoder().to(device)

config = BlockTTConfig(
    max_rank=8,
    row_blocks=2,
    col_blocks=2,
)

print("\n===== BEFORE =====")

print(type(model.layers[0].attention.q_proj))
print(type(model.layers[0].ffn.fc1))

model = compress_transformer_encoder(
    model,
    config,
)

x = torch.randn(
    2,
    196,
    384,
    device=device,
)

with torch.no_grad():
    y = model(x)

print("\nForward Pass")
print("Input :", x.shape)
print("Output:", y.shape)

print("\n===== AFTER =====")

print(type(model.layers[0].attention.q_proj))
print(type(model.layers[0].ffn.fc1))