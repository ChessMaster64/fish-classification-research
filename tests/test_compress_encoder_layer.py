import torch

from compression.compress_encoder_layer import compress_encoder_layer
from compression.config import BlockTTConfig
from models.transformer_encoder_layer import TransformerEncoderLayer


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = TransformerEncoderLayer().to(device)

print("\n===== BEFORE =====")

print("\nAttention")
print(type(model.attention.q_proj))
print(type(model.attention.k_proj))
print(type(model.attention.v_proj))
print(type(model.attention.out_proj))

print("\nFeedForward")
print(type(model.ffn.fc1))
print(type(model.ffn.fc2))

config = BlockTTConfig(
    max_rank=8,
    row_blocks=2,
    col_blocks=2,
)

model = compress_encoder_layer(
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

print("\nAttention")
print(type(model.attention.q_proj))
print(type(model.attention.k_proj))
print(type(model.attention.v_proj))
print(type(model.attention.out_proj))

print("\nFeedForward")
print(type(model.ffn.fc1))
print(type(model.ffn.fc2))