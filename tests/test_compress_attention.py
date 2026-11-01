import torch

from models.multihead_attention import MultiHeadAttention

from compression.compress_attention import compress_attention
from compression.config import BlockTTConfig


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = MultiHeadAttention().to(device)

print("\n===== BEFORE =====")
print(type(model.q_proj))
print(type(model.k_proj))
print(type(model.v_proj))
print(type(model.out_proj))

config = BlockTTConfig(
    max_rank=8,
    row_blocks=2,
    col_blocks=2,
)

model = compress_attention(
    model,
    config,
)
x = torch.randn(
    2,
    196,
    384,
).to(device)

with torch.no_grad():
    y = model(x)

print("\nForward Pass")
print("Input :", x.shape)
print("Output:", y.shape)
print("\n===== AFTER =====")
print(type(model.q_proj))
print(type(model.k_proj))
print(type(model.v_proj))
print(type(model.out_proj))