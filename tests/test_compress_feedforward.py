import torch

from compression.compress_feedforward import compress_feedforward
from compression.config import BlockTTConfig
from models.feedforward import FeedForward

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = FeedForward().to(device)

print("\n===== BEFORE =====")
print(type(model.fc1))
print(type(model.fc2))

config = BlockTTConfig(
    max_rank=8,
    row_blocks=2,
    col_blocks=2,
)

model = compress_feedforward(
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
print(type(model.fc1))
print(type(model.fc2))