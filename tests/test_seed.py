import torch

from training.seed import seed_everything

seed_everything(42)

x = torch.randn(5)

seed_everything(42)

y = torch.randn(5)

print(torch.equal(x, y))