import torch

from training.losses import get_loss


criterion = get_loss(
    label_smoothing=0.1
)

logits = torch.randn(8, 23)

targets = torch.randint(
    0,
    23,
    (8,)
)

loss = criterion(
    logits,
    targets,
)

print("Loss:", loss.item())