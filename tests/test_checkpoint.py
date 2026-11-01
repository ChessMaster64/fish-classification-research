import os

import torch.nn as nn

from training.optimizer import create_optimizer
from training.optimizer import create_scheduler

from training.checkpoint import save_checkpoint
from training.checkpoint import load_checkpoint


model = nn.Linear(10, 5)

optimizer = create_optimizer(model)

scheduler = create_scheduler(
    optimizer,
    epochs=10,
)

save_checkpoint(
    "checkpoints/test_checkpoint.pth",
    model,
    optimizer,
    scheduler,
    epoch=7,
    best_accuracy=94.6,
)

epoch, best_acc = load_checkpoint(
    "checkpoints/test_checkpoint.pth",
    model,
    optimizer,
    scheduler,
)

print("Epoch:", epoch)
print("Best Accuracy:", best_acc)

os.remove("checkpoints/test_checkpoint.pth")