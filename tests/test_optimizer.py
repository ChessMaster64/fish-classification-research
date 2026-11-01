import torch.nn as nn

from training.optimizer import create_optimizer
from training.optimizer import create_scheduler


model = nn.Linear(128, 23)

optimizer = create_optimizer(model)

scheduler = create_scheduler(
    optimizer,
    epochs=100,
)

print(optimizer)

print(scheduler)