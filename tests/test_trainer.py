import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from training.losses import get_loss
from training.optimizer import create_optimizer

from training.trainer import train_one_epoch
from training.trainer import validate


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(
        3 * 32 * 32,
        23,
    ),
).to(device)

criterion = get_loss()

optimizer = create_optimizer(model)

images = torch.randn(
    128,
    3,
    32,
    32,
)

labels = torch.randint(
    0,
    23,
    (128,),
)

loader = DataLoader(
    TensorDataset(
        images,
        labels,
    ),
    batch_size=16,
)

scaler = (
    torch.amp.GradScaler("cuda")
    if device.type == "cuda"
    else None
)

train_metrics = train_one_epoch(
    model=model,
    loader=loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    scaler=scaler,
)

val_metrics = validate(
    model=model,
    loader=loader,
    criterion=criterion,
    device=device,
)

print()

print("Train:", train_metrics)

print("Validation:", val_metrics)