import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from evaluation.inference import predict


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


images = torch.randn(
    20,
    3,
    32,
    32,
)

labels = torch.randint(
    0,
    23,
    (20,),
)

loader = DataLoader(
    TensorDataset(
        images,
        labels,
    ),
    batch_size=5,
)

y_true, y_pred = predict(
    model,
    loader,
    device,
)

print("True labels :", y_true[:10])
print("Pred labels :", y_pred[:10])

print()

print("Total samples :", len(y_true))
print("Predictions   :", len(y_pred))