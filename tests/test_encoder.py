print(">>> TEST FILE STARTED")

import torch
from research.models.resnet50_encoder import ResNet50Encoder

print(">>> Imports successful")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = ResNet50Encoder().to(device)
print(">>> Model created")

model.eval()

x = torch.randn(2, 3, 224, 224).to(device)

with torch.no_grad():
    y = model(x)

print("Output shape:", y.shape)

print(">>> TEST FINISHED")