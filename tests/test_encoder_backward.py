import torch

from models.resnet50_encoder import ResNet50Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet50Encoder().to(device)
model.train()

x = torch.randn(2, 3, 224, 224).to(device)

out = model(x)

loss = out.mean()

loss.backward()

print("Gradient Check")

for name, p in model.named_parameters():
    if p.grad is not None:
        print(name, p.grad.norm().item())