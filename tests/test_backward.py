import torch
import config

from models.hybrid_model import DeepInterleavedResNet50ViTGNN
from compression.replace import replace_linear_layers

device = config.DEVICE

model = DeepInterleavedResNet50ViTGNN()

model = replace_linear_layers(
    model,
    config.BlockTTConfig(max_rank=16),
)

model.to(device)
model.train()

x = torch.randn(2, 3, 224, 224).to(device)
y = torch.randint(0, config.NUM_CLASSES, (2,)).to(device)

criterion = torch.nn.CrossEntropyLoss()

output = model(x)

loss = criterion(output, y)

loss.backward()

print("Loss:", loss.item())

num_grads = 0

for name, p in model.named_parameters():

    if p.grad is not None:

        num_grads += 1

        if "blocks" in name or "proj" in name:
            print(f"Gradient OK: {name}")

print(f"\nTotal parameters with gradients: {num_grads}")
print("\nBackward pass successful.")