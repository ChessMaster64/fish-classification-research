import torch

from models.dynamic_gnn import DynamicGNNLayer


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


layer = DynamicGNNLayer(

    dim=384,

    k=6,

).to(device)


x = torch.randn(

    8,

    196,

    384,

    device=device,

)


y = layer(x)

print()

print("Input :", x.shape)

print("Output:", y.shape)

assert x.shape == y.shape

print()

print("✓ Dynamic GNN Passed")