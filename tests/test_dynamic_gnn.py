import torch

from gnn.graph_builder import GraphBuilder
from gnn.dynamic_gnn import DynamicGNN


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

builder = GraphBuilder(
    k=8,
).to(device)

gnn = DynamicGNN().to(device)

x = torch.randn(
    2,
    196,
    384,
    device=device,
)

edge_indices = builder(x)

with torch.no_grad():

    y = gnn(
        x,
        edge_indices,
    )

print()

print("Input :", x.shape)

print("Output:", y.shape)

print()

print("Output dtype :", y.dtype)

print("Output device:", y.device)