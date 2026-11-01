import torch

from gnn.graph_builder import GraphBuilder


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

builder = GraphBuilder(
    k=8,
).to(device)

x = torch.randn(
    2,
    196,
    384,
    device=device,
)

edge_indices = builder(x)

print()

for i, edge_index in enumerate(edge_indices):

    print(f"Graph {i}")

    print("edge_index:", edge_index.shape)

    print("dtype     :", edge_index.dtype)

    print("device    :", edge_index.device)

    print()