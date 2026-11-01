import torch
import torch.nn as nn

from torch_geometric.nn import SAGEConv


class DynamicGNN(nn.Module):
    """
    Dynamic Graph Neural Network.

    Input:
        x : (B, N, C)

        edge_indices :
            List[edge_index]

    Output:
        (B, N, C)
    """

    def __init__(
        self,
        embed_dim=384,
    ):
        super().__init__()

        self.conv1 = SAGEConv(
            embed_dim,
            embed_dim,
        )

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = SAGEConv(
            embed_dim,
            embed_dim,
        )

    def forward(
        self,
        x,
        edge_indices,
    ):

        outputs = []

        for b in range(x.size(0)):

            y = self.conv1(
                x[b],
                edge_indices[b],
            )

            y = self.relu(y)

            y = self.conv2(
                y,
                edge_indices[b],
            )

            outputs.append(y)

        return torch.stack(
            outputs,
            dim=0,
        )