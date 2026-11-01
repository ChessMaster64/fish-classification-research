import torch
import torch.nn as nn


class GraphBuilder(nn.Module):
    """
    Builds a dynamic k-NN graph using pure PyTorch.

    Input:
        x : (B, N, C)

    Output:
        List[edge_index]
    """

    def __init__(
        self,
        k=8,
        loop=False,
    ):
        super().__init__()

        self.k = k
        self.loop = loop

    def forward(self, x):

        B, N, C = x.shape

        edge_indices = []

        for b in range(B):

            features = x[b]                      # (N,C)

            distance = torch.cdist(
                features,
                features,
                p=2,
            )

            if not self.loop:
                distance.fill_diagonal_(float("inf"))

            _, neighbors = torch.topk(
                distance,
                k=self.k,
                largest=False,
            )

            source = torch.arange(
                N,
                device=x.device,
            ).repeat_interleave(self.k)

            target = neighbors.reshape(-1)

            edge_index = torch.stack(
                [source, target],
                dim=0,
            )

            edge_indices.append(edge_index)

        return edge_indices