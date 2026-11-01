import torch
import torch.nn as nn


class DynamicGNNLayer(nn.Module):
    """
    Dynamic k-NN Graph Neural Network Layer.

    Each node dynamically finds its k nearest neighbours
    in the embedding space and aggregates edge features.
    """

    def __init__(
        self,
        dim,
        k=6,
        linear_layer=nn.Linear,
    ):
        super().__init__()

        self.dim = dim
        self.k = k

        self.edge_mlp = nn.Sequential(

            linear_layer(
                dim * 2,
                dim,
            ),

            nn.GELU(),

            linear_layer(
                dim,
                dim,
            ),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        """
        Parameters
        ----------
        x : Tensor
            Shape:
                (B, N, D)

        Returns
        -------
        Tensor
            Shape:
                (B, N, D)
        """

        B, N, D = x.shape

        # --------------------------
        # Pairwise distances
        # --------------------------

        dist = torch.cdist(
            x,
            x,
        )

        # --------------------------
        # k-NN
        # --------------------------

        knn_idx = dist.topk(
            k=self.k,
            largest=False,
        ).indices

        idx_base = (
            torch.arange(
                B,
                device=x.device,
            )
            .view(-1, 1, 1)
            * N
        )

        knn_idx = knn_idx + idx_base

        neighbors = (
            x.reshape(B * N, D)[
                knn_idx.reshape(-1)
            ]
            .reshape(B, N, self.k, D)
        )

        # --------------------------
        # Edge Features
        # --------------------------

        x_expanded = (
            x.unsqueeze(2)
            .expand(
                -1,
                -1,
                self.k,
                -1,
            )
        )

        edge_feat = torch.cat(

            [
                x_expanded,
                neighbors - x_expanded,
            ],

            dim=-1,
        )

        out = self.edge_mlp(
            edge_feat
        )

        out = out.max(
            dim=2
        )[0]

        return self.norm(
            x + out
        )