import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """
    Learnable attention pooling over tokens.

    Input:
        (B, N, C)

    Output:
        (B, C)
    """

    def __init__(self, embed_dim=384):
        super().__init__()

        self.attention = nn.Linear(
            embed_dim,
            1,
        )

    def forward(self, x):

        weights = self.attention(x)

        weights = torch.softmax(
            weights,
            dim=1,
        )

        x = (weights * x).sum(dim=1)

        return x