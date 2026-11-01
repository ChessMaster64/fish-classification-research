import torch.nn as nn

from models.multihead_attention import MultiHeadAttention
from models.feedforward import FeedForward
from models.drop_path import DropPath


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        embed_dim=384,
        num_heads=6,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = FeedForward(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.drop_path = DropPath(
            drop_prob=0.1,
        )

    def forward(self, x):

        x = x + self.drop_path(
            self.attention(
                self.norm1(x)
            )
        )

        x = x + self.drop_path(
            self.ffn(
                self.norm2(x)
            )
        )

        return x