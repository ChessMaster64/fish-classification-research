import torch.nn as nn

from models.transformer_encoder_layer import TransformerEncoderLayer


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        depth=6,
        embed_dim=384,
        num_heads=6,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x