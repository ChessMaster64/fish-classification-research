import torch.nn as nn


class TransformerEncoder(nn.Module):
    """
    Wrapper around PyTorch TransformerEncoder.

    Parameters
    ----------
    embed_dim : int
    num_heads : int
    depth : int
    mlp_ratio : float
    dropout : float
    """

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        depth=2,
        mlp_ratio=2.0,
        dropout=0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

    def forward(self, x):

        return self.encoder(x)