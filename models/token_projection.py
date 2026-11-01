import torch.nn as nn


class TokenProjection(nn.Module):

    def __init__(self,
                 in_dim=1024,
                 embed_dim=384):

        super().__init__()

        self.proj = nn.Linear(
            in_dim,
            embed_dim
        )

    def forward(self, x):

        return self.proj(x)