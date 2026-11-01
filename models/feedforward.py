import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(
        self,
        embed_dim=384,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()

        hidden_dim = embed_dim * mlp_ratio

        self.fc1 = nn.Linear(
            embed_dim,
            hidden_dim,
        )

        self.act = nn.GELU()

        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(
            hidden_dim,
            embed_dim,
        )

        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.drop2(x)

        return x