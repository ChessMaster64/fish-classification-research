import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embed_dim=384,
        num_heads=6,
        bias=True,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x):

        B, N, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            B,
            N,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            B,
            N,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            B,
            N,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        scores = (
            q @ k.transpose(-2, -1)
        ) / (self.head_dim ** 0.5)

        attention = F.softmax(
            scores,
            dim=-1,
        )

        out = attention @ v

        out = out.transpose(
            1,
            2,
        ).contiguous()

        out = out.view(
            B,
            N,
            C,
        )

        out = self.out_proj(out)

        return out