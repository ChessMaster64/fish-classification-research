import torch.nn as nn

from models.transformer_encoder import TransformerEncoder
from gnn.graph_builder import GraphBuilder
from gnn.dynamic_gnn import DynamicGNN


class HybridEncoder(nn.Module):
    """
    Hybrid Transformer-GNN Encoder

    Architecture:

    Transformer × 2
            ↓
    Dynamic GNN
            ↓
    Transformer × 2
            ↓
    Dynamic GNN
            ↓
    Transformer × 2
    """

    def __init__(
        self,
        embed_dim=384,
        num_heads=6,
        mlp_ratio=4,
        dropout=0.1,
        k=8,
    ):
        super().__init__()

        self.transformer1 = TransformerEncoder(
            depth=2,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.graph1 = GraphBuilder(k=k)

        self.gnn1 = DynamicGNN(
            embed_dim=embed_dim,
        )

        self.transformer2 = TransformerEncoder(
            depth=2,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.graph2 = GraphBuilder(k=k)

        self.gnn2 = DynamicGNN(
            embed_dim=embed_dim,
        )

        self.transformer3 = TransformerEncoder(
            depth=2,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, x):

        x = self.transformer1(x)

        edge_index = self.graph1(x)

        x = x + self.gnn1(
            x,
            edge_index,
        )

        x = self.transformer2(x)

        edge_index = self.graph2(x)

        x = x + self.gnn2(
            x,
            edge_index,
        )

        x = self.transformer3(x)

        return x