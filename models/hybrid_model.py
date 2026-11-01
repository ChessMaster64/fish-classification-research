import torch
import torch.nn as nn
from torchvision import models

import config

from models.dynamic_gnn import DynamicGNNLayer
from models.transformer_encoder import TransformerEncoder


class DeepInterleavedResNet50ViTGNN(nn.Module):

    def __init__(
        self,
        num_classes=config.NUM_CLASSES,
        linear_layer=nn.Linear,
        gnn_layer=DynamicGNNLayer,
    ):
        super().__init__()

        self.embed_dim = config.EMBED_DIM

        # --------------------------------------------------
        # ResNet50
        # --------------------------------------------------

        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # --------------------------------------------------
        # Projection
        # --------------------------------------------------

        self.proj_in = linear_layer(
            1024,
            self.embed_dim,
        )

        self.pos_embed = nn.Parameter(
            torch.randn(
                1,
                196,
                self.embed_dim,
            )
        )

        # --------------------------------------------------
        # Transformers
        # --------------------------------------------------

        depths = config.TRANSFORMER_DEPTH

        self.transformer1 = TransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=config.NUM_HEADS,
            depth=depths[0],
            dropout=config.DROPOUT,
        )

        self.transformer2 = TransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=config.NUM_HEADS,
            depth=depths[1],
            dropout=config.DROPOUT,
        )

        self.transformer3 = TransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=config.NUM_HEADS,
            depth=depths[2],
            dropout=config.DROPOUT,
        )

        # --------------------------------------------------
        # GNN
        # --------------------------------------------------

        self.gnn1 = gnn_layer(
            dim=self.embed_dim,
            k=config.KNN_NEIGHBORS,
            linear_layer=linear_layer,
        )

        self.gnn2 = gnn_layer(
            dim=self.embed_dim,
            k=config.KNN_NEIGHBORS,
            linear_layer=linear_layer,
        )

        # --------------------------------------------------
        # Projection Out
        # --------------------------------------------------

        self.proj_out = linear_layer(
            self.embed_dim,
            1024,
        )

        # --------------------------------------------------
        # Head
        # --------------------------------------------------

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = linear_layer(
            2048,
            num_classes,
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)

        x = self.proj_in(x)

        x = x + self.pos_embed

        x = self.transformer1(x)

        identity = x
        x = self.gnn1(x)
        x = x + identity

        x = self.transformer2(x)

        identity = x
        x = self.gnn2(x)
        x = x + identity

        x = self.transformer3(x)

        x = self.proj_out(x)

        x = x.transpose(1, 2).reshape(
            B,
            1024,
            H,
            W,
        )

        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.head(x)

        return x