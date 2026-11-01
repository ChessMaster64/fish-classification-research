import torch.nn as nn

from backbones.resnet50_backbone import ResNet50Backbone
from .tokenizer import FeatureTokenizer
from .token_projection import TokenProjection


class ResNet50Encoder(nn.Module):

    def __init__(
        self,
        embed_dim=384,
        pretrained=True,
    ):

        super().__init__()

        self.backbone = ResNet50Backbone(
            pretrained=pretrained,
            output_layer=3,
        )

        self.tokenizer = FeatureTokenizer()

        self.projection = TokenProjection()

    def forward(self, x):

        x = self.backbone(x)

        x = self.tokenizer(x)

        x = self.projection(x)

        return x