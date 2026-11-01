import torch
import torch.nn as nn
from torchvision import models


class ResNet50Backbone(nn.Module):
    """
    ResNet50 feature extractor.

    Output:
        Feature map after Layer3
        Shape: (B, 1024, 14, 14)
    """

    def __init__(
        self,
        pretrained=True,
        output_layer=3,
    ):
        super().__init__()

        self.output_layer = output_layer

        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None

        resnet = models.resnet50(weights=weights)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        if self.output_layer == 3:
            return x

        x = self.layer4(x)

        return x