import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """
    Converts CNN feature map into transformer tokens.

    Input:
        (B,1024,14,14)

    Output:
        (B,196,1024)
    """

    def forward(self, x):

        B, C, H, W = x.shape

        x = x.flatten(2)

        x = x.transpose(1, 2)

        return x