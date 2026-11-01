import torch
import torch.nn as nn


def get_loss(
    name: str = "cross_entropy",
    label_smoothing: float = 0.0,
):
    """
    Factory for training loss functions.

    Supported:
        - cross_entropy
    """

    name = name.lower()

    if name == "cross_entropy":
        return nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )

    raise ValueError(f"Unsupported loss: {name}")