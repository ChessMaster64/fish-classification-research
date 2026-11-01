"""
utils.py

Common utility functions used throughout the tensor compression framework.
"""

from math import prod
from typing import List, Tuple

import torch


def count_parameters(model) -> int:
    """
    Count trainable parameters in a PyTorch model.
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def tensor_numel(shape: Tuple[int, ...]) -> int:
    """
    Number of elements represented by a tensor shape.
    """

    return prod(shape)


def compression_ratio(
    original_params: int,
    compressed_params: int,
) -> float:
    """
    Compute compression ratio.

    Example

    Original = 1,000,000

    Compressed = 100,000

    Ratio = 10x
    """

    if compressed_params == 0:
        raise ValueError("Compressed parameters cannot be zero.")

    return original_params / compressed_params


def reconstruction_error(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> float:
    """
    Frobenius norm reconstruction error.
    """

    return torch.norm(original - reconstructed).item()


def relative_error(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> float:
    """
    Relative reconstruction error.

    ||A-B|| / ||A||
    """

    numerator = torch.norm(original - reconstructed)

    denominator = torch.norm(original)

    return (numerator / denominator).item()


def memory_size(
    tensor: torch.Tensor,
) -> float:
    """
    Memory usage in MB.
    """

    return tensor.numel() * tensor.element_size() / (1024 ** 2)


def tensor_info(
    tensor: torch.Tensor,
):
    """
    Print useful tensor statistics.
    """

    print("=" * 60)
    print("Tensor Information")
    print("=" * 60)
    print("Shape :", tuple(tensor.shape))
    print("Elements :", tensor.numel())
    print("Memory (MB): {:.4f}".format(memory_size(tensor)))
    print("dtype :", tensor.dtype)
    print("device :", tensor.device)
    print("=" * 60)