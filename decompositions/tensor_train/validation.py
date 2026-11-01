"""
validation.py

Validation utilities for Tensor Train decompositions.
"""

import torch

from decompositions.tensor_train.tensor import TTTensor


def validate_ranks(tt: TTTensor):
    """
    Validate TT rank consistency.
    """

    if len(tt.ranks) != tt.order + 1:
        raise ValueError(
            "Number of TT ranks must equal number of cores + 1."
        )

    if tt.ranks[0] != 1:
        raise ValueError(
            "First TT rank must be 1."
        )

    if tt.ranks[-1] != 1:
        raise ValueError(
            "Last TT rank must be 1."
        )

    for i, core in enumerate(tt.cores):

        left_rank, mode, right_rank = core.shape

        if left_rank != tt.ranks[i]:
            raise ValueError(
                f"Core {i} left rank mismatch."
            )

        if right_rank != tt.ranks[i + 1]:
            raise ValueError(
                f"Core {i} right rank mismatch."
            )

    return True


def validate_dimensions(tt: TTTensor):
    """
    Validate tensor dimensions.
    """

    if len(tt.mode_dims) != tt.order:
        raise ValueError(
            "Number of mode dimensions must equal tensor order."
        )

    for mode, core in zip(tt.mode_dims, tt.cores):

        if core.shape[1] != mode:
            raise ValueError(
                "Mode dimension mismatch."
            )

    return True


def validate(tt: TTTensor):
    """
    Run all validation checks.
    """

    validate_ranks(tt)
    validate_dimensions(tt)

    return True