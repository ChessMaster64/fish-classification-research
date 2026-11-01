"""
tensorize.py

Utilities for converting matrices into tensors and back.

These operations are decomposition-independent and are shared by
Tensor Train (TT), Tensor Ring (TR), Matrix Product Operator (MPO),
and Block Tensor Train (BlockTT).
"""

from typing import Tuple

import torch

from tensorization.factorization import factorize_matrix


def tensorize(
    matrix: torch.Tensor,
    input_order: int = 4,
    output_order: int = 4,
):
    """
    Convert a 2D matrix into a higher-order tensor.

    Parameters
    ----------
    matrix : torch.Tensor
        Shape (out_features, in_features)

    Returns
    -------
    tensor : torch.Tensor
        Tensorized weight.

    output_dims : list
        Factorization of output dimension.

    input_dims : list
        Factorization of input dimension.
    """

    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")

    out_features, in_features = matrix.shape

    output_dims, input_dims = factorize_matrix(
        (out_features, in_features),
        input_order=input_order,
        output_order=output_order,
    )

    tensor_shape = tuple(output_dims + input_dims)

    tensor = matrix.reshape(tensor_shape)

    return tensor, output_dims, input_dims


def detensorize(
    tensor: torch.Tensor,
    output_dims,
    input_dims,
):
    """
    Convert a tensor back into its original matrix.
    """

    out_features = int(torch.tensor(output_dims).prod().item())
    in_features = int(torch.tensor(input_dims).prod().item())

    matrix = tensor.reshape(out_features, in_features)

    return matrix


def tensorize_from_shape(
    matrix: torch.Tensor,
    output_dims,
    input_dims,
):
    """
    Tensorize using manually supplied dimensions.

    Useful for future experiments with custom tensorizations.
    """

    if matrix.ndim != 2:
        raise ValueError("Input must be 2D.")

    expected_out = int(torch.tensor(output_dims).prod().item())
    expected_in = int(torch.tensor(input_dims).prod().item())

    if matrix.shape != (expected_out, expected_in):
        raise ValueError("Shape does not match provided tensor dimensions.")

    tensor = matrix.reshape(tuple(output_dims + input_dims))

    return tensor