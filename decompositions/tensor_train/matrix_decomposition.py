"""
matrix_decomposition.py

Tensor Train Matrix decomposition.
"""

import torch

from decompositions.tensor_train.matrix import TTMatrix
from decompositions.tensor_train.decomposition import (
    truncated_svd,
    reshape_for_svd,
    extract_core,
    update_residual,
)

def tt_matrix_svd(
    matrix: torch.Tensor,
    input_dims: tuple[int, ...],
    output_dims: tuple[int, ...],
    max_rank: int | None = None,
    energy: float | None = None,
    epsilon: float | None = None,
) -> TTMatrix:
    """
    Tensor Train Matrix decomposition.

    Parameters
    ----------
    matrix
        Dense weight matrix of shape (out_features, in_features).

    input_dims
        Tensorized input dimensions.

    output_dims
        Tensorized output dimensions.

    max_rank
        Maximum TT rank.

    energy
        Energy threshold.

    epsilon
        Reconstruction error threshold.
    """
    if matrix.ndim != 2:
        raise ValueError("Input must be a matrix.")

    if len(input_dims) != len(output_dims):
        raise ValueError(
            "Input and output dimensions must have the same order."
        )

    out_features = 1
    for d in output_dims:
        out_features *= d

    in_features = 1
    for d in input_dims:
        in_features *= d

    if matrix.shape != (out_features, in_features):
        raise ValueError(
            "Matrix shape does not match the provided tensorization."
        )
    
    order = len(input_dims)

    tensor = matrix.reshape(*output_dims, *input_dims)

    perm = []

    for k in range(order):
        perm.append(k)
        perm.append(order + k)

    tensor = tensor.permute(*perm).contiguous()

    cores = []
    ranks = [1]

    residual = tensor
    left_rank = 1

    for k in range(order - 1):

        out_dim = output_dims[k]
        in_dim = input_dims[k]

        mode_size = out_dim * in_dim

        matrix = reshape_for_svd(
            residual,
            left_rank,
            mode_size,
        )

        U, S, Vh = truncated_svd(
            matrix,
            max_rank=max_rank,
            energy=energy,
            epsilon=epsilon,
        )

        rank = S.shape[0]

        core = extract_core(
            U,
            left_rank,
            mode_size,
            rank,
        )

        core = core.reshape(
            left_rank,
            out_dim,
            in_dim,
            rank,
        )

        cores.append(core)

        residual = update_residual(
            S,
            Vh,
        )

        remaining_shape = []

        for o, i in zip(
            output_dims[k + 1 :],
            input_dims[k + 1 :],
        ):
            remaining_shape.extend([o, i])

        residual = residual.reshape(
            rank,
            *remaining_shape,
        )

        ranks.append(rank)

        left_rank = rank
    last_core = residual.reshape(
    left_rank,
        output_dims[-1],
        input_dims[-1],
        1,
    )

    cores.append(last_core)

    ranks.append(1)

    return TTMatrix(
        cores=cores,
        output_dims=list(output_dims),
        input_dims=list(input_dims),
        ranks=ranks,
    )


