"""
decomposition.py

Core mathematical operations used by Tensor Train decomposition.

Version 1:
- Tensor unfolding
- Truncated SVD

TT-SVD will be added next.
"""

from typing import Tuple

import torch


def unfold_tensor(
    tensor: torch.Tensor,
    split: int,
) -> torch.Tensor:
    """
    Unfold a tensor into a matrix.

    Parameters
    ----------
    tensor
        Input tensor.

    split
        Number of leading dimensions that form the rows.

    Returns
    -------
    matrix
    """

    if split <= 0:
        raise ValueError("split must be positive.")

    if split >= tensor.ndim:
        raise ValueError("split must be smaller than tensor order.")

    left = int(torch.tensor(tensor.shape[:split]).prod().item())
    right = int(torch.tensor(tensor.shape[split:]).prod().item())

    return tensor.reshape(left, right)


def compute_svd(
    matrix: torch.Tensor,
):
    """
    Compute the compact Singular Value Decomposition.

    Returns
    -------
    U, S, Vh
    """

    if matrix.ndim != 2:
        raise ValueError("Input must be a matrix.")

    return torch.linalg.svd(
        matrix,
        full_matrices=False,
    )


def select_rank(
    singular_values: torch.Tensor,
    max_rank: int | None = None,
    energy: float | None = None,
    epsilon: float | None = None,
):
    """
    Select the truncation rank.

    Priority

    epsilon
        Reconstruction error threshold.

    energy
        Fraction of retained energy.

    max_rank
        Upper bound.
    """

    total_rank = singular_values.numel()

    rank = total_rank

    # -----------------------------
    # Energy criterion
    # -----------------------------

    if energy is not None:

        cumulative = torch.cumsum(
            singular_values ** 2,
            dim=0,
        )

        total_energy = cumulative[-1].clone()
        cumulative = cumulative / total_energy

        rank = int(
            torch.searchsorted(
                cumulative,
                energy,
            ).item()
        ) + 1

    # -----------------------------
    # Error criterion
    # -----------------------------

    if epsilon is not None:

        squared = singular_values ** 2

        total = squared.sum()

        for r in range(total_rank):

            error = torch.sqrt(
                squared[r:].sum() / total
            )

            if error <= epsilon:
                rank = r
                break

    if max_rank is not None:
        rank = min(rank, max_rank)

    rank = max(rank, 1)

    return rank


def truncated_svd(
    matrix: torch.Tensor,
    max_rank: int | None = None,
    energy: float | None = None,
    epsilon: float | None = None,
):
    """
    Compute truncated SVD.

    Rank can be chosen using

    - max_rank
    - energy
    - epsilon
    """

    U, S, Vh = compute_svd(matrix)

    rank = select_rank(
        S,
        max_rank=max_rank,
        energy=energy,
        epsilon=epsilon,
    )

    return (
        U[:, :rank],
        S[:rank],
        Vh[:rank],
    )

def reconstruct_from_svd(
    U: torch.Tensor,
    S: torch.Tensor,
    Vh: torch.Tensor,
):
    """
    Reconstruct matrix from SVD.
    """

    return U @ torch.diag(S) @ Vh

def reshape_for_svd(
    tensor: torch.Tensor,
    left_rank: int,
    mode_size: int,
):
    """
    Reshape the current tensor into a matrix suitable for TT-SVD.

    Matrix shape:

    (left_rank * mode_size, remaining_elements)
    """

    rows = left_rank * mode_size

    cols = tensor.numel() // rows

    return tensor.reshape(rows, cols)


def extract_core(
    U: torch.Tensor,
    left_rank: int,
    mode_size: int,
    rank: int,
):
    """
    Convert the left singular vectors into a TT core.

    Output shape:

    (left_rank, mode_size, rank)
    """

    return U.reshape(
        left_rank,
        mode_size,
        rank,
    )


def update_residual(
    S: torch.Tensor,
    Vh: torch.Tensor,
):
    """
    Compute the remaining tensor after extracting one TT core.

    Residual = diag(S) @ Vh
    """

    return torch.diag(S) @ Vh

from decompositions.tensor_train.tensor import TTTensor


def tt_svd(
    tensor: torch.Tensor,
    max_rank: int | None = None,
    energy: float | None = None,
    epsilon: float | None = None,
):
    
    """
    Tensor Train decomposition using TT-SVD.

    Parameters
    ----------
    tensor
        Tensor to decompose.

    mode_sizes
        Shape of the tensor.
        Example:
        (4,4,4,4)

    max_rank
        Maximum TT rank.

    Returns
    -------
    TTTensor
    """
    mode_sizes = tuple(tensor.shape)
    tensor = tensor.contiguous()

    d = len(mode_sizes)

    ranks = [1]

    cores = []

    residual = tensor

    left_rank = 1

    for k in range(d - 1):

        n = mode_sizes[k]

        matrix = reshape_for_svd(
            residual,
            left_rank,
            n,
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
            n,
            rank,
        )

        cores.append(core)

        residual = update_residual(
            S,
            Vh,
        )

        remaining_shape = mode_sizes[k + 1 :]

        residual = residual.reshape(
            rank,
            *remaining_shape,
        )

        ranks.append(rank)

        left_rank = rank

    last_core = residual.reshape(
        left_rank,
        mode_sizes[-1],
        1,
    )

    cores.append(last_core)

    ranks.append(1)

    return TTTensor(
        cores=cores,
        mode_dims=list(mode_sizes),
        ranks=ranks,
    )