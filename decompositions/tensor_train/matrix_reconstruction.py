"""
matrix_reconstruction.py

Reconstruct a dense matrix from a TTMatrix.
"""

import torch

from decompositions.tensor_train.matrix import TTMatrix


def reconstruct_matrix(
    tt: TTMatrix,
) -> torch.Tensor:
    """
    Reconstruct a dense matrix from a TTMatrix.
    """

    tensor = tt.cores[0]

    for core in tt.cores[1:]:

        tensor = torch.tensordot(
            tensor,
            core,
            dims=([-1], [0]),
        )

    # Remove boundary TT ranks
    tensor = tensor.squeeze(0).squeeze(-1)
    order = len(tt.output_dims)

    perm = []

    # Output modes first
    for k in range(order):
        perm.append(2 * k)

    # Input modes next
    for k in range(order):
        perm.append(2 * k + 1)

    tensor = tensor.permute(*perm).contiguous()

    # Current shape:
    # (o1, i1, o2, i2, ..., od, id)

    output_dims = tt.output_dims
    input_dims = tt.input_dims

    out_features = 1
    for d in output_dims:
        out_features *= d

    in_features = 1
    for d in input_dims:
        in_features *= d

    return tensor.reshape(
        out_features,
        in_features,
    )