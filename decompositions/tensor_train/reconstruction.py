"""
reconstruction.py

Tensor Train reconstruction.
"""

import torch

from decompositions.tensor_train.tensor import TTTensor


def reconstruct(tt: TTTensor) -> torch.Tensor:
    """
    Reconstruct a dense tensor from a Tensor Train.

    Parameters
    ----------
    tt
        Tensor Train decomposition.

    Returns
    -------
    torch.Tensor
        Dense reconstructed tensor.
    """

    result = tt.cores[0]

    # result shape:
    # (1, n1, r1)

    for core in tt.cores[1:]:

        # Contract along TT rank
        result = torch.tensordot(
            result,
            core,
            dims=([-1], [0])
        )

    # Remove boundary ranks
    result = result.squeeze(0).squeeze(-1)

    return result