import torch

from decompositions.tensor_train.matrix_decomposition import (
    tt_matrix_svd,
)
from decompositions.tensor_train.matrix_reconstruction import (
    reconstruct_matrix,
)
from tensorization.utils import relative_error


def test_matrix_reconstruction():

    W = torch.randn(64, 64)

    tt = tt_matrix_svd(
        W,
        input_dims=(4, 4, 4),
        output_dims=(4, 4, 4),
        max_rank=64,
    )

    reconstructed = reconstruct_matrix(tt)

    print("Original Shape     :", W.shape)
    print("Reconstructed Shape:", reconstructed.shape)

    error = relative_error(
        W,
        reconstructed,
    )

    print("Relative Error     :", error)

    assert error < 1e-5

    print("✓ Matrix reconstruction passed")


if __name__ == "__main__":

    test_matrix_reconstruction()

    print("=" * 60)
    print("MATRIX RECONSTRUCTION TEST PASSED")
    print("=" * 60)