import torch

from decompositions.tensor_train.decomposition import (
    unfold_tensor,
    truncated_svd,
    reconstruct_from_svd,
)


def test_unfold():

    tensor = torch.randn(4,5,6,7)

    matrix = unfold_tensor(
        tensor,
        split=2,
    )

    assert matrix.shape == (20,42)

    print("✓ Tensor unfolding")


def test_svd():

    matrix = torch.randn(100,60)

    U, S, Vh = truncated_svd(
        matrix,
        max_rank=60,
    )
    reconstructed = reconstruct_from_svd(U, S, Vh)

    relative_error = (
        torch.norm(matrix - reconstructed)
        / torch.norm(matrix)
    ).item()

    print("Relative Error:", relative_error)

    assert relative_error < 1e-5

    print("✓ Full-rank SVD")


def test_truncated():

    matrix = torch.randn(100,60)

    U, S, Vh = truncated_svd(
        matrix,
        max_rank=20,
    )

    reconstructed = reconstruct_from_svd(
        U,S,Vh
    )

    error = torch.norm(
        matrix-reconstructed
    ).item()

    print("Rank-20 Error:",error)

    assert reconstructed.shape == matrix.shape

    print("✓ Truncated SVD")


if __name__=="__main__":

    test_unfold()

    test_svd()

    test_truncated()

    print("="*60)
    print("ALL DECOMPOSITION TESTS PASSED")
    print("="*60)