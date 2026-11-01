import torch

from decompositions.tensor_train.decomposition import (
    reshape_for_svd,
    truncated_svd,
    extract_core,
    update_residual,
)


def test_one_step():

    tensor = torch.randn(
        4,
        5,
        6,
    )

    matrix = reshape_for_svd(
        tensor,
        left_rank=1,
        mode_size=4,
    )

    print("Matrix Shape:", matrix.shape)

    U, S, Vh = truncated_svd(
        matrix,
        max_rank=3,
    )

    core = extract_core(
        U,
        left_rank=1,
        mode_size=4,
        rank=3,
    )

    residual = update_residual(
        S,
        Vh,
    )

    print("Core Shape:", core.shape)
    print("Residual Shape:", residual.shape)

    assert core.shape == (1, 4, 3)

    assert residual.shape == (3, 30)

    print("✓ Single TT iteration passed")


if __name__ == "__main__":

    test_one_step()

    print("=" * 60)
    print("ALL TT STEP TESTS PASSED")
    print("=" * 60)