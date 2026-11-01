import torch

from tensorization.tensorize import (
    tensorize,
    detensorize,
    tensorize_from_shape,
)


def test_tensorization():

    matrix = torch.randn(768, 3072)

    tensor, out_dims, in_dims = tensorize(matrix)

    reconstructed = detensorize(
        tensor,
        out_dims,
        in_dims,
    )

    assert torch.allclose(matrix, reconstructed)

    print("✓ Tensorization reconstruction passed")


def test_manual_tensorization():

    matrix = torch.randn(64, 128)

    out_dims = [4, 4, 4]
    in_dims = [4, 4, 8]

    tensor = tensorize_from_shape(
        matrix,
        out_dims,
        in_dims,
    )

    reconstructed = detensorize(
        tensor,
        out_dims,
        in_dims,
    )

    assert torch.allclose(matrix, reconstructed)

    print("✓ Manual tensorization passed")


if __name__ == "__main__":

    test_tensorization()

    test_manual_tensorization()

    print("=" * 60)
    print("ALL TENSORIZATION TESTS PASSED")
    print("=" * 60)