import torch

from decompositions.tensor_train.decomposition import tt_svd
from decompositions.tensor_train.reconstruction import reconstruct


def test_reconstruction():

    torch.manual_seed(42)

    tensor = torch.randn(
        4,
        4,
        4,
        4,
    )

    tt = tt_svd(
        tensor,
        max_rank=32,
    )

    reconstructed = reconstruct(tt)

    relative_error = (
        torch.norm(tensor - reconstructed)
        / torch.norm(tensor)
    ).item()

    print("Original Shape     :", tensor.shape)
    print("Reconstructed Shape:", reconstructed.shape)
    print("Relative Error     :", relative_error)

    assert reconstructed.shape == tensor.shape
    assert relative_error < 1e-5

    print("✓ Reconstruction passed")


if __name__ == "__main__":

    test_reconstruction()

    print("=" * 60)
    print("RECONSTRUCTION TEST PASSED")
    print("=" * 60)