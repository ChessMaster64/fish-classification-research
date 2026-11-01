import torch

from decompositions.tensor_train.tensor import TTTensor


def test_tttensor():

    cores = [
        torch.randn(1, 4, 8),
        torch.randn(8, 4, 8),
        torch.randn(8, 4, 1),
    ]

    tt = TTTensor(
        cores=cores,
        mode_dims=[4, 4, 4],
        ranks=[1, 8, 8, 1],
    )

    assert tt.order == 3
    assert tt.num_cores == 3

    tt.summary()

    print("Original Parameters :", tt.original_parameters())
    print("TT Parameters :", tt.num_parameters())
    print("Compression :", tt.compression_ratio())

    print("✓ TTTensor")


if __name__ == "__main__":

    test_tttensor()

    print("=" * 60)
    print("TTTensor TEST PASSED")
    print("=" * 60)