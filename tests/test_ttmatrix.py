import torch

from decompositions.tensor_train.matrix import TTMatrix


def test_ttmatrix():

    cores = [
        torch.randn(1, 4, 4, 8),
        torch.randn(8, 4, 4, 8),
        torch.randn(8, 4, 4, 1),
    ]

    tt = TTMatrix(
        cores=cores,
        output_dims=[4, 4, 4],
        input_dims=[4, 4, 4],
        ranks=[1, 8, 8, 1],
    )

    tt.validate()

    tt.summary()

    print(
        "Original Parameters :",
        tt.original_parameters(),
    )

    print(
        "TT Parameters :",
        tt.num_parameters(),
    )

    print(
        "Compression :",
        tt.compression_ratio(),
    )

    print("✓ TTMatrix")


if __name__ == "__main__":

    test_ttmatrix()

    print("=" * 60)
    print("TTMatrix TEST PASSED")
    print("=" * 60)