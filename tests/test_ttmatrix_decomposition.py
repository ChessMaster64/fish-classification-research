import torch

from decompositions.tensor_train.matrix_decomposition import tt_matrix_svd


def test_tt_matrix_svd():

    # 64 x 64 matrix
    W = torch.randn(64, 64)

    tt = tt_matrix_svd(
        W,
        input_dims=(4, 4, 4),
        output_dims=(4, 4, 4),
        max_rank=8,
    )

    tt.validate()

    tt.summary()

    print("Number of cores :", tt.num_cores)

    assert tt.num_cores == 3

    print("✓ TT-Matrix SVD")


if __name__ == "__main__":

    test_tt_matrix_svd()

    print("=" * 60)
    print("TTMatrix SVD TEST PASSED")
    print("=" * 60)