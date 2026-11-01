import torch

from decompositions.tensor_train.decomposition import tt_svd


def test_tt():

    tensor = torch.randn(
        4,
        4,
        4,
        4,
    )

    tt = tt_svd(
        tensor,
        max_rank=8,
    )

    tt.summary()

    assert tt.order == 4

    assert len(tt.cores) == 4

    assert tt.ranks[0] == 1

    assert tt.ranks[-1] == 1

    print("✓ TT-SVD")


if __name__ == "__main__":

    test_tt()

    print("=" * 60)
    print("TT-SVD TEST PASSED")
    print("=" * 60)