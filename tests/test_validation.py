import torch

from decompositions.tensor_train.decomposition import tt_svd
from decompositions.tensor_train.validation import validate


def test_validation():

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

    assert validate(tt)

    print("✓ Validation passed")


if __name__ == "__main__":

    test_validation()

    print("=" * 60)
    print("VALIDATION TEST PASSED")
    print("=" * 60)