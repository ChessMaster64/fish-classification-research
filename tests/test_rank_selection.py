import torch

from decompositions.tensor_train.decomposition import (
    select_rank,
)


def test_max_rank():

    s = torch.tensor(
        [10.,9.,8.,7.,6.,5.]
    )

    rank = select_rank(
        s,
        max_rank=3,
    )

    assert rank == 3

    print("✓ Max Rank")


def test_energy():

    s = torch.tensor(
        [10.,9.,1.,0.5]
    )

    rank = select_rank(
        s,
        energy=0.99,
    )

    print("Energy Rank:", rank)

    assert rank >= 2

    print("✓ Energy")


def test_epsilon():

    s = torch.tensor(
        [10.,9.,8.,0.001]
    )

    rank = select_rank(
        s,
        epsilon=1e-3,
    )

    print("Epsilon Rank:", rank)

    assert rank == 3

    print("✓ Epsilon")


if __name__ == "__main__":

    test_max_rank()

    test_energy()

    test_epsilon()

    print("=" * 60)
    print("RANK SELECTION TESTS PASSED")
    print("=" * 60)