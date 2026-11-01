from logging import config

import torch
import torch.nn as nn

from compression.linear_to_blocktt import compress_linear_to_blocktt
from layers.tt_linear import TTLinear


def test_blocktt_creation():

    # 8x8 dense layer
    linear = nn.Linear(8, 8)

    # Each block is 4x4
    # 4 = 2 × 2
    blocktt = compress_linear_to_blocktt(
        linear,
        config
    )

    # Number of blocks
    assert len(blocktt.blocks) == 4

    print("✓ Correct number of blocks")

    # Every block should be TTLinear
    for block in blocktt.blocks:
        assert isinstance(block, TTLinear)

    print("✓ All blocks are TTLinear")

    # Bias copied correctly
    assert torch.equal(
        blocktt.bias,
        linear.bias,
    )

    print("✓ Bias copied")

    print("\nAll BlockTT creation tests passed.")


if __name__ == "__main__":
    test_blocktt_creation()