import torch

from layers.block_tt_linear import BlockTTLinear


def test_split_weight():

    W = torch.arange(64).reshape(8, 8)

    blocks = BlockTTLinear.split_weight(
        W,
        row_blocks=2,
        col_blocks=2,
    )

    W11 = blocks[0][0]
    W12 = blocks[0][1]
    W21 = blocks[1][0]
    W22 = blocks[1][1]

    reconstructed = torch.cat(
        [
            torch.cat([W11, W12], dim=1),
            torch.cat([W21, W22], dim=1),
        ],
        dim=0,
    )

    assert torch.equal(W, reconstructed)

    print("✓ Block split test passed.")

if __name__ == "__main__":
    test_split_weight()