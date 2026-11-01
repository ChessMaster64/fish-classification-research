from networkx import config
import torch
import torch.nn as nn

from compression.linear_to_blocktt import compress_linear_to_blocktt


def test_blocktt_correctness():

    torch.manual_seed(42)

    linear = nn.Linear(8, 8)

    blocktt = compress_linear_to_blocktt(
        linear,
        config
    )

    x = torch.randn(32, 8)

    dense = linear(x)
    tt = blocktt(x)

    max_error = (dense - tt).abs().max()

    relative_error = (
        torch.norm(dense - tt)
        / torch.norm(dense)
    )

    print(f"Max Error      : {max_error:.8f}")
    print(f"Relative Error : {relative_error:.8e}")


if __name__ == "__main__":
    test_blocktt_correctness()