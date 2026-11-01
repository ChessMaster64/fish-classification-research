import torch

from decompositions.tensor_train.matrix import TTMatrix
from layers.tt_linear import TTLinear


def test_forward_placeholder():

    cores = [
        torch.randn(1,4,4,8),
        torch.randn(8,4,4,8),
        torch.randn(8,4,4,1),
    ]

    tt = TTMatrix(
        cores=cores,
        output_dims=[4,4,4],
        input_dims=[4,4,4],
        ranks=[1,8,8,1],
    )

    layer = TTLinear(tt)

    x = torch.randn(64)

    try:
        layer(x)
    except NotImplementedError:
        print("✓ Forward placeholder reached")
        return



if __name__ == "__main__":

    test_forward_placeholder()

    print("="*60)
    print("TTLinear FORWARD PLACEHOLDER PASSED")
    print("="*60)