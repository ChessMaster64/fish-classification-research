import torch

from decompositions.tensor_train.matrix import TTMatrix
from layers.tt_linear import TTLinear


def test_ttlinear():

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

    layer = TTLinear(tt)

    print(layer)

    print("Input Features :", layer.in_features)
    print("Output Features:", layer.out_features)

    print("Bias Shape :", layer.bias.shape)
    print()

    print("Trainable Parameters:")

    for name, param in layer.named_parameters():
        print(name, param.shape)

    print("✓ TTLinear Constructor")
    print()

    x = torch.randn(
        5,
        layer.in_features,
    )

    y = layer(x)

    print()
    print("Forward Shape :", y.shape)

    assert y.shape == (5, layer.out_features)

    print("✓ Full forward pass")


if __name__ == "__main__":

    test_ttlinear()

    print("=" * 60)
    print("TTLinear TEST PASSED")
    print("=" * 60)
