import torch
import torch.nn as nn

from tensorization.utils import (
    compression_ratio,
    reconstruction_error,
    relative_error,
    tensor_info,
    count_parameters,
)


class Dummy(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(128,64)


def test_parameters():

    model = Dummy()

    params = count_parameters(model)

    print("Trainable Parameters:",params)


def test_compression():

    ratio = compression_ratio(1000000,100000)

    print("Compression Ratio:",ratio)


def test_errors():

    A = torch.randn(32,32)

    B = A.clone()

    print("Reconstruction Error:",
          reconstruction_error(A,B))

    print("Relative Error:",
          relative_error(A,B))


def test_tensor_info():

    tensor = torch.randn(64,64)

    tensor_info(tensor)


if __name__=="__main__":

    test_parameters()

    test_compression()

    test_errors()

    test_tensor_info()

    print("="*60)
    print("ALL UTILITY TESTS PASSED")
    print("="*60)