import torch

from decompositions.tensor_train.matrix_decomposition import tt_matrix_svd
from layers.tt_linear import TTLinear

torch.manual_seed(42)

W = torch.randn(4, 4)

tt = tt_matrix_svd(
    W,
    input_dims=(2,2),
    output_dims=(2,2),
    max_rank=16,
)

layer = TTLinear(
    tt_matrix=tt,
    bias=False,
)

x = torch.randn(8,4)

dense = x @ W.T
tt_out = layer(x)

print("Dense")
print(dense)

print()

print("TT")
print(tt_out)

print()

print("Relative error")

print(
    torch.norm(dense-tt_out) /
    torch.norm(dense)
)

print(layer._einsum_equation)