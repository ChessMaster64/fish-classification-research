import torch

from decompositions.tensor_train.matrix import TTMatrix
from layers.tt_linear import TTLinear


# Create a random dense matrix
W = torch.randn(64, 128)

# Tensorize:
# output = 4×4×4 = 64
# input  = 8×4×4 = 128
tt = TTMatrix.from_matrix(
    W,
    output_dims=(4, 4, 4),
    input_dims=(8, 4, 4),
    rank=8,
)

layer = TTLinear(tt)

print("Order:", layer.order)
print("Input dims:", layer.input_dims)
print("Output dims:", layer.output_dims)
print("Ranks:", layer.ranks)
print("In features:", layer.in_features)
print("Out features:", layer.out_features)

print("\nCore shapes")
for i, core in enumerate(layer.cores):
    print(f"Core {i}: {tuple(core.shape)}")

print("\nBias:", layer.bias.shape if layer.bias is not None else None)