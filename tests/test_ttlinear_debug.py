import torch

from decompositions.tensor_train.matrix_decomposition import tt_matrix_svd
from decompositions.tensor_train.matrix_reconstruction import reconstruct_matrix
from layers.tt_linear import TTLinear
from layers.tt_linear_reference import TTLinearReference


torch.manual_seed(42)

# ==========================================================
# Small example
# ==========================================================

input_dims = (2, 2)
output_dims = (2, 2)

in_features = 4
out_features = 4

# ==========================================================
# Random dense weight
# ==========================================================

W = torch.randn(
    out_features,
    in_features,
)

print("=" * 70)
print("Original Weight")
print("=" * 70)
print(W)

# ==========================================================
# TT decomposition
# ==========================================================

tt = tt_matrix_svd(
    matrix=W,
    input_dims=input_dims,
    output_dims=output_dims,
    max_rank=16,
)

# ==========================================================
# Reconstruction
# ==========================================================

W_reconstructed = reconstruct_matrix(tt)

print("\n")
print("=" * 70)
print("Reconstructed Weight")
print("=" * 70)
print(W_reconstructed)

print("\n")
print("=" * 70)
print("Weight Reconstruction Error")
print("=" * 70)

print(
    torch.norm(W - W_reconstructed)
)

print(
    torch.max(torch.abs(W - W_reconstructed))
)

# ==========================================================
# Inputs
# ==========================================================

x = torch.randn(
    3,
    in_features,
)

print("\n")
print("=" * 70)
print("Input")
print("=" * 70)
print(x)

# ==========================================================
# Dense output
# ==========================================================

y_dense = x @ W.t()

print("\n")
print("=" * 70)
print("Dense Output")
print("=" * 70)
print(y_dense)

# ==========================================================
# Reference TT
# ==========================================================

reference = TTLinearReference(
    tt,
    bias=False,
)

y_reference = reference(x)

print("\n")
print("=" * 70)
print("Reference Output")
print("=" * 70)
print(y_reference)

# ==========================================================
# TTLinear
# ==========================================================

layer = TTLinear(
    tt,
    bias=False,
)

print("\n")
print("=" * 70)
print("Einsum Equation")
print("=" * 70)
print(layer._einsum_equation)

y_tt = layer(x)

print("\n")
print("=" * 70)
print("TT Output")
print("=" * 70)
print(y_tt)

# ==========================================================
# Errors
# ==========================================================

print("\n")
print("=" * 70)
print("Dense vs Reference")
print("=" * 70)

print(
    "Max:",
    torch.max(torch.abs(y_dense - y_reference)).item(),
)

print(
    "Relative:",
    (
        torch.norm(y_dense - y_reference)
        /
        torch.norm(y_dense)
    ).item(),
)

print("\n")
print("=" * 70)
print("Dense vs TT")
print("=" * 70)

print(
    "Max:",
    torch.max(torch.abs(y_dense - y_tt)).item(),
)

print(
    "Relative:",
    (
        torch.norm(y_dense - y_tt)
        /
        torch.norm(y_dense)
    ).item(),
)

print("\n")
print("=" * 70)
print("Reference vs TT")
print("=" * 70)

print(
    "Max:",
    torch.max(torch.abs(y_reference - y_tt)).item(),
)

print(
    "Relative:",
    (
        torch.norm(y_reference - y_tt)
        /
        torch.norm(y_reference)
    ).item(),
)