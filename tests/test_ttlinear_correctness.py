import torch

from decompositions.tensor_train.matrix_decomposition import tt_matrix_svd
from layers.tt_linear_reference import TTLinearReference

torch.manual_seed(42)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

input_dims = (4, 4, 4)
output_dims = (4, 4, 4)

in_features = 64
out_features = 64

# ------------------------------------------------------------------
# Create dense weight
# ------------------------------------------------------------------

W = torch.randn(
    out_features,
    in_features,
)

# ------------------------------------------------------------------
# TT decomposition
# ------------------------------------------------------------------

tt = tt_matrix_svd(
    matrix=W,
    input_dims=input_dims,
    output_dims=output_dims,
)

# ------------------------------------------------------------------
# TT Linear
# ------------------------------------------------------------------

layer = TTLinearReference(
    tt,
    bias=False,
)

# ------------------------------------------------------------------
# Random input
# ------------------------------------------------------------------

x = torch.randn(
    32,
    in_features,
)

# ------------------------------------------------------------------
# Dense output
# ------------------------------------------------------------------

y_dense = x @ W.t()

# ------------------------------------------------------------------
# TT output
# ------------------------------------------------------------------

y_tt = layer(x)

# ------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------

max_error = torch.max(
    torch.abs(
        y_dense - y_tt
    )
)

relative_error = (
    torch.norm(y_dense - y_tt)
    /
    torch.norm(y_dense)
)

print("=" * 60)
print("TTLinear Correctness Test")
print("=" * 60)

print(f"Max Error      : {max_error:.8f}")
print(f"Relative Error : {relative_error:.8e}")

print("=" * 60)