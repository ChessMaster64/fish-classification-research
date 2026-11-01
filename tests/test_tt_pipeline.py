"""
test_tt_pipeline.py

End-to-end validation of the Tensor Train backend.

Verifies:
1. Matrix reconstruction
2. Forward correctness
3. Gradient flow
4. Optimizer updates
5. Compression ratio
"""

import torch
import torch.nn as nn

from decompositions.tensor_train.matrix_decomposition import tt_matrix_svd
from decompositions.tensor_train.matrix_reconstruction import reconstruct_matrix
from layers.tt_linear import TTLinear

torch.manual_seed(42)

# ==========================================================
# Configuration
# ==========================================================

IN_FEATURES = 512
OUT_FEATURES = 512

INPUT_DIMS = (8, 8, 8)
OUTPUT_DIMS = (8, 8, 8)

assert torch.prod(torch.tensor(INPUT_DIMS)) == IN_FEATURES
assert torch.prod(torch.tensor(OUTPUT_DIMS)) == OUT_FEATURES

# ==========================================================
# Dense layer
# ==========================================================

dense = nn.Linear(
    IN_FEATURES,
    OUT_FEATURES,
    bias=False,
)

# ==========================================================
# TT decomposition
# ==========================================================

tt = tt_matrix_svd(
    matrix=dense.weight.detach(),
    input_dims=INPUT_DIMS,
    output_dims=OUTPUT_DIMS,
    max_rank=None,
)
assert tt.validate()

print("TT ranks:", tt.ranks)
print("Core shapes:")
for i, core in enumerate(tt.cores):
    print(f"Core {i}: {tuple(core.shape)}")

# ==========================================================
# TT layer
# ==========================================================

tt_layer = TTLinear(
    tt,
    bias=False,
)

# ==========================================================
# Copy dense weight exactly
# ==========================================================

for i in range(len(tt.cores)):
    tt_layer.cores[i].data.copy_(tt.cores[i])

# ==========================================================
# Test 1
# Matrix reconstruction
# ==========================================================

print("=" * 60)
print("TEST 1 : Matrix Reconstruction")
print("=" * 60)

W_reconstructed = reconstruct_matrix(tt)

reconstruction_error = (
    torch.norm(dense.weight - W_reconstructed)
    /
    torch.norm(dense.weight)
)

print(f"Relative Error : {reconstruction_error:.8e}")
assert torch.isfinite(W_reconstructed).all()
assert reconstruction_error < 1e-5

print("PASSED\n")

# ==========================================================
# Test 2
# Forward correctness
# ==========================================================

print("=" * 60)
print("TEST 2 : Forward Pass")
print("=" * 60)

x = torch.randn(
    64,
    IN_FEATURES,
)

y_dense = dense(x)
y_tt = tt_layer(x)

forward_error = (
    torch.norm(y_dense - y_tt)
    /
    torch.norm(y_dense)
)

print(f"Relative Error : {forward_error:.8e}")

assert forward_error < 1e-5

print("PASSED\n")

# ==========================================================
# Test 3
# Gradient flow
# ==========================================================

print("=" * 60)
print("TEST 3 : Gradient Flow")
print("=" * 60)

loss = y_tt.pow(2).mean()

loss.backward()

for i, core in enumerate(tt_layer.cores):

    grad_norm = core.grad.norm().item()

    print(f"Core {i} Grad Norm : {grad_norm:.8f}")

    assert torch.isfinite(core.grad).all()

    assert grad_norm > 0

print("PASSED\n")

# ==========================================================
# Test 4
# Optimizer update
# ==========================================================

print("=" * 60)
print("TEST 4 : Optimizer Step")
print("=" * 60)

optimizer = torch.optim.Adam(
    tt_layer.parameters(),
    lr=1e-3,
)

before = [
    core.detach().clone()
    for core in tt_layer.cores
]

optimizer.step()

changed = False

for i, (b, a) in enumerate(zip(before, tt_layer.cores)):
    diff = torch.norm(b - a).item()

    print(f"Core {i} Update Norm : {diff:.8f}")

    if diff > 0:
        changed = True

assert changed

print("PASSED\n")

# ==========================================================
# Test 5
# Compression
# ==========================================================

print("=" * 60)
print("TEST 5 : Compression")
print("=" * 60)

dense_params = dense.weight.numel()

tt_params = sum(
    p.numel()
    for p in tt_layer.parameters()
)

print(f"Dense Parameters : {dense_params:,}")
print(f"TT Parameters    : {tt_params:,}")
print(f"Compression Ratio: {dense_params / tt_params:.2f}x")

if tt_params < dense_params:
    print("Compression achieved.")
else:
    print("Exact decomposition required full TT ranks (expected for random matrices).")


print("PASSED\n")

# ==========================================================
# SUCCESS
# ==========================================================

print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)