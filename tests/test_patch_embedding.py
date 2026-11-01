import torch

from layers.patch_embedding import PatchEmbedding


model = PatchEmbedding()

x = torch.randn(
    2,
    3,
    224,
    224,
)

y = model(x)

print("Output shape:", y.shape)

print("Number of patches:", model.num_patches)