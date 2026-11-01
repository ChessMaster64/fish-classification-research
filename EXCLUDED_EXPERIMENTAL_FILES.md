# Excluded experimental files

This document records what should not be included in a clean public release
without manual confirmation. Nothing listed here was deleted by the audit.

## Legacy root implementation

The original tracked root scripts and weights are outside this release.
Historical copies in the source repository included earlier CNN, ViT,
ViT-GNN, test-time augmentation, data-loader, and evaluation scripts. They are
historical evidence, not the current root execution path.

## Archived variants

`older start/` contains multiple ResNet50/ViT/GNN interleaving and MPO/TT
variants plus checkpoints. `archive/v1_baseline` through
`archive/v8_tr_transformer` contain prior experiment directories. Keep them
outside the release unless a paper table explicitly depends on a particular
variant and its provenance is verified.

## Third-party tree

`torch-mpo/` is a separately authored project with its own MIT/Apache license
metadata and dependency manifest. It is not imported by the current
root execution path. Do not publish it as original project code; either
retain it as a clearly attributed dependency or remove it after rights review.

## Large assets

The dataset, checkpoints, zip archives, and generated images/reports should be
distributed separately or regenerated. Checkpoint provenance and redistribution
terms are not established.
