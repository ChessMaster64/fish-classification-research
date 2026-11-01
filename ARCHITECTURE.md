# Architecture

## Dense candidate

The candidate model in
[`models/hybrid_model.py`](./models/hybrid_model.py) uses:

```text
Image
  -> ResNet50 stem, layer1, layer2, layer3
  -> 14 x 14 x 1024 feature map
  -> 196 tokens
  -> Linear 1024 -> 384 + learned positional embedding
  -> Transformer depth 3
  -> DynamicGNN k=6 + residual
  -> Transformer depth 3
  -> DynamicGNN k=6 + residual
  -> Transformer depth 2
  -> Linear 384 -> 1024
  -> reshape to feature map
  -> ResNet50 layer4
  -> global average pooling
  -> Linear 2048 -> 23
```

The configured embedding size is 384, with 8 attention heads and dropout 0.1.
The exact feed-forward width and normalization behavior must be read from the
encoder implementation when reporting architecture details; older notes are
not treated as authoritative.

## BlockTT variant

[`compression/replace.py`](./compression/replace.py)
deep-copies the dense model and recursively converts `nn.Linear` modules to
`BlockTTLinear`. The default configuration uses two row blocks, two column
blocks, and maximum TT rank 16. The module named `head` is intentionally
excluded. ResNet50 convolutions, batch normalization, layer normalization,
pooling, and residual connections remain dense/unchanged.

## Provenance warning

The repository also contains `HybridEncoder`, `models/dynamic_gnn.py`, legacy
models, and multiple archived architectures. They are not the execution path
of `train_dense.py` or `train_blocktt.py` unless explicitly imported there.
