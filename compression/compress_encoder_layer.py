"""
compress_encoder_layer.py

Compress a Transformer encoder layer using BlockTT.
"""

from compression.compress_attention import compress_attention
from compression.compress_feedforward import compress_feedforward


def compress_encoder_layer(
    layer,
    config,
):

    layer.attention = compress_attention(
        layer.attention,
        config,
    )

    layer.ffn = compress_feedforward(
        layer.ffn,
        config,
    )

    return layer