from models.multihead_attention import MultiHeadAttention
from compression.linear_to_blocktt import compress_linear_to_blocktt
from compression.config import BlockTTConfig


def compress_attention(
    attention: MultiHeadAttention,
    config: BlockTTConfig,
):

    attention.q_proj = compress_linear_to_blocktt(
        attention.q_proj,
        config,
    )

    attention.k_proj = compress_linear_to_blocktt(
        attention.k_proj,
        config,
    )

    attention.v_proj = compress_linear_to_blocktt(
        attention.v_proj,
        config,
    )

    attention.out_proj = compress_linear_to_blocktt(
        attention.out_proj,
        config,
    )

    return attention