import torch.nn as nn

from compression.build_tt_block import build_tt_block
from layers.block_tt_linear import BlockTTLinear
from compression.config import BlockTTConfig
from compression.tensorization import infer_tensor_shape

def compress_linear_to_blocktt(
    linear,
    config: BlockTTConfig,
):
    """
    Convert a dense nn.Linear into BlockTTLinear.

    Parameters
    ----------
    linear
        Dense linear layer.

    input_dims
        Tensorization of ONE input block.

    output_dims
        Tensorization of ONE output block.

    max_rank
        TT rank.

    row_blocks
        Number of output blocks.

    col_blocks
        Number of input blocks.
    """

    weight = linear.weight.detach()
    out_block = (
        linear.out_features //
        config.row_blocks
    )

    in_block = (
        linear.in_features //
        config.col_blocks
    )

    output_dims = infer_tensor_shape(
        out_block
    )

    input_dims = infer_tensor_shape(
        in_block
    )
    
    blocks = BlockTTLinear.split_weight(
        weight,
        row_blocks=config.row_blocks,
        col_blocks=config.col_blocks,
    )

    tt_blocks = []

    
    for row in blocks:

        tt_row = []

        for block in row:

            layer = build_tt_block(
                matrix=block,
                input_dims=input_dims,
                output_dims=output_dims,
                max_rank=config.max_rank,
            )

            tt_row.append(layer)

        tt_blocks.append(tt_row)

    return BlockTTLinear(
        blocks=tt_blocks,
        row_blocks=config.row_blocks,
        col_blocks=config.col_blocks,
        bias=linear.bias,
    )