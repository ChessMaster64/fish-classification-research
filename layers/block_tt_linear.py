"""
block_tt_linear.py

Block Tensor Train Linear layer.
"""

import torch
import torch.nn as nn


class BlockTTLinear(nn.Module):

    def __init__(
        self,
        blocks,
        row_blocks,
        col_blocks,
        bias=None,
    ):
        super().__init__()

        self.row_blocks = row_blocks
        self.col_blocks = col_blocks

        self.blocks = nn.ModuleList()

        for row in blocks:
            for block in row:
                self.blocks.append(block)

        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(
                bias.detach().clone()
            )
    
    @staticmethod
    def split_weight(
        weight: torch.Tensor,
        row_blocks: int,
        col_blocks: int,
    ):
        """
        Split a dense weight matrix into equally sized blocks.

        Parameters
        ----------
        weight : Tensor
            Shape (out_features, in_features)

        row_blocks : int
            Number of row partitions.

        col_blocks : int
            Number of column partitions.

        Returns
        -------
        list[list[Tensor]]
            2D list of weight blocks.
        """

        out_features, in_features = weight.shape

        if out_features % row_blocks != 0:
            raise ValueError(
                "Output features must be divisible by row_blocks."
            )

        if in_features % col_blocks != 0:
            raise ValueError(
                "Input features must be divisible by col_blocks."
            )

        rows_per_block = out_features // row_blocks
        cols_per_block = in_features // col_blocks

        blocks = []

        for i in range(row_blocks):

            row = []

            row_start = i * rows_per_block
            row_end = row_start + rows_per_block

            for j in range(col_blocks):

                col_start = j * cols_per_block
                col_end = col_start + cols_per_block

                block = weight[
                    row_start:row_end,
                    col_start:col_end,
                ]

                row.append(block)

            blocks.append(row)

        return blocks
    
    def _block(self, row: int, col: int):
        """
        Return the TTLinear at (row, col).
        """
        return self.blocks[
            row * self.col_blocks + col
        ]
    
    def forward(self, x):

        in_features = x.shape[-1]

        cols_per_block = (
            in_features //
            self.col_blocks
        )

        x_chunks = torch.split(
            x,
            cols_per_block,
            dim=-1,
        )

        outputs = []

        rows_per_block = self.blocks[0].out_features

        if self.bias is not None:
            bias_chunks = torch.split(
                self.bias,
                rows_per_block,
            )
        else:
            bias_chunks = None

        for r in range(self.row_blocks):

            y = None

            for c in range(self.col_blocks):

                block = self._block(r, c)

                block_output = block(
                    x_chunks[c]
                )

                if y is None:
                    y = block_output
                else:
                    y = y + block_output

            if bias_chunks is not None:
                y = y + bias_chunks[r]

            outputs.append(y)

        return torch.cat(
            outputs,
            dim=-1,
        )