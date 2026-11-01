"""
matrix.py

Tensor Train Matrix representation.
"""

from dataclasses import dataclass
from typing import List

import torch

from decompositions.base import TensorDecomposition


@dataclass
class TTMatrix(TensorDecomposition):
    """
    Tensor Train Matrix.

    Each core has shape

    (r_k, out_mode, in_mode, r_{k+1})
    """

    cores: List[torch.Tensor]
    output_dims: List[int]
    input_dims: List[int]
    ranks: List[int]

    @property
    def order(self) -> int:
        return len(self.output_dims)

    @property
    def num_cores(self) -> int:
        return len(self.cores)

    def num_parameters(self) -> int:
        return sum(core.numel() for core in self.cores)

    def original_parameters(self) -> int:
        out_features = 1
        in_features = 1

        for dim in self.output_dims:
            out_features *= dim

        for dim in self.input_dims:
            in_features *= dim

        return out_features * in_features

    def compression_ratio(self) -> float:
        return (
            self.original_parameters()
            / self.num_parameters()
        )

    def summary(self) -> None:

        print("=" * 60)
        print("Tensor Train Matrix")
        print("=" * 60)

        print(f"Order : {self.order}")
        print(f"TT Ranks : {self.ranks}")
        print(f"Output Dims : {self.output_dims}")
        print(f"Input Dims : {self.input_dims}")
        print(f"Original Params : {self.original_parameters()}")
        print(f"TT Params : {self.num_parameters()}")
        print(f"Compression : {self.compression_ratio()}")

        print("=" * 60)

    def validate(self) -> bool:

        if len(self.cores) == 0:
            raise ValueError(
                "TTMatrix must contain at least one core."
            )

        if len(self.output_dims) != len(self.cores):
            raise ValueError(
                "Output dimensions must match number of cores."
            )

        if len(self.input_dims) != len(self.cores):
            raise ValueError(
                "Input dimensions must match number of cores."
            )

        if len(self.ranks) != len(self.cores) + 1:
            raise ValueError(
                "Ranks must have length number_of_cores + 1."
            )

        for i, core in enumerate(self.cores):

            if core.ndim != 4:
                raise ValueError(
                    f"Core {i} must be 4-dimensional."
                )

            if core.shape[0] != self.ranks[i]:
                raise ValueError(
                    f"Left rank mismatch in core {i}."
                )

            if core.shape[1] != self.output_dims[i]:
                raise ValueError(
                    f"Output dimension mismatch in core {i}."
                )

            if core.shape[2] != self.input_dims[i]:
                raise ValueError(
                    f"Input dimension mismatch in core {i}."
                )

            if core.shape[3] != self.ranks[i + 1]:
                raise ValueError(
                    f"Right rank mismatch in core {i}."
                )

        return True