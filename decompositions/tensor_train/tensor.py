from dataclasses import dataclass
from typing import List
from decompositions.base import TensorDecomposition

import torch


@dataclass
class TTTensor(TensorDecomposition):
    """
    Container for a Tensor Train decomposition of a tensor.

    Parameters
    ----------
    cores
        List of TT cores.

    mode_dims
        Dimensions of the original tensor.

    ranks
        TT ranks.
    """

    cores: List[torch.Tensor]
    mode_dims: List[int]
    ranks: List[int]

    @property
    def order(self):
        return len(self.cores)

    @property
    def num_cores(self):
        return len(self.cores)

    def num_parameters(self):
        """
        Number of parameters stored in TT format.
        """
        return sum(core.numel() for core in self.cores)

    def original_parameters(self):
        """
        Number of parameters in the original dense tensor.
        """
        total = 1

        for d in self.mode_dims:
            total *= d

        return total

    def compression_ratio(self):
        return (
            self.original_parameters()
            / self.num_parameters()
        )

    def summary(self):

        print("=" * 60)
        print("Tensor Train Tensor")
        print("=" * 60)

        print("Order :", self.order)
        print("TT Ranks :", self.ranks)
        print("Mode Dims :", self.mode_dims)

        print("Original Params :", self.original_parameters())
        print("TT Params :", self.num_parameters())
        print("Compression :", self.compression_ratio())

        print("=" * 60)

    def validate(self) -> bool:
        """
        Validate the Tensor Train decomposition.
        """

        if len(self.cores) == 0:
            raise ValueError("TTTensor must contain at least one core.")

        if len(self.mode_dims) != len(self.cores):
            raise ValueError(
                "Number of mode dimensions must match number of TT cores."
            )

        if len(self.ranks) != len(self.cores) + 1:
            raise ValueError(
                "TT ranks must have length number_of_cores + 1."
            )

        for i, core in enumerate(self.cores):

            if core.ndim != 3:
                raise ValueError(
                    f"Core {i} must be 3-dimensional."
                )

            if core.shape[0] != self.ranks[i]:
                raise ValueError(
                    f"Left rank mismatch in core {i}."
                )

            if core.shape[1] != self.mode_dims[i]:
                raise ValueError(
                    f"Mode dimension mismatch in core {i}."
                )

            if core.shape[2] != self.ranks[i + 1]:
                raise ValueError(
                    f"Right rank mismatch in core {i}."
                )

        return True