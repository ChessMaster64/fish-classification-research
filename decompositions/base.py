"""
base.py

Abstract base class for tensor decomposition formats.
"""

from abc import ABC, abstractmethod


class TensorDecomposition(ABC):
    """
    Base interface for all tensor decomposition formats.
    """

    @abstractmethod
    def num_parameters(self) -> int:
        """Return the number of parameters in the decomposition."""
        pass

    @abstractmethod
    def original_parameters(self) -> int:
        """Return the number of parameters in the original tensor/matrix."""
        pass

    @abstractmethod
    def compression_ratio(self) -> float:
        """Return the compression ratio."""
        pass

    @abstractmethod
    def summary(self) -> None:
        """Print a summary of the decomposition."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate the decomposition."""
        pass