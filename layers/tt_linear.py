"""
tt_linear.py

Tensor Train implementation of nn.Linear.
"""

import math

import torch
import torch.nn as nn

from decompositions.tensor_train.matrix import TTMatrix


class TTLinear(nn.Module):
    """
    Tensor Train implementation of nn.Linear.
    """

    def __init__(
        self,
        tt_matrix: TTMatrix,
        bias: bool = True,
    ):
        super().__init__()

        # Tensorization metadata
        self.input_dims = tuple(tt_matrix.input_dims)
        self.output_dims = tuple(tt_matrix.output_dims)
        self.ranks = tuple(tt_matrix.ranks)
        self.order = len(self.input_dims)

        # Feature sizes
        self.in_features = math.prod(self.input_dims)
        self.out_features = math.prod(self.output_dims)

        # Trainable TT cores
        self.cores = nn.ParameterList(
            [
                nn.Parameter(core.clone())
                for core in tt_matrix.cores
            ]
        )

        # Optional bias
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features)
            )
        else:
            self.register_parameter(
                "bias",
                None,
            )

        self.reset_parameters()
        # Cache einsum equation (built once)
        self._einsum_equation = self._build_einsum()

    def reset_parameters(self):
        """
        Initialize trainable parameters.
        """

        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    
    def _build_einsum(self):
        """
        Build the einsum equation for TT-matrix × dense-vector contraction.

        Input:
            (batch, i1, i2, ..., id)

        Core k:
            (rk, ok, ik, rk+1)

        Output:
            (batch, o1, o2, ..., od)
        """

        letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

        d = self.order

        needed = 1 + d + d + (d + 1)

        if needed > len(letters):
            raise ValueError(
                f"TT order {d} is too large for einsum index generation."
            )

        batch = letters[0]

        inputs = list(letters[1:1 + d])

        outputs = list(
            letters[1 + d:1 + 2 * d]
        )

        ranks = list(
            letters[1 + 2 * d:1 + 3 * d + 1]
        )

        input_tensor = batch + "".join(inputs)

        core_terms = []

        for k in range(d):

            core_terms.append(
                ranks[k]
                + outputs[k]
                + inputs[k]
                + ranks[k + 1]
            )

        output_tensor = batch + "".join(outputs)

        equation = (
            input_tensor
            + ","
            + ",".join(core_terms)
            + "->"
            + output_tensor
        )

        return equation

    def forward(self, x):

        # Preserve all leading dimensions
        leading_shape = x.shape[:-1]

        # Check feature dimension
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected input feature dimension {self.in_features}, "
                f"got {x.shape[-1]}."
            )

        # Tensorize only the last dimension
        x = x.reshape(
            *leading_shape,
            *self.input_dims,
        )

        # Build einsum equation dynamically for arbitrary leading dims
        # Flatten leading dims temporarily
        x = x.reshape(
            -1,
            *self.input_dims,
        )

        y = torch.einsum(
            self._einsum_equation,
            x,
            *self.cores,
        )

        y = y.reshape(
            *leading_shape,
            self.out_features,
        )

        if self.bias is not None:
            y = y + self.bias

        return y