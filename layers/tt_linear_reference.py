import math
import torch
import torch.nn as nn

from decompositions.tensor_train.matrix_reconstruction import reconstruct_matrix


class TTLinearReference(nn.Module):

    def __init__(self, tt_matrix, bias=False):
        super().__init__()

        weight = reconstruct_matrix(tt_matrix)

        self.register_buffer("weight", weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(weight.shape[0]))
        else:
            self.bias = None

    def forward(self, x):

        y = x @ self.weight.t()

        if self.bias is not None:
            y = y + self.bias

        return y