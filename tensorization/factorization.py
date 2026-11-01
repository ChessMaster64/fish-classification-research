"""
factorization.py

Utilities for automatically factorizing matrix dimensions into tensor shapes.

These utilities are decomposition-independent and are shared by
Tensor Train (TT), Tensor Ring (TR), Matrix Product Operator (MPO),
and Block Tensor Train (BlockTT).

Author: Research Framework v1
"""

from math import prod
from typing import List, Tuple


def prime_factors(n: int) -> List[int]:
    """
    Compute the prime factorization of an integer.

    Example:
        768 -> [2,2,2,2,2,2,2,2,3]
    """

    if n <= 0:
        raise ValueError("Dimension must be positive.")

    factors = []

    while n % 2 == 0:
        factors.append(2)
        n //= 2

    divisor = 3

    while divisor * divisor <= n:

        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor

        divisor += 2

    if n > 1:
        factors.append(n)

    return factors


def balanced_factorization(n: int, order: int = 4) -> List[int]:
    """
    Factorize an integer into 'order' approximately balanced factors.

    Example:
        balanced_factorization(768,4)
        -> [4,4,6,8]

    Product is always preserved.
    """

    if order <= 0:
        raise ValueError("Order must be positive.")

    primes = sorted(prime_factors(n), reverse=True)

    bins = [1] * order

    for p in primes:

        idx = bins.index(min(bins))
        bins[idx] *= p

    return sorted(bins)


def factorize_dimension(
    dim: int,
    order: int = 4
) -> List[int]:
    """
    Automatically factorize one dimension.

    Parameters
    ----------
    dim : int
        Matrix dimension.

    order : int
        Desired tensor order.

    Returns
    -------
    list
        Tensor dimensions.
    """

    factors = balanced_factorization(dim, order)

    if prod(factors) != dim:
        raise RuntimeError("Factorization failed.")

    return factors


def factorize_matrix(
    shape: Tuple[int, int],
    input_order: int = 4,
    output_order: int = 4,
):
    """
    Factorize a matrix shape into tensor dimensions.

    Example

    (768,3072)

    →

    input_dims = [4,4,6,8]
    output_dims = [4,8,8,24]
    """

    out_dim, in_dim = shape

    input_dims = factorize_dimension(in_dim, input_order)
    output_dims = factorize_dimension(out_dim, output_order)

    return output_dims, input_dims


def validate_factorization(
    original: int,
    factors: List[int]
) -> bool:
    """
    Verify that factors reconstruct original dimension.
    """

    return prod(factors) == original


def tensor_shape(
    matrix_shape: Tuple[int, int],
    input_order: int = 4,
    output_order: int = 4,
):
    """
    Return full tensor shape.

    Example

    Matrix:

        (768,3072)

    →

    Tensor:

        (4,4,6,8,4,8,8,24)
    """

    out_dims, in_dims = factorize_matrix(
        matrix_shape,
        input_order,
        output_order,
    )

    return tuple(out_dims + in_dims)