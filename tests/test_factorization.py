"""
Tests for factorization.py
"""

from tensorization.factorization import (
    prime_factors,
    factorize_dimension,
    factorize_matrix,
    tensor_shape,
    validate_factorization,
)


def test_prime_factors():

    assert prime_factors(8) == [2,2,2]

    assert prime_factors(12) == [2,2,3]

    assert prime_factors(25) == [5,5]


def test_factorize_dimension():

    dims = factorize_dimension(768,4)

    assert validate_factorization(768,dims)


def test_matrix():

    out_dims,in_dims = factorize_matrix((768,3072))

    assert validate_factorization(768,out_dims)

    assert validate_factorization(3072,in_dims)


def test_tensor_shape():

    shape = tensor_shape((768,3072))

    assert len(shape)==8


if __name__=="__main__":

    test_prime_factors()

    test_factorize_dimension()

    test_matrix()

    test_tensor_shape()

    print("="*50)
    print("ALL FACTORIZATION TESTS PASSED")
    print("="*50)