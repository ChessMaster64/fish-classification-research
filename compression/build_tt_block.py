from decompositions.tensor_train.matrix_decomposition import tt_matrix_svd
from layers.tt_linear import TTLinear


def build_tt_block(
    matrix,
    input_dims,
    output_dims,
    max_rank,
):

    tt = tt_matrix_svd(
        matrix=matrix,
        input_dims=input_dims,
        output_dims=output_dims,
        max_rank=max_rank,
    )

    return TTLinear(
        tt_matrix=tt,
        bias=False,
    )