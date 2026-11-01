def infer_tensor_shape(size):

    lookup = {
        192: (8,8,3),
        384: (8,8,6),
        512: (8,8,8),
        768: (8,8,12),
        1024:(8,8,16),
        1536:(8,8,24),
    }

    if size not in lookup:
        raise ValueError(
            f"No tensorization defined for {size}"
        )

    return lookup[size]