"""
compress_transformer_encoder.py
"""

from compression.compress_encoder_layer import compress_encoder_layer


def compress_transformer_encoder(
    encoder,
    config,
):
    """
    Compress every TransformerEncoderLayer in the encoder.
    """

    for i in range(len(encoder.layers)):
        encoder.layers[i] = compress_encoder_layer(
            encoder.layers[i],
            config,
        )

    return encoder