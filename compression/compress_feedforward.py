"""
compress_feedforward.py

Replace FeedForward linear layers with BlockTTLinear.
"""

from compression.linear_to_blocktt import compress_linear_to_blocktt


def compress_feedforward(
    feedforward,
    config,
):
    """
    Replace the dense linear layers inside a FeedForward block.
    """

    feedforward.fc1 = compress_linear_to_blocktt(
        feedforward.fc1,
        config,
    )

    feedforward.fc2 = compress_linear_to_blocktt(
        feedforward.fc2,
        config,
    )

    return feedforward