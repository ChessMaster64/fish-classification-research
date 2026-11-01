import copy
import torch.nn as nn

from compression.linear_to_blocktt import compress_linear_to_blocktt


def replace_linear_layers(
    module,
    blocktt_config,
    verbose=True,
):
    """
    Recursively replace every eligible nn.Linear
    with BlockTTLinear.

    The final classifier is intentionally kept dense
    because its output dimension (23 classes) is not
    suitable for BlockTT tensorization and contributes
    negligibly to the total parameter count.
    """

    module = copy.deepcopy(module)

    _replace_recursive(
        module,
        blocktt_config,
        verbose,
    )

    return module


def _replace_recursive(
    module,
    blocktt_config,
    verbose,
):

    for name, child in module.named_children():

        if isinstance(child, nn.Linear):

            # Keep the final classifier dense
            if name == "head":

                if verbose:
                    print(
                        f"Skipping {name}: "
                        f"{child.in_features} -> {child.out_features}"
                    )

                continue

            if verbose:
                print(
                    f"Replacing {name}: "
                    f"{child.in_features} -> {child.out_features}"
                )

            new_layer = compress_linear_to_blocktt(
                child,
                blocktt_config,
            )

            setattr(
                module,
                name,
                new_layer,
            )

        else:

            _replace_recursive(
                child,
                blocktt_config,
                verbose,
            )