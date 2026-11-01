import torch


def print_model_statistics(model):

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    frozen_params = total_params - trainable_params

    model_size = (
        total_params * 4
    ) / (1024 ** 2)

    print()

    print("=" * 40)
    print("Model Statistics")
    print("=" * 40)

    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Frozen Parameters    : {frozen_params:,}")
    print(f"Estimated Size       : {model_size:.2f} MB")

    print("=" * 40)

    print()