import torch


def create_optimizer(
    model,
    lr=3e-4,
    weight_decay=1e-4,
):
    """
    Creates the optimizer.
    """

    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )


def create_scheduler(
    optimizer,
    epochs,
    eta_min=1e-6,
):
    """
    Cosine Annealing learning rate scheduler.
    """

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=eta_min,
    )