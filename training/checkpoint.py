from pathlib import Path

import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_accuracy,
):
    """
    Save training checkpoint.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
            "best_accuracy": best_accuracy,
        },
        path,
    )


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
):
    """
    Load training checkpoint.
    """

    checkpoint = torch.load(
        path,
        map_location=map_location,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if (
        scheduler is not None
        and checkpoint["scheduler_state_dict"] is not None
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    return (
        checkpoint["epoch"],
        checkpoint["best_accuracy"],
    )