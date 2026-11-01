import os
import sys

import torch
import torch.nn as nn

from dataset.dataloader import create_dataloaders
from models.hybrid_model import DeepInterleavedResNet50ViTGNN
from training.trainer import train_one_epoch, validate

from compression.replace import replace_linear_layers
import config


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, classes = create_dataloaders(
        batch_size=config.BATCH_SIZE
    )

    model = DeepInterleavedResNet50ViTGNN()

    model = replace_linear_layers(
        model,
        config.BlockTTConfig(max_rank=16),
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS,
    )

    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda"
        else None
    )

    # ==========================================================
    # Checkpoint paths
    # ==========================================================

    CHECKPOINT_DIR = "checkpoints"

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    LATEST_CHECKPOINT = os.path.join(
        CHECKPOINT_DIR,
        "blocktt_latest.pth",
    )

    BEST_CHECKPOINT = os.path.join(
        CHECKPOINT_DIR,
        "blocktt_best.pth",
    )

    # ==========================================================
    # Resume if checkpoint exists
    # ==========================================================

    start_epoch = 0
    best_acc = 0.0

    if os.path.exists(LATEST_CHECKPOINT):

        print("\nFound checkpoint. Resuming...")

        checkpoint = torch.load(
            LATEST_CHECKPOINT,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

        if (
            scaler is not None
            and checkpoint["scaler"] is not None
        ):
            scaler.load_state_dict(
                checkpoint["scaler"]
            )

        start_epoch = checkpoint["epoch"] + 1

        best_acc = checkpoint["best_acc"]

        print(
            f"Resumed from epoch {start_epoch}"
        )

    # ==========================================================
    # Training
    # ==========================================================

    try:

        for epoch in range(
            start_epoch,
            config.EPOCHS,
        ):

            print(
                f"\nEpoch {epoch + 1}/{config.EPOCHS}"
            )

            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                grad_clip=1.0,
            )

            val_metrics = validate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            scheduler.step()

            print(
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.2f}"
            )

            print(
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.2f}"
            )

            # =====================================
            # Save latest checkpoint
            # =====================================

            checkpoint = {

                "epoch": epoch,

                "model": model.state_dict(),

                "optimizer": optimizer.state_dict(),

                "scheduler": scheduler.state_dict(),

                "scaler": (
                    scaler.state_dict()
                    if scaler is not None
                    else None
                ),

                "best_acc": best_acc,

            }

            torch.save(
                checkpoint,
                LATEST_CHECKPOINT,
            )

            # =====================================
            # Save best checkpoint
            # =====================================

            if val_metrics["accuracy"] > best_acc:

                best_acc = val_metrics["accuracy"]

                checkpoint["best_acc"] = best_acc

                torch.save(
                    checkpoint,
                    BEST_CHECKPOINT,
                )

                print("Best model saved.")

        print(
            f"\nBest Validation Accuracy: {best_acc:.2f}"
        )

    except KeyboardInterrupt:

        print("\n\nTraining interrupted.")

        checkpoint = {

            "epoch": epoch,

            "model": model.state_dict(),

            "optimizer": optimizer.state_dict(),

            "scheduler": scheduler.state_dict(),

            "scaler": (
                scaler.state_dict()
                if scaler is not None
                else None
            ),

            "best_acc": best_acc,

        }

        torch.save(
            checkpoint,
            LATEST_CHECKPOINT,
        )

        print(
            "Latest checkpoint saved."
        )

        sys.exit(0)


if __name__ == "__main__":
    main()