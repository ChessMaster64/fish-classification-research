import os
import sys

import torch
import torch.nn as nn

from dataset.dataloader import create_dataloaders
from models.hybrid_model import DeepInterleavedResNet50ViTGNN
from training.trainer import train_one_epoch, validate

import config


# ==========================================================
# PARAMETER COUNT
# ==========================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ======================================================
    # DEVICE
    # ======================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("DENSE DEEP INTERLEAVED RESNET50 + VIT + 2GNN")
    print("=" * 70)

    print(f"Using device: {device}")

    if device.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ======================================================
    # DATA
    # ======================================================

    train_loader, val_loader, test_loader, classes = (
        create_dataloaders(
            batch_size=config.BATCH_SIZE
        )
    )

    print(
        f"Number of classes: {len(classes)}"
    )

    print(
        f"Training batches: {len(train_loader)}"
    )

    print(
        f"Validation batches: {len(val_loader)}"
    )

    print(
        f"Test batches: {len(test_loader)}"
    )

    # ======================================================
    # MODEL
    # ======================================================

    print("\nCreating dense model...")

    model = DeepInterleavedResNet50ViTGNN()

    model = model.to(device)

    params = count_parameters(model)

    print(
        f"Dense Trainable Parameters: {params:,}"
    )

    # ======================================================
    # LOSS
    # ======================================================

    criterion = nn.CrossEntropyLoss()

    # ======================================================
    # OPTIMIZER
    # ======================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # ======================================================
    # SCHEDULER
    # ======================================================

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS,
    )

    # ======================================================
    # AMP
    # ======================================================

    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda"
        else None
    )

    # ======================================================
    # CHECKPOINT DIRECTORY
    # ======================================================

    CHECKPOINT_DIR = "checkpoints"

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    LATEST_CHECKPOINT = os.path.join(
        CHECKPOINT_DIR,
        "dense_latest.pth",
    )

    BEST_CHECKPOINT = os.path.join(
        CHECKPOINT_DIR,
        "dense_best.pth",
    )

    # ======================================================
    # RESUME VARIABLES
    # ======================================================

    start_epoch = 0

    best_acc = 0.0

    # ======================================================
    # RESUME
    # ======================================================

    if os.path.exists(LATEST_CHECKPOINT):

        print("\n" + "=" * 70)
        print("CHECKPOINT FOUND")
        print("=" * 70)

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
            and checkpoint.get("scaler") is not None
        ):

            scaler.load_state_dict(
                checkpoint["scaler"]
            )

        start_epoch = (
            checkpoint["epoch"] + 1
        )

        best_acc = checkpoint.get(
            "best_acc",
            0.0,
        )

        print(
            f"Resuming from epoch "
            f"{start_epoch + 1}"
        )

        print(
            f"Previous best validation accuracy: "
            f"{best_acc:.2f}%"
        )

    else:

        print(
            "\nNo checkpoint found."
        )

        print(
            "Starting dense training from epoch 1."
        )

    # ======================================================
    # NOTHING LEFT TO TRAIN
    # ======================================================

    if start_epoch >= config.EPOCHS:

        print(
            "\nTraining has already completed."
        )

        print(
            f"Best validation accuracy: "
            f"{best_acc:.2f}%"
        )

        return

    # ======================================================
    # TRAINING
    # ======================================================

    try:

        for epoch in range(
            start_epoch,
            config.EPOCHS,
        ):

            print("\n")
            print("=" * 70)
            print(
                f"EPOCH {epoch + 1}/{config.EPOCHS}"
            )
            print("=" * 70)

            # ==================================================
            # TRAIN
            # ==================================================

            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                grad_clip=1.0,
            )

            # ==================================================
            # VALIDATION
            # ==================================================

            val_metrics = validate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            # ==================================================
            # SCHEDULER
            # ==================================================

            scheduler.step()

            # ==================================================
            # PRINT METRICS
            # ==================================================

            print("\n" + "-" * 70)

            print(
                f"Train Loss : "
                f"{train_metrics['loss']:.4f}"
            )

            print(
                f"Train Acc  : "
                f"{train_metrics['accuracy']:.2f}%"
            )

            print(
                f"Val Loss   : "
                f"{val_metrics['loss']:.4f}"
            )

            print(
                f"Val Acc    : "
                f"{val_metrics['accuracy']:.2f}%"
            )

            print(
                f"Learning Rate : "
                f"{optimizer.param_groups[0]['lr']:.8f}"
            )

            # ==================================================
            # UPDATE BEST
            # ==================================================

            is_best = (
                val_metrics["accuracy"]
                > best_acc
            )

            if is_best:

                best_acc = (
                    val_metrics["accuracy"]
                )

                print(
                    "\n🔥 New best validation accuracy!"
                )

                print(
                    f"🔥 Best Val Acc: "
                    f"{best_acc:.2f}%"
                )

            # ==================================================
            # SAVE LATEST
            # ==================================================

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
                "\n💾 Latest checkpoint saved:"
            )

            print(
                LATEST_CHECKPOINT
            )

            # ==================================================
            # SAVE BEST
            # ==================================================

            if is_best:

                torch.save(
                    checkpoint,
                    BEST_CHECKPOINT,
                )

                print(
                    "💾 Best checkpoint saved:"
                )

                print(
                    BEST_CHECKPOINT
                )

    # ======================================================
    # SAFE INTERRUPTION
    # ======================================================

    except KeyboardInterrupt:

        print("\n\n")
        print("=" * 70)
        print("TRAINING INTERRUPTED")
        print("=" * 70)

        print(
            "\nSaving current checkpoint..."
        )

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
            "✅ Current checkpoint saved:"
        )

        print(
            LATEST_CHECKPOINT
        )

        print(
            "\nRun the same command again to resume."
        )

        sys.exit(0)

    # ======================================================
    # FINISHED
    # ======================================================

    print("\n")
    print("=" * 70)
    print("DENSE TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Best Validation Accuracy: "
        f"{best_acc:.2f}%"
    )

    print(
        f"\nBest checkpoint:"
    )

    print(
        BEST_CHECKPOINT
    )

    print(
        f"\nLatest checkpoint:"
    )

    print(
        LATEST_CHECKPOINT
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()