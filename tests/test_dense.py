import os
import time

import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    balanced_accuracy_score,
    cohen_kappa_score,
)

from dataset.dataloader import create_dataloaders
from models.hybrid_model import DeepInterleavedResNet50ViTGNN

from evaluation.confusion import save_confusion_matrix
from evaluation.report import generate_classification_report

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
# MODEL SIZE
# ==========================================================

def model_size_mb(model):

    temp_file = "temp_dense_model.pth"

    torch.save(
        model.state_dict(),
        temp_file,
    )

    size = (
        os.path.getsize(temp_file)
        / (1024 ** 2)
    )

    os.remove(temp_file)

    return size


# ==========================================================
# SPECIFICITY
# ==========================================================

def calculate_specificity(
    y_true,
    y_pred,
):

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    specificities = []

    for i in range(len(cm)):

        TP = cm[i, i]

        FN = (
            cm[i, :].sum()
            - TP
        )

        FP = (
            cm[:, i].sum()
            - TP
        )

        TN = (
            cm.sum()
            - TP
            - FN
            - FP
        )

        specificity = (
            TN
            / (TN + FP + 1e-8)
        )

        specificities.append(
            specificity
        )

    return np.mean(
        specificities
    )


# ==========================================================
# INFERENCE
# ==========================================================

def run_inference(
    model,
    loader,
    device,
):

    model.eval()

    all_preds = []
    all_labels = []

    total_images = 0

    if device.type == "cuda":

        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            preds = outputs.argmax(
                dim=1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            total_images += (
                labels.size(0)
            )

    if device.type == "cuda":

        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        np.array(all_labels),
        np.array(all_preds),
        elapsed,
        total_images,
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

    print(
        f"Using device: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ======================================================
    # DATA
    # ======================================================

    (
        _,
        _,
        test_loader,
        class_names,
    ) = create_dataloaders(
        batch_size=config.BATCH_SIZE
    )

    # ======================================================
    # MODEL
    # ======================================================

    print(
        "\nLoading dense model..."
    )

    model = (
        DeepInterleavedResNet50ViTGNN()
    )

    # IMPORTANT:
    # No BlockTT replacement here.
    #
    # The model therefore contains
    # normal nn.Linear layers.

    model = model.to(device)

    # ======================================================
    # LOAD BEST CHECKPOINT
    # ======================================================

    checkpoint_path = (
        "checkpoints/dense_best.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # The training script saved a
    # complete training checkpoint.

    model.load_state_dict(
        checkpoint["model"]
    )

    best_val_acc = checkpoint.get(
        "best_acc",
        None,
    )

    model.eval()

    print(
        "Dense model loaded successfully."
    )

    if best_val_acc is not None:

        print(
            f"Best Validation Accuracy: "
            f"{best_val_acc:.2f}%"
        )

    # ======================================================
    # PARAMETERS
    # ======================================================

    params = count_parameters(
        model
    )

    # ======================================================
    # MODEL SIZE
    # ======================================================

    size = model_size_mb(
        model
    )

    # ======================================================
    # INFERENCE
    # ======================================================

    print(
        "\nRunning inference..."
    )

    (
        y_true,
        y_pred,
        total_time,
        total_images,
    ) = run_inference(
        model,
        test_loader,
        device,
    )

    # ======================================================
    # CLASSIFICATION METRICS
    # ======================================================

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    precision_macro = (
        precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    precision_weighted = (
        precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    recall_macro = (
        recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    recall_weighted = (
        recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    f1_macro = (
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    f1_weighted = (
        f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    specificity = (
        calculate_specificity(
            y_true,
            y_pred,
        )
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred,
    )

    kappa = cohen_kappa_score(
        y_true,
        y_pred,
    )

    # ======================================================
    # LATENCY
    # ======================================================

    latency = (
        total_time
        / total_images
    ) * 1000

    fps = (
        total_images
        / total_time
    )

    # ======================================================
    # OUTPUT DIRECTORY
    # ======================================================

    os.makedirs(
        "results",
        exist_ok=True,
    )

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    save_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        save_path=(
            "results/"
            "dense_confusion_matrix.png"
        ),
    )

    # ======================================================
    # CLASSIFICATION REPORT
    # ======================================================

    generate_classification_report(
        y_true,
        y_pred,
        class_names,
        save_path=(
            "results/"
            "dense_classification_report.txt"
        ),
    )

    # ======================================================
    # PRINT RESULTS
    # ======================================================

    print("\n")
    print("=" * 60)
    print(
        "       DENSE MODEL EVALUATION RESULTS"
    )
    print("=" * 60)

    # ======================================================
    # CLASSIFICATION
    # ======================================================

    print("\n📊 CLASSIFICATION METRICS")
    print("-" * 60)

    print(
        f"Accuracy                 : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy        : "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Precision (Macro)        : "
        f"{precision_macro * 100:.2f}%"
    )

    print(
        f"Precision (Weighted)     : "
        f"{precision_weighted * 100:.2f}%"
    )

    print(
        f"Recall (Macro)           : "
        f"{recall_macro * 100:.2f}%"
    )

    print(
        f"Recall (Weighted)        : "
        f"{recall_weighted * 100:.2f}%"
    )

    print(
        f"F1 Score (Macro)         : "
        f"{f1_macro * 100:.2f}%"
    )

    print(
        f"F1 Score (Weighted)      : "
        f"{f1_weighted * 100:.2f}%"
    )

    print(
        f"Specificity              : "
        f"{specificity * 100:.2f}%"
    )

    print(
        f"Matthews Corr. Coef.     : "
        f"{mcc:.4f}"
    )

    print(
        f"Cohen's Kappa            : "
        f"{kappa:.4f}"
    )

    # ======================================================
    # MODEL COMPLEXITY
    # ======================================================

    print("\n🧠 MODEL COMPLEXITY")
    print("-" * 60)

    print(
        f"Trainable Parameters     : "
        f"{params:,}"
    )

    print(
        f"Model Size               : "
        f"{size:.2f} MB"
    )

    # ======================================================
    # INFERENCE PERFORMANCE
    # ======================================================

    print("\n⚡ INFERENCE PERFORMANCE")
    print("-" * 60)

    print(
        f"Images Evaluated         : "
        f"{total_images}"
    )

    print(
        f"Total Inference Time     : "
        f"{total_time:.2f} sec"
    )

    print(
        f"Average Time / Image     : "
        f"{total_time / total_images:.6f} sec"
    )

    print(
        f"Latency                  : "
        f"{latency:.3f} ms/image"
    )

    print(
        f"FPS                      : "
        f"{fps:.2f}"
    )

    # ======================================================
    # OUTPUTS
    # ======================================================

    print("\n📁 OUTPUT FILES")
    print("-" * 60)

    print(
        "✓ results/"
        "dense_confusion_matrix.png"
    )

    print(
        "✓ results/"
        "dense_classification_report.txt"
    )

    print("\n")
    print("=" * 60)
    print(
        "Dense evaluation completed successfully."
    )
    print("=" * 60)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()