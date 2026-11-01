import os
import time

import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
)

from dataset.dataloader import create_dataloaders

from models.hybrid_model import (
    DeepInterleavedResNet50ViTGNN,
)

from compression.replace import (
    replace_linear_layers,
)

from evaluation.inference import predict
from evaluation.confusion import save_confusion_matrix
from evaluation.report import generate_classification_report

import config


# ==========================================================
# Utility Functions
# ==========================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def model_size_mb(model):

    torch.save(
        model.state_dict(),
        "temp_model.pth",
    )

    size = (
        os.path.getsize("temp_model.pth")
        / (1024 ** 2)
    )

    os.remove("temp_model.pth")

    return size


def measure_latency(
    model,
    loader,
    device,
):

    model.eval()

    if device.type == "cuda":
        torch.cuda.synchronize()

    starter = time.perf_counter()

    total_images = 0

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(device)

            _ = model(images)

            total_images += images.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - starter

    latency = (
        elapsed / total_images
    ) * 1000

    fps = total_images / elapsed

    return (
        elapsed,
        latency,
        fps,
        total_images,
    )


# ==========================================================
# Main
# ==========================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    (
        _,
        _,
        test_loader,
        class_names,
    ) = create_dataloaders(
        batch_size=config.BATCH_SIZE
    )

    # ------------------------------------------------------
    # Original Model
    # ------------------------------------------------------

    original_model = (
        DeepInterleavedResNet50ViTGNN()
    )

    original_params = count_parameters(
        original_model
    )

    # ------------------------------------------------------
    # BlockTT Compression
    # ------------------------------------------------------

    model = replace_linear_layers(
        original_model,
        config.BlockTTConfig(
            max_rank=16,
        ),
    )

    compressed_params = count_parameters(
        model
    )

    parameter_reduction = (
        original_params
        - compressed_params
    )

    reduction_percent = (
        parameter_reduction
        / original_params
    ) * 100

    compression_ratio = (
        original_params
        / compressed_params
    )

    checkpoint = torch.load(
        "checkpoints/blocktt_best.pth",
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.to(device)

    model.eval()

    print("\nRunning inference...")

        # ------------------------------------------------------
    # Inference
    # ------------------------------------------------------

    start = time.time()

    y_true, y_pred = predict(
        model,
        test_loader,
        device,
    )

    end = time.time()

    total_time = end - start

    total_images = len(y_true)

    # ------------------------------------------------------
    # Classification Metrics
    # ------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    precision_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    precision_weighted = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    recall_weighted = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    # ------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    specificities = []

    for i in range(len(cm)):

        TP = cm[i, i]

        FN = cm[i, :].sum() - TP

        FP = cm[:, i].sum() - TP

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

    specificity = np.mean(
        specificities
    )

    # ------------------------------------------------------
    # MCC & Cohen Kappa
    # ------------------------------------------------------

    mcc = matthews_corrcoef(
        y_true,
        y_pred,
    )

    kappa = cohen_kappa_score(
        y_true,
        y_pred,
    )

    # ------------------------------------------------------
    # Model Statistics
    # ------------------------------------------------------

    model_size = model_size_mb(
        model
    )

    (
        inference_time,
        latency,
        fps,
        _,
    ) = measure_latency(
        model,
        test_loader,
        device,
    )

    # ------------------------------------------------------
    # Save Outputs
    # ------------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True,
    )

    save_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        save_path="results/confusion_matrix.png",
    )

    generate_classification_report(
        y_true,
        y_pred,
        class_names,
        save_path="results/classification_report.txt",
    )

        # ------------------------------------------------------
    # Final Results
    # ------------------------------------------------------

    print("\n==========================================================")
    print("         BLOCKTT MODEL EVALUATION RESULTS")
    print("==========================================================")

    print("\n📊 CLASSIFICATION METRICS")
    print("----------------------------------------------------------")
    print(f"Accuracy                 : {accuracy*100:.2f}%")
    print(f"Balanced Accuracy        : {balanced_accuracy*100:.2f}%")
    print(f"Precision (Macro)        : {precision_macro*100:.2f}%")
    print(f"Precision (Weighted)     : {precision_weighted*100:.2f}%")
    print(f"Recall (Macro)           : {recall_macro*100:.2f}%")
    print(f"Recall (Weighted)        : {recall_weighted*100:.2f}%")
    print(f"F1 Score (Macro)         : {f1_macro*100:.2f}%")
    print(f"F1 Score (Weighted)      : {f1_weighted*100:.2f}%")
    print(f"Specificity              : {specificity*100:.2f}%")
    print(f"Matthews Corr. Coef.     : {mcc:.4f}")
    print(f"Cohen's Kappa            : {kappa:.4f}")

    print("\n🧠 MODEL COMPRESSION")
    print("----------------------------------------------------------")
    print(f"Original Parameters      : {original_params:,}")
    print(f"Compressed Parameters    : {compressed_params:,}")
    print(f"Parameters Reduced       : {parameter_reduction:,}")
    print(f"Parameter Reduction      : {reduction_percent:.2f}%")
    print(f"Compression Ratio        : {compression_ratio:.2f}x")
    print(f"Model Size              : {model_size:.2f} MB")

    print("\n⚡ INFERENCE PERFORMANCE")
    print("----------------------------------------------------------")
    print(f"Images Evaluated         : {total_images}")
    print(f"Total Inference Time     : {total_time:.2f} sec")
    print(f"Average Time / Image     : {total_time/total_images:.6f} sec")
    print(f"Latency                 : {latency:.3f} ms/image")
    print(f"FPS                     : {fps:.2f}")

    print("\n📁 OUTPUT FILES")
    print("----------------------------------------------------------")
    print("✓ results/confusion_matrix.png")
    print("✓ results/classification_report.txt")

    print("\n==========================================================")
    print("Evaluation Completed Successfully")
    print("==========================================================")

if __name__ == "__main__":
    main()