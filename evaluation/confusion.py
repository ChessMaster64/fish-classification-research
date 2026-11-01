from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


def save_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    save_path="results/confusion_matrix.png",
    figsize=(12, 10),
    dpi=300,
):
    """
    Generate and save a confusion matrix heatmap.

    Parameters
    ----------
    y_true : list
    y_pred : list
    class_names : list[str]
    save_path : str or Path
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xlabel("Predicted")

    plt.ylabel("True")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close()

    return cm