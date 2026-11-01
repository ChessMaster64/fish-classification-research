from pathlib import Path

from sklearn.metrics import classification_report


def generate_classification_report(
    y_true,
    y_pred,
    class_names=None,
    save_path=None,
):
    """
    Generate a classification report.

    Parameters
    ----------
    y_true : list
    y_pred : list
    class_names : list[str] | None
    save_path : str | Path | None

    Returns
    -------
    str
        Classification report.
    """

    report = classification_report(
        y_true,
        y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0,
    )

    if save_path is not None:

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            save_path,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(report)

    return report