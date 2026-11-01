from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

def specificity_score(y_true, y_pred, labels=None):
    """Return macro one-vs-rest specificity in percentage points."""
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    total = matrix.sum()
    values = []
    for index in range(len(labels)):
        true_positive = matrix[index, index]
        false_negative = matrix[index, :].sum() - true_positive
        false_positive = matrix[:, index].sum() - true_positive
        true_negative = total - true_positive - false_negative - false_positive
        denominator = true_negative + false_positive
        values.append(true_negative / denominator if denominator else 0.0)
    return sum(values) / len(values) * 100 if values else 0.0


def classification_metrics(
    y_true,
    y_pred,
):

    return {

        "accuracy":
        accuracy_score(
            y_true,
            y_pred,
        ) * 100,

        "precision":
        precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ) * 100,

        "recall":
        recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ) * 100,

        "f1":
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ) * 100,

        "mcc":
        matthews_corrcoef(
            y_true,
            y_pred,
        ),

    }


def paper_classification_metrics(y_true, y_pred, labels=None):
    """Calculate the paper-facing weighted metrics and macro specificity."""
    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "precision": precision_score(
            y_true, y_pred, average="weighted", labels=labels, zero_division=0
        ) * 100,
        "recall": recall_score(
            y_true, y_pred, average="weighted", labels=labels, zero_division=0
        ) * 100,
        "f1": f1_score(
            y_true, y_pred, average="weighted", labels=labels, zero_division=0
        ) * 100,
        "specificity": specificity_score(y_true, y_pred, labels=labels),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
