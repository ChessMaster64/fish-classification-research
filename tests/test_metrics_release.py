from evaluation.metrics import paper_classification_metrics, specificity_score


def test_weighted_metrics_and_specificity():
    y_true = [0, 0, 0, 1, 1, 2]
    y_pred = [0, 0, 1, 1, 2, 2]
    metrics = paper_classification_metrics(y_true, y_pred, labels=[0, 1, 2])
    assert metrics["accuracy"] == 4 / 6 * 100
    assert 0 <= metrics["f1"] <= 100
    assert 0 <= metrics["specificity"] <= 100
    assert specificity_score(y_true, y_pred, labels=[0, 1, 2]) == metrics["specificity"]
