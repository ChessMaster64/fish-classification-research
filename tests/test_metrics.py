from evaluation.metrics import classification_metrics

y_true = [
    0,
    1,
    2,
    1,
    0,
    2,
    2,
    1,
]

y_pred = [
    0,
    1,
    2,
    0,
    0,
    1,
    2,
    1,
]

metrics = classification_metrics(
    y_true,
    y_pred,
)

print(metrics)