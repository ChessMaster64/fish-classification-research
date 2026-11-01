from evaluation.report import generate_classification_report

y_true = [
    0,0,1,1,2,2,0,1,2,0
]

y_pred = [
    0,1,1,1,2,0,0,2,2,0
]

class_names = [
    "Class0",
    "Class1",
    "Class2",
]

report = generate_classification_report(
    y_true,
    y_pred,
    class_names,
    save_path="results/classification_report.txt",
)

print(report)