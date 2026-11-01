from evaluation.confusion import save_confusion_matrix

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

cm = save_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    save_path="results/test_confusion.png",
)

print(cm)