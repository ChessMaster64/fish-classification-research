import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef
)

from data_loader import test_loader

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print("✅ Device:", device)

if torch.cuda.is_available():
    print(
        "✅ GPU:",
        torch.cuda.get_device_name(0)
    )


class MiddleInterleavedResNet50ViT(nn.Module):

    def __init__(self, num_classes=23):
        super().__init__()

        self.embed_dim = 384

        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.proj_in = nn.Linear(
            512,
            self.embed_dim
        )

        self.pos_embed = nn.Parameter(
            torch.randn(
                1,
                784,
                self.embed_dim
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=self.embed_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=8
        )

        self.proj_out = nn.Linear(
            self.embed_dim,
            512
        )

        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.head = nn.Linear(
            2048,
            num_classes
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)

        x = self.layer2(x)

        B, C, H, W = x.shape

        x = x.flatten(2)

        x = x.transpose(1, 2)

        x = self.proj_in(x)

        x = x + self.pos_embed

        x = self.transformer(x)

        x = self.proj_out(x)

        x = x.transpose(1, 2)

        x = x.reshape(
            B,
            512,
            H,
            W
        )

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.head(x)

        return x


model = MiddleInterleavedResNet50ViT().to(device)

model.load_state_dict(
    torch.load(
        "resnet50_vit_middleinterleaved_best.pth",
        map_location=device
    )
)

model.eval()

print(
    "✅ Middle Interleaved ResNet50+ViT model loaded"
)

params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\n📊 PARAMETERS")
print(f"{params:,}")
# =========================
# EVALUATION
# =========================

all_preds = []
all_labels = []

total_images = 0

start = time.time()

with torch.no_grad():

    for imgs, labels in test_loader:

        imgs = imgs.to(device)

        labels = labels.to(device)

        outputs = model(imgs)

        preds = outputs.argmax(1)

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        total_images += labels.size(0)

end = time.time()

total_time = end - start

print("\n⏱️ TIME")
print(
    f"Total Time: {total_time:.2f} sec"
)

print(
    f"Per Image: "
    f"{total_time / total_images:.6f} sec"
)

accuracy = accuracy_score(
    all_labels,
    all_preds
)

precision = precision_score(
    all_labels,
    all_preds,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_preds,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_preds,
    average="weighted",
    zero_division=0
)

cm = confusion_matrix(
    all_labels,
    all_preds
)

specificities = []

for i in range(len(cm)):

    TP = cm[i, i]

    FN = cm[i, :].sum() - TP

    FP = cm[:, i].sum() - TP

    TN = cm.sum() - TP - FN - FP

    specificity = (
        TN /
        (TN + FP + 1e-8)
    )

    specificities.append(
        specificity
    )

specificity = np.mean(
    specificities
)

mcc = matthews_corrcoef(
    all_labels,
    all_preds
)

print("\n📊 METRICS")

print(
    f"Accuracy: "
    f"{accuracy*100:.2f}%"
)

print(
    f"Precision: "
    f"{precision:.4f}"
)

print(
    f"Recall: "
    f"{recall:.4f}"
)

print(
    f"F1 Score: "
    f"{f1:.4f}"
)

print("\n📊 EXTRA METRICS")

print(
    f"Specificity: "
    f"{specificity:.4f}"
)

print(
    f"MCC: "
    f"{mcc:.4f}"
)

print(
    f"FPS: "
    f"{total_images / total_time:.2f}"
)
# =========================
# CLASSIFICATION REPORT
# =========================

class_names = (
    test_loader.dataset.dataset.classes
)

if class_names is None:

    class_names = [
        f"Class_{i}"
        for i in range(len(cm))
    ]

print(
    "\n📋 CLASSIFICATION REPORT\n"
)

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# =========================
# CONFUSION MATRIX
# =========================

plt.figure(
    figsize=(16, 14)
)

sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title(
    "Middle Interleaved ResNet50+ViT"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(
    "middleinterleaved_confusion_matrix.png",
    dpi=300
)

print(
    "\n✅ Saved: middleinterleaved_confusion_matrix.png"
)