import torch
import torch.nn as nn
import numpy as np
import time

from torchvision import models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef
)

from data_loader import test_loader


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("✅ Device:", device)

if torch.cuda.is_available():
    print(
        "✅ GPU:",
        torch.cuda.get_device_name(0)
    )


# =========================
# MODEL
# =========================

class ResNet50_ViT(nn.Module):
    def __init__(self, num_classes=23):
        super().__init__()

        self.embed_dim = 384

        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,

            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        self.proj = nn.Linear(
            2048,
            self.embed_dim
        )

        self.pos_embed = nn.Parameter(
            torch.randn(
                1,
                49,
                self.embed_dim
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=self.embed_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=8
        )

        self.norm = nn.LayerNorm(
            self.embed_dim
        )

        self.head = nn.Linear(
            self.embed_dim,
            num_classes
        )

    def forward(self, x):

        x = self.backbone(x)

        x = x.flatten(2).transpose(1, 2)

        x = self.proj(x)

        x = x + self.pos_embed

        x = self.transformer(x)

        x = self.norm(
            x.mean(dim=1)
        )

        return self.head(x)


# =========================
# LOAD MODEL
# =========================

model = ResNet50_ViT().to(device)

model.load_state_dict(
    torch.load(
        "resnet50_vit_best.pth",
        map_location=device
    )
)

model.eval()

print("✅ ResNet50+ViT model loaded")


# =========================
# PARAMETERS
# =========================

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
print(f"Total Time: {total_time:.2f} sec")
print(
    f"Per Image: "
    f"{total_time / total_images:.6f} sec"
)


# =========================
# MAIN METRICS
# =========================

accuracy = accuracy_score(
    all_labels,
    all_preds
)

precision = precision_score(
    all_labels,
    all_preds,
    average='weighted'
)

recall = recall_score(
    all_labels,
    all_preds,
    average='weighted'
)

f1 = f1_score(
    all_labels,
    all_preds,
    average='weighted'
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


# =========================
# SPECIFICITY
# =========================

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

    specificity = TN / (TN + FP + 1e-8)

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

print("\n📊 EXTRA METRICS")
print(
    f"Specificity: "
    f"{specificity:.4f}"
)
print(
    f"MCC: "
    f"{mcc:.4f}"
)