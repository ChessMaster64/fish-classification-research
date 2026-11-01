# eval_resnet50_vit_deepinterleaved_gnn.py

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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("✅ Device:", device)
if torch.cuda.is_available():
    print("✅ GPU:", torch.cuda.get_device_name(0))


class DynamicGNNLayer(nn.Module):

    def __init__(self, dim, k=6):
        super().__init__()

        self.k = k

        self.edge_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        B, N, D = x.shape

        dist = torch.cdist(x, x)

        knn_idx = dist.topk(
            k=self.k,
            largest=False
        ).indices

        idx_base = torch.arange(
            0, B, device=x.device
        ).view(-1, 1, 1) * N

        knn_idx = knn_idx + idx_base

        neighbors = x.reshape(
            B * N, D
        )[knn_idx.reshape(-1)].reshape(
            B, N, self.k, D
        )

        x_expanded = x.unsqueeze(2).expand(
            -1, -1, self.k, -1
        )

        edge_feat = torch.cat(
            [x_expanded, neighbors - x_expanded],
            dim=-1
        )

        out = self.edge_mlp(edge_feat)

        out = out.max(dim=2)[0]

        return self.norm(x + out)


class DeepInterleavedResNet50ViTGNN(nn.Module):

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

        self.proj_in = nn.Linear(1024, self.embed_dim)

        self.pos_embed = nn.Parameter(
            torch.randn(1, 196, self.embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=self.embed_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )

        self.transformer_1 = nn.TransformerEncoder(
            encoder_layer, num_layers=3
        )

        self.transformer_2 = nn.TransformerEncoder(
            encoder_layer, num_layers=3
        )

        self.transformer_3 = nn.TransformerEncoder(
            encoder_layer, num_layers=2
        )

        self.gnn1 = DynamicGNNLayer(
            dim=self.embed_dim,
            k=6
        )

        self.gnn2 = DynamicGNNLayer(
            dim=self.embed_dim,
            k=6
        )

        self.proj_out = nn.Linear(
            self.embed_dim,
            1024
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Linear(
            2048,
            num_classes
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)

        x = self.proj_in(x)

        x = x + self.pos_embed

        x = self.transformer_1(x)

        identity = x
        x = self.gnn1(x)
        x = x + identity

        x = self.transformer_2(x)

        identity = x
        x = self.gnn2(x)
        x = x + identity

        x = self.transformer_3(x)

        x = self.proj_out(x)

        x = x.transpose(1, 2).reshape(
            B, 1024, H, W
        )

        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.head(x)

        return x


model = DeepInterleavedResNet50ViTGNN().to(device)

model.load_state_dict(
    torch.load(
        "resnet50_vit_deepinterleaved_gnn_best.pth",
        map_location=device
    )
)

model.eval()

params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"✅ Trainable Parameters: {params:,}")

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

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        total_images += labels.size(0)

end = time.time()

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

cm = confusion_matrix(all_labels, all_preds)

specificities = []

for i in range(len(cm)):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = cm.sum() - TP - FN - FP

    specificities.append(
        TN / (TN + FP + 1e-8)
    )

specificity = np.mean(specificities)

mcc = matthews_corrcoef(
    all_labels,
    all_preds
)

class_names = getattr(
    test_loader.dataset,
    "classes",
    [f"Class_{i}" for i in range(len(cm))]
)

print("\n===== RESULTS =====")
print(f"Accuracy     : {accuracy*100:.2f}%")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"Specificity  : {specificity:.4f}")
print(f"MCC          : {mcc:.4f}")
print(f"Parameters   : {params:,}")
print(f"FPS          : {total_images/(end-start):.2f}")

print("\n===== CLASSIFICATION REPORT =====\n")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

plt.figure(figsize=(16, 14))

sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

print("Saved confusion_matrix.png")
