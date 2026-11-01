# FILE: resnet50_vit.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import numpy as np
from collections import Counter

from data_loader import train_loader, val_loader

device = torch.device("cuda:0")
print("✅ Using GPU:", torch.cuda.get_device_name(0))


# =========================
# MIXUP
# =========================

def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


# =========================
# MODEL
# =========================

class ResNet50_ViT(nn.Module):
    def __init__(self, num_classes=23):
        super().__init__()

        self.embed_dim = 384

        # ---------------------
        # ResNet50 Backbone
        # ---------------------

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

        # ---------------------
        # Projection
        # ---------------------

        self.proj = nn.Linear(
            2048,
            self.embed_dim
        )

        # ---------------------
        # Position Embedding
        # ---------------------

        self.pos_embed = nn.Parameter(
            torch.randn(
                1,
                49,
                self.embed_dim
            )
        )

        # ---------------------
        # Transformer
        # ---------------------

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

        # ---------------------
        # Classification Head
        # ---------------------

        self.norm = nn.LayerNorm(
            self.embed_dim
        )

        self.head = nn.Linear(
            self.embed_dim,
            num_classes
        )

    def forward(self, x):

        x = self.backbone(x)

        # B x 2048 x 7 x 7

        x = x.flatten(2)

        x = x.transpose(1, 2)

        x = x.contiguous()

        x = self.proj(x)

        # B x 49 x 384

        x = x + self.pos_embed

        x = self.transformer(x)

        x = self.norm(
            x.mean(dim=1)
        )

        return self.head(x)


model = ResNet50_ViT().to(device)

params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"✅ Trainable Parameters: {params:,}"
)


# =========================
# CLASS WEIGHTS
# =========================

counts = Counter()
for _, labels in train_loader:
    counts.update(labels.numpy())

num_classes = len(counts)

weights = torch.tensor([
    1.0 / np.sqrt(counts[i])
    for i in range(num_classes)
], dtype=torch.float32)

weights = weights / weights.sum() * num_classes
weights = weights.to(device)

criterion = nn.CrossEntropyLoss(
    weight=weights,
    label_smoothing=0.1
)


# =========================
# OPTIMIZER
# =========================

optimizer = optim.AdamW(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=80
)

EPOCHS = 80
best_val_acc = 0


# =========================
# TRAIN LOOP
# =========================

if __name__ == "__main__":

    for epoch in range(EPOCHS):

        model.train()

        train_correct = 0
        total = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}"
        )

        for imgs, labels in pbar:

            imgs = imgs.to(device)
            labels = labels.to(device)

            if np.random.rand() < 0.5:
                imgs, ya, yb, lam = mixup_data(
                    imgs,
                    labels
                )
                use_mixup = True
            else:
                ya, yb = labels, labels
                lam = 1.0
                use_mixup = False

            optimizer.zero_grad()

            outputs = model(imgs)

            loss = (
                lam * criterion(outputs, ya)
                + (1 - lam) * criterion(outputs, yb)
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            preds = outputs.argmax(1)

            if use_mixup:
                dominant = ya if lam > 0.5 else yb
            else:
                dominant = labels

            train_correct += (
                preds == dominant
            ).sum().item()

            total += labels.size(0)

            pbar.set_postfix(
                loss=loss.item()
            )

        train_acc = (
            100 * train_correct / total
        )

        # =====================
        # VALIDATION
        # =====================

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for imgs, labels in val_loader:

                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)

                preds = outputs.argmax(1)

                val_correct += (
                    preds == labels
                ).sum().item()

                val_total += labels.size(0)

        val_acc = (
            100 * val_correct / val_total
        )

        scheduler.step()

        print(f"\nEpoch {epoch+1}")
        print(f"Train Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                "resnet50_vit_best.pth"
            )

            print("💾 Best model saved!")

print(
    f"\n🔥 RESNET50+VIT BEST VAL ACC: "
    f"{best_val_acc:.2f}%"
)