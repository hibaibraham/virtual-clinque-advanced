"""
Brain Tumor Classifier - Training Script
Classes: glioma | meningioma | pituitary | notumor
Model: EfficientNet-B0 (Transfer Learning) - PyTorch
"""

import os
import time
import copy
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES  = len(CLASS_NAMES)
IMG_SIZE     = 224
BATCH_SIZE   = 32
NUM_EPOCHS   = 30
LR_HEAD      = 1e-3    # Phase 1 : tête seulement
LR_FINETUNE  = 1e-4    # Phase 2 : tout le modèle
PHASE1_EPOCHS = 10     # Epochs pour entraîner la tête
DROPOUT      = 0.4
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Dispositif : {device}")


# ─────────────────────────────────────────────
# TRANSFORMATIONS
# ─────────────────────────────────────────────
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
def load_datasets(data_dir: str):
    """
    Structure attendue :
    data_dir/
      Training/
        glioma/
        meningioma/
        notumor/
        pituitary/
      Testing/
        glioma/
        ...
    """
    train_tf, val_tf = get_transforms()
    train_dir = os.path.join(data_dir, "Training")
    test_dir  = os.path.join(data_dir, "Testing")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    test_ds  = datasets.ImageFolder(test_dir,  transform=val_tf)

    # Split test → val + test (50/50)
    val_size  = len(test_ds) // 2
    test_size = len(test_ds) - val_size
    val_ds, test_ds_final = torch.utils.data.random_split(
        test_ds, [val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    loaders = {
        "train": DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True),
        "val":   DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True),
        "test":  DataLoader(test_ds_final, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True),
    }
    sizes = {
        "train": len(train_ds),
        "val":   val_size,
        "test":  test_size,
    }
    print(f"📦 Train: {sizes['train']} | Val: {sizes['val']} | Test: {sizes['test']}")
    print(f"🏷️  Classes: {train_ds.classes}")
    return loaders, sizes, train_ds.classes


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_model(num_classes=NUM_CLASSES, freeze_base=True):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # Remplacer le classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model.to(device)


def unfreeze_model(model):
    """Débloquer tout le modèle pour le fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    return model


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct  += preds.eq(labels).sum().item()
        total    += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss    = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, preds = probs.max(1)
        correct  += preds.eq(labels).sum().item()
        total    += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (running_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


# ─────────────────────────────────────────────
# TRAINING (2 phases)
# ─────────────────────────────────────────────
def train(data_dir: str, save_dir: str = "output"):
    os.makedirs(save_dir, exist_ok=True)
    loaders, sizes, classes = load_datasets(data_dir)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {"train_loss": [], "val_loss": [],
               "train_acc": [],  "val_acc":  []}

    # ── Phase 1 : tête seulement ──────────────────
    print("\n🔒 Phase 1 — Entraînement de la tête (base gelée)...")
    model = build_model(freeze_base=True)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS)
    best_acc, best_weights = 0.0, None

    for epoch in range(PHASE1_EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, loaders["train"], criterion, optimizer)
        vl_loss, vl_acc, _, _, _ = eval_epoch(model, loaders["val"], criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_acc:
            best_acc = vl_acc
            best_weights = copy.deepcopy(model.state_dict())

        print(f"  Epoch {epoch+1:2d}/{PHASE1_EPOCHS} | "
              f"Train {tr_acc:.3f} ({tr_loss:.4f}) | "
              f"Val {vl_acc:.3f} ({vl_loss:.4f}) | "
              f"{time.time()-t0:.1f}s")

    model.load_state_dict(best_weights)

    # ── Phase 2 : fine-tuning complet ─────────────
    remaining = NUM_EPOCHS - PHASE1_EPOCHS
    print(f"\n🔓 Phase 2 — Fine-tuning complet ({remaining} epochs)...")
    model = unfreeze_model(model)
    optimizer = optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=remaining)

    for epoch in range(remaining):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, loaders["train"], criterion, optimizer)
        vl_loss, vl_acc, _, _, _ = eval_epoch(model, loaders["val"], criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_acc:
            best_acc = vl_acc
            best_weights = copy.deepcopy(model.state_dict())

        print(f"  Epoch {PHASE1_EPOCHS+epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train {tr_acc:.3f} ({tr_loss:.4f}) | "
              f"Val {vl_acc:.3f} ({vl_loss:.4f}) | "
              f"{time.time()-t0:.1f}s")

    model.load_state_dict(best_weights)
    print(f"\n🏆 Meilleure val acc : {best_acc:.4f}")

    # ── Sauvegarde ─────────────────────────────────
    ckpt_path = os.path.join(save_dir, "best_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": classes,
        "val_acc": best_acc,
        "config": {
            "img_size": IMG_SIZE,
            "num_classes": NUM_CLASSES,
            "dropout": DROPOUT,
        }
    }, ckpt_path)
    print(f"💾 Modèle sauvegardé → {ckpt_path}")

    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f)

    # ── Évaluation finale sur test ─────────────────
    evaluate(model, loaders["test"], classes, save_dir)

    return model, history


# ─────────────────────────────────────────────
# ÉVALUATION & VISUALISATIONS
# ─────────────────────────────────────────────
def evaluate(model, test_loader, classes, save_dir):
    criterion = nn.CrossEntropyLoss()
    _, acc, preds, labels, probs = eval_epoch(model, test_loader, criterion)

    print(f"\n📊 Test Accuracy : {acc:.4f}")
    print("\n" + classification_report(labels, preds, target_names=classes))

    plot_confusion_matrix(labels, preds, classes, save_dir)
    plot_roc_curves(labels, probs, classes, save_dir)
    plot_training_history_from_file(save_dir)


def plot_confusion_matrix(labels, preds, classes, save_dir):
    cm = confusion_matrix(labels, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Brain Tumor Classifier — Matrice de confusion", fontsize=14, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_pct],
        ["d", ".1f"],
        ["Counts", "Pourcentage (%)"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=classes, yticklabels=classes,
                    linewidths=0.5, ax=ax)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Prédiction", fontsize=11)
        ax.set_ylabel("Réel", fontsize=11)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Confusion matrix → {path}")


def plot_roc_curves(labels, probs, classes, save_dir):
    y_bin = label_binarize(labels, classes=list(range(len(classes))))
    colors = ["#e74c3c", "#f39c12", "#27ae60", "#2980b9"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Brain Tumor Classifier — Courbes ROC", fontsize=14, fontweight="bold")

    # Per-class ROC
    ax = axes[0]
    mean_auc = []
    for i, (cls, col) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        mean_auc.append(roc_auc)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{cls} (AUC={roc_auc:.3f})")
    ax.plot([0,1],[0,1],"--", color="gray", lw=1)
    ax.set_xlabel("Taux de faux positifs"); ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("ROC par classe"); ax.legend(loc="lower right"); ax.grid(alpha=0.3)

    # AUC bar chart
    ax2 = axes[1]
    bars = ax2.bar(classes, mean_auc, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, mean_auc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set_ylim(0.8, 1.02); ax2.set_ylabel("AUC")
    ax2.set_title(f"AUC par classe (moyenne: {np.mean(mean_auc):.3f})")
    ax2.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()

    path = os.path.join(save_dir, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ ROC curves → {path}")


def plot_training_history_from_file(save_dir):
    history_path = os.path.join(save_dir, "history.json")
    if not os.path.exists(history_path):
        return
    with open(history_path) as f:
        h = json.load(f)
    plot_training_history(h, save_dir)


def plot_training_history(history, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Courbes d'entraînement", fontsize=14, fontweight="bold")

    ax1.plot(epochs, history["train_loss"], "b-o", ms=4, label="Train")
    ax1.plot(epochs, history["val_loss"],   "r-o", ms=4, label="Val")
    ax1.axvline(x=PHASE1_EPOCHS + 0.5, color="gray", ls="--", lw=1, label="Fine-tuning")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss (CrossEntropy)")
    ax1.set_title("Perte"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, [a*100 for a in history["train_acc"]], "b-o", ms=4, label="Train")
    ax2.plot(epochs, [a*100 for a in history["val_acc"]],   "r-o", ms=4, label="Val")
    ax2.axvline(x=PHASE1_EPOCHS + 0.5, color="gray", ls="--", lw=1, label="Fine-tuning")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Précision"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Training curves → {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Tumor Classifier — Training")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Chemin vers le dataset (contient Training/ et Testing/)")
    parser.add_argument("--save_dir", type=str, default="output",
                        help="Dossier de sauvegarde (défaut: output/)")
    args = parser.parse_args()

    train(args.data_dir, args.save_dir)
