"""
Brain Tumor Classifier — Script de Prédiction (Inférence)
Usage:
  python predict.py --model output/best_model.pth --image path/to/img.jpg
  python predict.py --model output/best_model.pth --folder path/to/images/
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_SIZE    = 224
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
COLORS      = {"glioma": "#e74c3c", "meningioma": "#f39c12",
               "notumor": "#27ae60", "pituitary": "#2980b9"}
CONFIDENCE_THRESHOLD = 0.5   # En dessous → afficher "Incertain"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# CHARGER LE MODÈLE
# ─────────────────────────────────────────────
def load_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    class_names = ckpt.get("class_names", CLASS_NAMES)
    num_classes  = len(class_names)

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    print(f"✅ Modèle chargé | Val acc: {ckpt.get('val_acc', '?'):.4f}")
    print(f"   Classes : {class_names}")
    return model, class_names


# ─────────────────────────────────────────────
# PRÉDICTION SUR UNE IMAGE
# ─────────────────────────────────────────────
@torch.no_grad()
def predict_image(model, image_path: str, class_names: list):
    """
    Retourne : (label_prédit, confiance, dict probabilités)
    """
    img = Image.open(image_path).convert("RGB")
    tensor = val_transform(img).unsqueeze(0).to(device)

    outputs = model(tensor)
    probs   = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    pred_idx = np.argmax(probs)
    label    = class_names[pred_idx]
    conf     = float(probs[pred_idx])

    if conf < CONFIDENCE_THRESHOLD:
        label_display = f"Incertain ({label})"
    else:
        label_display = label

    result = {
        "predicted_class": label,
        "display_label":   label_display,
        "confidence":      conf,
        "probabilities":   {c: float(p) for c, p in zip(class_names, probs)},
        "has_tumor":       label != "notumor",
    }
    return result, img


# ─────────────────────────────────────────────
# VISUALISATION D'UNE PRÉDICTION
# ─────────────────────────────────────────────
def visualize_prediction(result: dict, img: Image.Image, save_path: str = None):
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#f8f9fa")

    # --- Image ---
    ax_img.imshow(img)
    ax_img.axis("off")

    label = result["display_label"]
    conf  = result["confidence"]
    color = COLORS.get(result["predicted_class"], "#7f8c8d")

    title = f"{label.upper()}\nConfiance : {conf*100:.1f}%"
    ax_img.set_title(title, fontsize=14, fontweight="bold", color=color, pad=10)

    status = "⚠️  TUMEUR DÉTECTÉE" if result["has_tumor"] else "✅  PAS DE TUMEUR"
    ax_img.text(0.5, -0.04, status, transform=ax_img.transAxes,
                ha="center", fontsize=11, color=color, fontweight="bold")

    # --- Barres de probabilité ---
    probs  = result["probabilities"]
    labels = list(probs.keys())
    values = [probs[l] * 100 for l in labels]
    bar_colors = [COLORS.get(l, "#7f8c8d") for l in labels]

    bars = ax_bar.barh(labels, values, color=bar_colors, edgecolor="white",
                       linewidth=0.8, height=0.55)
    for bar, val in zip(bars, values):
        ax_bar.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=11, fontweight="bold")

    ax_bar.set_xlim(0, 115)
    ax_bar.set_xlabel("Probabilité (%)", fontsize=11)
    ax_bar.set_title("Distribution des probabilités", fontsize=12, fontweight="bold")
    ax_bar.axvline(x=CONFIDENCE_THRESHOLD*100, color="gray", ls="--", lw=1,
                   label=f"Seuil ({CONFIDENCE_THRESHOLD*100:.0f}%)")
    ax_bar.legend(fontsize=9)
    ax_bar.grid(axis="x", alpha=0.3)
    ax_bar.set_facecolor("#f0f0f0")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Visualisation → {save_path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────
# PRÉDICTION SUR UN DOSSIER (batch)
# ─────────────────────────────────────────────
def predict_folder(model, folder_path: str, class_names: list, save_dir: str = "predictions"):
    os.makedirs(save_dir, exist_ok=True)
    images = [p for p in Path(folder_path).rglob("*") if p.suffix.lower() in VALID_EXTS]

    if not images:
        print(f"❌ Aucune image trouvée dans {folder_path}")
        return

    print(f"\n🔍 {len(images)} images trouvées. Prédiction en cours...\n")

    results_list = []
    for img_path in images:
        result, img = predict_image(model, str(img_path), class_names)
        print(f"  {img_path.name:40s} → {result['predicted_class']:12s} "
              f"({result['confidence']*100:.1f}%)")
        results_list.append({
            "file": img_path.name,
            **result
        })

    # ── Résumé ───────────────────────────────────
    print("\n" + "="*55)
    print("📊 RÉSUMÉ")
    from collections import Counter
    counts = Counter(r["predicted_class"] for r in results_list)
    for cls in class_names:
        n = counts.get(cls, 0)
        bar = "█" * n + "░" * (max(counts.values()) - n) if counts else ""
        print(f"  {cls:12s}: {n:3d}  {bar}")
    print("="*55)

    # ── Plot grille ───────────────────────────────
    _plot_batch_grid(images[:16], results_list[:16], class_names, save_dir)


def _plot_batch_grid(images, results, class_names, save_dir):
    n    = len(images)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).flatten()

    for i, (img_path, res) in enumerate(zip(images, results)):
        img   = Image.open(img_path).convert("RGB")
        color = COLORS.get(res["predicted_class"], "#7f8c8d")
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(
            f"{res['predicted_class']}\n{res['confidence']*100:.0f}%",
            fontsize=9, color=color, fontweight="bold"
        )
        for spine in axes[i].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    patches = [mpatches.Patch(color=c, label=l)
               for l, c in COLORS.items() if l in class_names]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0))
    fig.suptitle("Prédictions batch", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = os.path.join(save_dir, "batch_predictions.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Grille de prédictions → {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Tumor — Inférence")
    parser.add_argument("--model",  required=True, help="Chemin vers best_model.pth")
    parser.add_argument("--image",  default=None,  help="Chemin vers une image unique")
    parser.add_argument("--folder", default=None,  help="Dossier d'images (batch)")
    parser.add_argument("--save",   default=None,  help="Fichier de sortie pour la viz")
    args = parser.parse_args()

    model, class_names = load_model(args.model)

    if args.image:
        result, img = predict_image(model, args.image, class_names)
        print(f"\n🧠 Résultat : {result['predicted_class']} ({result['confidence']*100:.1f}%)")
        print(f"   Tumeur   : {'OUI ⚠️' if result['has_tumor'] else 'NON ✅'}")
        for cls, prob in result["probabilities"].items():
            bar = "█" * int(prob * 20)
            print(f"   {cls:12s} {bar:<20} {prob*100:.1f}%")
        visualize_prediction(result, img, save_path=args.save)

    elif args.folder:
        predict_folder(model, args.folder, class_names,
                       save_dir=args.save or "predictions")
    else:
        parser.print_help()
