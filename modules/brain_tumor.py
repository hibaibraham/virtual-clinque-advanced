"""
Module — Diagnostic Tumeur Cérébrale (IRM)
Modèle : EfficientNet-B0 (PyTorch) — 4 classes
"""

import os
import io
import sys
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image

# ── Chemin vers le modèle ──────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "brain_tumer_deep" / "output" / "best_model.pth"

# ── Classes & couleurs ─────────────────────────────────────────────────────────
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_COLORS = {
    "glioma":      "#e74c3c",
    "meningioma":  "#f39c12",
    "notumor":     "#27ae60",
    "pituitary":   "#2980b9",
}
CLASS_LABELS_FR = {
    "glioma":      "Gliome",
    "meningioma":  "Méningiome",
    "notumor":     "Pas de tumeur",
    "pituitary":   "Tumeur hypophysaire",
}
CLASS_DESC = {
    "glioma":     "Tumeur des cellules gliales — souvent agressive, nécessite une prise en charge urgente.",
    "meningioma": "Tumeur des méninges — généralement bénigne, croissance lente.",
    "notumor":    "Aucune tumeur détectée sur l'IRM analysée.",
    "pituitary":  "Tumeur de l'hypophyse — peut affecter la régulation hormonale.",
}
CONFIDENCE_THRESHOLD = 0.50
IMG_SIZE = 224


# ── Chargement du modèle (mis en cache) ───────────────────────────────────────
@st.cache_resource(show_spinner="Chargement du modèle EfficientNet-B0…")
def _load_brain_model():
    """Charge le modèle PyTorch depuis best_model.pth."""
    import torch
    import torch.nn as nn
    from torchvision import models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(str(MODEL_PATH), map_location=device)
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

    val_acc = ckpt.get("val_acc", None)
    return model, class_names, device, val_acc


# ── Prétraitement ──────────────────────────────────────────────────────────────
def _preprocess(pil_image):
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return tf(pil_image.convert("RGB")).unsqueeze(0)


# ── Inférence ─────────────────────────────────────────────────────────────────
def _predict(model, tensor, class_names, device):
    import torch
    with torch.no_grad():
        outputs = model(tensor.to(device))
        probs   = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    label    = class_names[pred_idx]
    conf     = float(probs[pred_idx])
    return {
        "predicted_class": label,
        "confidence":      conf,
        "has_tumor":       label != "notumor",
        "uncertain":       conf < CONFIDENCE_THRESHOLD,
        "probabilities":   {c: float(p) for c, p in zip(class_names, probs)},
    }


# ── Graphique probabilités (Plotly) ───────────────────────────────────────────
def _prob_chart(result):
    import plotly.graph_objects as go

    probs  = result["probabilities"]
    labels = list(probs.keys())
    values = [probs[l] * 100 for l in labels]
    colors = [CLASS_COLORS.get(l, "#7f8c8d") for l in labels]
    labels_fr = [CLASS_LABELS_FR.get(l, l) for l in labels]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels_fr,
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.1)", width=1)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#f1f5f9", size=12),
    ))
    fig.add_vline(
        x=CONFIDENCE_THRESHOLD * 100,
        line_dash="dash", line_color="#94a3b8", line_width=1,
        annotation_text=f"Seuil {CONFIDENCE_THRESHOLD*100:.0f}%",
        annotation_font_color="#94a3b8",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis=dict(range=[0, 115], title="Probabilité (%)",
                   color="#94a3b8", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(color="#94a3b8"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=60),
        height=220,
        showlegend=False,
        font=dict(color="#94a3b8"),
    )
    return fig


# ── Rendu principal ────────────────────────────────────────────────────────────
def render():
    from utils.core import section_label

    # ── Vérification dépendances ───────────────────────────────────────────────
    try:
        import torch
        import torchvision
    except ImportError:
        st.error("❌ PyTorch / torchvision non installés.")
        st.code("pip install torch torchvision", language="bash")
        return

    # ── Vérification modèle ────────────────────────────────────────────────────
    model_exists = MODEL_PATH.exists()

    col_upload, col_result = st.columns([2, 3], gap="large")

    # ── Colonne gauche : upload + infos ───────────────────────────────────────
    with col_upload:
        section_label("🧠 Analyse IRM Cérébrale")

        uploaded = st.file_uploader(
            "Déposez une image IRM (.jpg, .png, .bmp)",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
            help="Image IRM cérébrale en coupe axiale ou coronale"
        )

        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption="IRM chargée", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Infos classes
        section_label("📋 Classes Diagnostiques")
        for cls in CLASS_NAMES:
            color = CLASS_COLORS[cls]
            label = CLASS_LABELS_FR[cls]
            desc  = CLASS_DESC[cls]
            st.markdown(f"""
            <div style='display:flex;align-items:flex-start;gap:0.6rem;
                        margin-bottom:0.6rem;padding:0.6rem 0.8rem;
                        background:rgba(255,255,255,0.03);border-radius:8px;
                        border-left:3px solid {color};'>
                <div style='font-size:0.85rem;'>
                    <b style='color:{color}'>{label}</b>
                    <div style='color:#94a3b8;font-size:0.78rem;margin-top:0.2rem;'>{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Statut modèle
        st.markdown("<br>", unsafe_allow_html=True)
        if model_exists:
            st.markdown("""
            <div style='padding:0.6rem 1rem;background:rgba(16,185,129,0.08);
                        border:1px solid rgba(16,185,129,0.25);border-radius:8px;
                        font-size:0.8rem;color:#6ee7b7;'>
                ✅ Modèle EfficientNet-B0 disponible
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='padding:0.6rem 1rem;background:rgba(239,68,68,0.08);
                        border:1px solid rgba(239,68,68,0.25);border-radius:8px;
                        font-size:0.8rem;color:#fca5a5;'>
                ⚠️ Modèle non entraîné — lancez d'abord l'entraînement
            </div>""", unsafe_allow_html=True)
            st.code(
                "cd brain_tumer_deep\n"
                "python train.py --data_dir brain-tumor-mri-dataset --save_dir output",
                language="bash"
            )

    # ── Colonne droite : résultats ─────────────────────────────────────────────
    with col_result:
        section_label("🎯 Résultat du Diagnostic")

        if not model_exists:
            st.markdown("""
            <div style='background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.15);
                        border-radius:12px;padding:2rem;text-align:center;color:#94a3b8;'>
                <div style='font-size:2.5rem;margin-bottom:0.8rem;'>🧠</div>
                <div style='font-weight:600;color:#f1f5f9;margin-bottom:0.4rem;'>Modèle non disponible</div>
                <div style='font-size:0.85rem;'>Entraînez le modèle pour activer ce module</div>
            </div>""", unsafe_allow_html=True)
            return

        if not uploaded:
            st.markdown("""
            <div style='background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.15);
                        border-radius:12px;padding:2rem;text-align:center;color:#94a3b8;'>
                <div style='font-size:2.5rem;margin-bottom:0.8rem;'>🔬</div>
                <div style='font-weight:600;color:#f1f5f9;margin-bottom:0.4rem;'>En attente d'une image IRM</div>
                <div style='font-size:0.85rem;'>Chargez une image pour lancer l'analyse</div>
            </div>""", unsafe_allow_html=True)
            return

        # ── Analyse ────────────────────────────────────────────────────────────
        with st.spinner("Analyse en cours…"):
            try:
                model, class_names, device, val_acc = _load_brain_model()
                tensor = _preprocess(pil_img)
                result = _predict(model, tensor, class_names, device)
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse : {e}")
                return

        cls   = result["predicted_class"]
        conf  = result["confidence"]
        color = CLASS_COLORS.get(cls, "#7f8c8d")
        label = CLASS_LABELS_FR.get(cls, cls)

        # ── Carte résultat ─────────────────────────────────────────────────────
        if result["uncertain"]:
            status_icon  = "❓"
            status_title = "Résultat Incertain"
            status_sub   = f"Confiance insuffisante ({conf*100:.1f}%) — consultation recommandée"
            card_bg      = "rgba(245,158,11,0.08)"
            card_border  = "rgba(245,158,11,0.3)"
            title_color  = "#fcd34d"
        elif result["has_tumor"]:
            status_icon  = "⚠️"
            status_title = "Tumeur Détectée"
            status_sub   = CLASS_DESC.get(cls, "")
            card_bg      = "rgba(239,68,68,0.08)"
            card_border  = "rgba(239,68,68,0.3)"
            title_color  = "#fca5a5"
        else:
            status_icon  = "✅"
            status_title = "Pas de Tumeur Détectée"
            status_sub   = CLASS_DESC.get(cls, "")
            card_bg      = "rgba(16,185,129,0.08)"
            card_border  = "rgba(16,185,129,0.3)"
            title_color  = "#6ee7b7"

        st.markdown(f"""
        <div style='background:{card_bg};border:1px solid {card_border};
                    border-radius:12px;padding:1.5rem;margin-bottom:1rem;'>
            <div style='font-size:2rem;margin-bottom:0.4rem;'>{status_icon}</div>
            <h3 style='color:{title_color};margin:0 0 0.3rem 0;'>{status_title}</h3>
            <div style='color:{color};font-size:1.1rem;font-weight:600;margin-bottom:0.4rem;'>
                {label}
            </div>
            <div style='color:#94a3b8;font-size:0.85rem;'>{status_sub}</div>
        </div>""", unsafe_allow_html=True)

        # ── Confiance ──────────────────────────────────────────────────────────
        section_label("📊 Distribution des Probabilités")
        st.plotly_chart(_prob_chart(result), use_container_width=True)

        # ── Barre de confiance ─────────────────────────────────────────────────
        conf_pct = conf * 100
        conf_color = color if conf >= CONFIDENCE_THRESHOLD else "#f59e0b"
        st.markdown(f"""
        <div style='margin-bottom:1rem;'>
            <div style='display:flex;justify-content:space-between;
                        font-size:0.82rem;color:#94a3b8;margin-bottom:0.3rem;'>
                <span>Confiance du modèle</span>
                <span style='color:{conf_color};font-weight:600;'>{conf_pct:.1f}%</span>
            </div>
            <div style='background:rgba(255,255,255,0.06);border-radius:999px;height:8px;overflow:hidden;'>
                <div style='width:{conf_pct}%;height:100%;background:{conf_color};
                            border-radius:999px;transition:width 0.4s;'></div>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Tableau détaillé ───────────────────────────────────────────────────
        section_label("🔢 Détail des Scores")
        import pandas as pd
        probs_df = pd.DataFrame([
            {
                "Classe":       CLASS_LABELS_FR.get(c, c),
                "Probabilité":  f"{result['probabilities'][c]*100:.2f}%",
                "Statut":       "✅ Sélectionné" if c == cls else "",
            }
            for c in class_names
        ]).sort_values("Probabilité", ascending=False)
        st.dataframe(probs_df, use_container_width=True, hide_index=True)

        # ── Rapport téléchargeable ─────────────────────────────────────────────
        section_label("📄 Rapport")
        import datetime
        rapport = f"""RAPPORT D'ANALYSE IRM — MedAI Brain Tumor
==========================================
Date       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
Fichier    : {uploaded.name}

RÉSULTAT   : {label.upper()}
Classe     : {cls}
Confiance  : {conf*100:.2f}%
Tumeur     : {'OUI ⚠️' if result['has_tumor'] else 'NON ✅'}
Incertain  : {'OUI ❓' if result['uncertain'] else 'NON'}

DISTRIBUTION DES PROBABILITÉS
"""
        for c in class_names:
            p = result["probabilities"][c] * 100
            bar = "█" * int(p / 5) + "░" * (20 - int(p / 5))
            rapport += f"  {CLASS_LABELS_FR.get(c,c):25s} {bar} {p:.2f}%\n"

        rapport += f"""
MODÈLE     : EfficientNet-B0 (Transfer Learning)
Val. Acc.  : {f'{val_acc*100:.2f}%' if val_acc else 'N/A'}

AVERTISSEMENT : Outil d'aide à la décision uniquement.
Ce résultat ne remplace pas un diagnostic médical professionnel.
"""
        st.download_button(
            "📥 Télécharger le rapport",
            rapport,
            file_name="rapport_irm_cerebrale.txt",
            mime="text/plain",
        )

        # ── Avertissement médical ──────────────────────────────────────────────
        st.markdown("""
        <div style='margin-top:1rem;padding:0.8rem 1rem;
                    background:rgba(245,158,11,0.06);
                    border:1px solid rgba(245,158,11,0.2);
                    border-radius:8px;font-size:0.78rem;color:#fcd34d;'>
            ⚠️ <strong>Avertissement médical</strong> — Ce système est un outil d'aide à la décision.
            Tout résultat doit être confirmé par un radiologue ou neurologue qualifié.
        </div>""", unsafe_allow_html=True)
