"""
Module — Diagnostic Maladies Oculaires
Modèle : CNN — 5 classes
"""

import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path

# ── Classes & couleurs ─────────────────────────────────────────────────────────
CLASS_NAMES = ["Bulging_Eyes", "Cataracts", "Crossed_Eyes", "Glaucoma", "Uveitis"]
CLASS_COLORS = {
    "Bulging_Eyes": "#e74c3c",
    "Cataracts": "#f39c12",
    "Crossed_Eyes": "#9b59b6",
    "Glaucoma": "#e67e22",
    "Uveitis": "#c0392b",
}
CLASS_LABELS_FR = {
    "Bulging_Eyes": "Yeux Exorbités",
    "Cataracts": "Cataracte",
    "Crossed_Eyes": "Strabisme",
    "Glaucoma": "Glaucome",
    "Uveitis": "Uvéite",
}
CLASS_DESC = {
    "Bulging_Eyes": "Protrusion anormale des yeux — souvent liée à l'hyperthyroïdie.",
    "Cataracts": "Opacification du cristallin — très fréquente après 60 ans.",
    "Crossed_Eyes": "Défaut d'alignement des yeux — nécessite une correction.",
    "Glaucoma": "Maladie du nerf optique — risque de cécité si non traité.",
    "Uveitis": "Inflammation de l'uvée — nécessite un traitement rapide.",
}
CONFIDENCE_THRESHOLD = 0.50
IMG_SIZE = 224


# ── Chargement du modèle (mis en cache) ───────────────────────────────────────
@st.cache_resource(show_spinner="Chargement du modèle Eye Disease…")
def _load_eye_model():
    """Charge le modèle depuis models/eye_disease_model.py."""
    from models.eye_disease_model import EyeDiseaseModel
    
    model = EyeDiseaseModel()
    model.load()
    
    return model


# ── Graphique probabilités (Plotly) ───────────────────────────────────────────
def _prob_chart(probabilities):
    import plotly.graph_objects as go

    labels = list(probabilities.keys())
    values = [probabilities[l] * 100 for l in labels]
    colors = [CLASS_COLORS.get(k, "#7f8c8d") for k in CLASS_NAMES]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
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
        height=250,
        showlegend=False,
        font=dict(color="#94a3b8"),
    )
    return fig


# ── Rendu principal ────────────────────────────────────────────────────────────
def render():
    from utils.core import section_label

    col_upload, col_result = st.columns([2, 3], gap="large")

    # ── Colonne gauche : upload + infos ───────────────────────────────────────
    with col_upload:
        section_label("👁️ Analyse Oculaire")

        uploaded = st.file_uploader(
            "Déposez une image oculaire (.jpg, .png, .bmp)",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
            help="Image de l'œil ou photo clinique"
        )

        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption="Image chargée", use_container_width=True)

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
        st.markdown("""
        <div style='padding:0.6rem 1rem;background:rgba(16,185,129,0.08);
                    border:1px solid rgba(16,185,129,0.25);border-radius:8px;
                    font-size:0.8rem;color:#6ee7b7;'>
            ✅ Modèle CNN Eye Disease disponible
        </div>""", unsafe_allow_html=True)

    # ── Colonne droite : résultats ─────────────────────────────────────────────
    with col_result:
        section_label("🎯 Résultat du Diagnostic")

        if not uploaded:
            st.markdown("""
            <div style='background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.15);
                        border-radius:12px;padding:2rem;text-align:center;color:#94a3b8;'>
                <div style='font-size:2.5rem;margin-bottom:0.8rem;'>👁️</div>
                <div style='font-weight:600;color:#f1f5f9;margin-bottom:0.4rem;'>En attente d'une image</div>
                <div style='font-size:0.85rem;'>Chargez une image pour lancer l'analyse</div>
            </div>""", unsafe_allow_html=True)
            return

        # ── Analyse ────────────────────────────────────────────────────────────
        with st.spinner("Analyse en cours…"):
            try:
                model = _load_eye_model()
                
                # Convertir l'image en bytes
                import io
                img_byte_arr = io.BytesIO()
                pil_img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Prédiction
                predicted_class, confidence, confidences = model.predict(img_byte_arr)
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse : {e}")
                return

        cls   = predicted_class
        conf  = confidence
        color = CLASS_COLORS.get(cls, "#7f8c8d")
        label = CLASS_LABELS_FR.get(cls, cls)
        
        # Sauvegarder la prédiction
        from utils.core import save_prediction
        patient_data = {
            'image_name': uploaded.name if uploaded else 'unknown',
            'confidence': conf,
            'disease_detected': True
        }
        save_prediction(patient_data, cls, conf, model_type='eye_disease')

        # ── Carte résultat ─────────────────────────────────────────────────────
        if conf < CONFIDENCE_THRESHOLD:
            status_icon  = "❓"
            status_title = "Résultat Incertain"
            status_sub   = f"Confiance insuffisante ({conf*100:.1f}%) — consultation recommandée"
            card_bg      = "rgba(245,158,11,0.08)"
            card_border  = "rgba(245,158,11,0.3)"
            title_color  = "#fcd34d"
        else:
            status_icon  = "⚠️"
            status_title = "Maladie Détectée"
            status_sub   = CLASS_DESC.get(cls, "")
            card_bg      = "rgba(239,68,68,0.08)"
            card_border  = "rgba(239,68,68,0.3)"
            title_color  = "#fca5a5"

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
        st.plotly_chart(_prob_chart(confidences), use_container_width=True)

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
                "Probabilité":  f"{confidences[CLASS_LABELS_FR.get(c, c)]*100:.2f}%",
                "Statut":       "✅ Sélectionné" if c == cls else "",
            }
            for c in CLASS_NAMES
        ]).sort_values("Probabilité", ascending=False)
        st.dataframe(probs_df, use_container_width=True, hide_index=True)

        # ── Rapport téléchargeable ─────────────────────────────────────────────
        section_label("📄 Rapport")
        import datetime
        rapport = f"""RAPPORT D'ANALYSE OCULAIRE — NovaClinic
==========================================
Date       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
Fichier    : {uploaded.name}

RÉSULTAT   : {label.upper()}
Classe     : {cls}
Confiance  : {conf*100:.2f}%

DISTRIBUTION DES PROBABILITÉS
"""
        for c in CLASS_NAMES:
            label_fr = CLASS_LABELS_FR.get(c, c)
            p = confidences[label_fr] * 100
            bar = "█" * int(p / 5) + "░" * (20 - int(p / 5))
            rapport += f"  {label_fr:25s} {bar} {p:.2f}%\n"

        rapport += f"""
MODÈLE     : CNN Eye Disease
Classes    : 5 (Bulging Eyes, Cataracts, Crossed Eyes, Glaucoma, Uveitis)

AVERTISSEMENT : Outil d'aide à la décision uniquement.
Ce résultat ne remplace pas un diagnostic médical professionnel.
"""
        st.download_button(
            "📥 Télécharger le rapport",
            rapport,
            file_name="rapport_analyse_oculaire.txt",
            mime="text/plain",
        )

        # ── Avertissement médical ──────────────────────────────────────────────
        st.markdown("""
        <div style='margin-top:1rem;padding:0.8rem 1rem;
                    background:rgba(245,158,11,0.06);
                    border:1px solid rgba(245,158,11,0.2);
                    border-radius:8px;font-size:0.78rem;color:#fcd34d;'>
            ⚠️ <strong>Avertissement médical</strong> — Ce système est un outil d'aide à la décision.
            Tout résultat doit être confirmé par un ophtalmologue qualifié.
        </div>""", unsafe_allow_html=True)
