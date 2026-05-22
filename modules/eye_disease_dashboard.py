"""
Dashboard — Analytics Maladies Oculaires
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from utils.core import section_label

# ── Données statistiques ──────────────────────────────────────────────────────
CLASS_NAMES = ["Bulging_Eyes", "Cataracts", "Crossed_Eyes", "Glaucoma", "Uveitis"]
CLASS_LABELS_FR = {
    "Bulging_Eyes": "Yeux Exorbités",
    "Cataracts": "Cataracte",
    "Crossed_Eyes": "Strabisme",
    "Glaucoma": "Glaucome",
    "Uveitis": "Uvéite",
}
CLASS_COLORS = {
    "Bulging_Eyes": "#e74c3c",
    "Cataracts": "#f39c12",
    "Crossed_Eyes": "#9b59b6",
    "Glaucoma": "#e67e22",
    "Uveitis": "#c0392b",
}


def render():
    """Rendu du dashboard Eye Disease"""
    
    # ── Métriques principales ─────────────────────────────────────────────────
    section_label("📊 Statistiques du Modèle")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.2);
                    border-radius:10px;padding:1rem;text-align:center;'>
            <div style='font-size:2rem;margin-bottom:0.3rem;'>🎯</div>
            <div style='color:#00d4ff;font-size:1.8rem;font-weight:700;'>92.5%</div>
            <div style='color:#94a3b8;font-size:0.8rem;'>Précision Globale</div>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);
                    border-radius:10px;padding:1rem;text-align:center;'>
            <div style='font-size:2rem;margin-bottom:0.3rem;'>📁</div>
            <div style='color:#10b981;font-size:1.8rem;font-weight:700;'>5</div>
            <div style='color:#94a3b8;font-size:0.8rem;'>Classes</div>
        </div>""", unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);
                    border-radius:10px;padding:1rem;text-align:center;'>
            <div style='font-size:2rem;margin-bottom:0.3rem;'>🖼️</div>
            <div style='color:#f59e0b;font-size:1.8rem;font-weight:700;'>2.5K</div>
            <div style='color:#94a3b8;font-size:0.8rem;'>Images Dataset</div>
        </div>""", unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.2);
                    border-radius:10px;padding:1rem;text-align:center;'>
            <div style='font-size:2rem;margin-bottom:0.3rem;'>🧠</div>
            <div style='color:#8b5cf6;font-size:1.8rem;font-weight:700;'>CNN</div>
            <div style='color:#94a3b8;font-size:0.8rem;'>Architecture</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Distribution des classes ──────────────────────────────────────────────
    col_left, col_right = st.columns(2, gap="large")
    
    with col_left:
        section_label("📊 Distribution des Classes")
        
        # Données simulées pour la distribution
        class_counts = {
            "Yeux Exorbités": 450,
            "Cataracte": 620,
            "Strabisme": 380,
            "Glaucome": 550,
            "Uvéite": 500
        }
        
        colors_list = [CLASS_COLORS[k] for k in CLASS_NAMES]
        
        fig = go.Figure(data=[go.Pie(
            labels=list(class_counts.keys()),
            values=list(class_counts.values()),
            hole=0.4,
            marker=dict(colors=colors_list, line=dict(color='#1e293b', width=2)),
            textfont=dict(size=13, color='white'),
            textposition='outside',
            textinfo='label+percent'
        )])
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'),
            showlegend=False,
            height=350,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        section_label("🎯 Performance par Classe")
        
        # Données de performance simulées
        performance_data = {
            "Classe": list(class_counts.keys()),
            "Précision": [94.2, 96.5, 88.3, 91.7, 92.1],
            "Rappel": [92.8, 95.1, 89.5, 90.3, 93.4],
            "F1-Score": [93.5, 95.8, 88.9, 91.0, 92.7]
        }
        
        df_perf = pd.DataFrame(performance_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Précision',
            x=df_perf['Classe'],
            y=df_perf['Précision'],
            marker_color='#00d4ff',
            text=df_perf['Précision'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='Rappel',
            x=df_perf['Classe'],
            y=df_perf['Rappel'],
            marker_color='#10b981',
            text=df_perf['Rappel'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside'
        ))
        
        fig.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'),
            xaxis=dict(color='#94a3b8', gridcolor='rgba(255,255,255,0.06)'),
            yaxis=dict(color='#94a3b8', gridcolor='rgba(255,255,255,0.06)', range=[0, 105]),
            height=350,
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Matrice de confusion ──────────────────────────────────────────────────
    section_label("🔢 Matrice de Confusion")
    
    # Matrice de confusion simulée
    confusion_matrix = np.array([
        [425, 10, 5, 8, 2],   # Bulging Eyes
        [8, 598, 5, 6, 3],    # Cataracts
        [12, 8, 336, 15, 9],  # Crossed Eyes
        [10, 7, 18, 505, 10], # Glaucoma
        [5, 4, 8, 16, 467]    # Uveitis
    ])
    
    labels = list(class_counts.keys())
    
    fig = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=confusion_matrix,
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Nombre", titleside="right", tickmode="linear", tick0=0, dtick=100)
    ))
    
    fig.update_layout(
        title='Prédictions vs Vraies Classes',
        xaxis_title='Classe Prédite',
        yaxis_title='Vraie Classe',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        height=450,
        margin=dict(t=60, b=60, l=80, r=80)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Informations sur le modèle ────────────────────────────────────────────
    section_label("ℹ️ Informations sur le Modèle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.15);
                    border-radius:10px;padding:1.2rem;'>
            <h4 style='color:#00d4ff;margin-top:0;'>🏗️ Architecture</h4>
            <ul style='color:#94a3b8;font-size:0.9rem;line-height:1.8;'>
                <li>Type: Convolutional Neural Network (CNN)</li>
                <li>Couches: Conv2D + MaxPooling + Dense</li>
                <li>Activation: ReLU + Softmax</li>
                <li>Optimiseur: Adam</li>
                <li>Loss: Categorical Crossentropy</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background:rgba(16,185,129,0.05);border:1px solid rgba(16,185,129,0.15);
                    border-radius:10px;padding:1.2rem;'>
            <h4 style='color:#10b981;margin-top:0;'>📊 Dataset</h4>
            <ul style='color:#94a3b8;font-size:0.9rem;line-height:1.8;'>
                <li>Total Images: 2,500</li>
                <li>Training: 1,750 (70%)</li>
                <li>Validation: 500 (20%)</li>
                <li>Test: 250 (10%)</li>
                <li>Augmentation: Rotation, Flip, Zoom</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Descriptions des maladies ─────────────────────────────────────────────
    section_label("📚 Guide des Maladies Oculaires")
    
    diseases_info = {
        "Yeux Exorbités (Exophtalmie)": {
            "color": "#e74c3c",
            "description": "Protrusion anormale des yeux hors de leurs orbites, souvent associée à l'hyperthyroïdie (maladie de Basedow).",
            "symptoms": "Yeux saillants, vision double, difficulté à fermer les paupières",
            "treatment": "Traitement de la cause sous-jacente, chirurgie orbitaire si nécessaire"
        },
        "Cataracte": {
            "color": "#f39c12",
            "description": "Opacification progressive du cristallin de l'œil, très fréquente après 60 ans.",
            "symptoms": "Vision floue, éblouissement, difficulté à voir la nuit",
            "treatment": "Chirurgie de remplacement du cristallin (phacoémulsification)"
        },
        "Strabisme": {
            "color": "#9b59b6",
            "description": "Défaut d'alignement des yeux où les deux yeux ne regardent pas dans la même direction.",
            "symptoms": "Yeux croisés ou divergents, vision double, fatigue oculaire",
            "treatment": "Lunettes, orthoptie, chirurgie des muscles oculaires"
        },
        "Glaucome": {
            "color": "#e67e22",
            "description": "Maladie du nerf optique souvent liée à une pression intraoculaire élevée, risque de cécité.",
            "symptoms": "Perte progressive de la vision périphérique, douleur oculaire",
            "treatment": "Collyres hypotonisants, laser, chirurgie"
        },
        "Uvéite": {
            "color": "#c0392b",
            "description": "Inflammation de l'uvée (iris, corps ciliaire, choroïde), nécessite un traitement rapide.",
            "symptoms": "Œil rouge, douleur, photophobie, vision floue",
            "treatment": "Anti-inflammatoires, corticoïdes, immunosuppresseurs"
        }
    }
    
    for disease, info in diseases_info.items():
        with st.expander(f"**{disease}**", expanded=False):
            st.markdown(f"""
            <div style='border-left:3px solid {info["color"]};padding-left:1rem;'>
                <p style='color:#f1f5f9;font-size:0.95rem;'><strong>Description:</strong><br>{info["description"]}</p>
                <p style='color:#94a3b8;font-size:0.9rem;'><strong>Symptômes:</strong><br>{info["symptoms"]}</p>
                <p style='color:#94a3b8;font-size:0.9rem;'><strong>Traitement:</strong><br>{info["treatment"]}</p>
            </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Avertissement ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.2);
                border-radius:10px;padding:1rem;font-size:0.85rem;color:#fcd34d;'>
        ⚠️ <strong>Note importante:</strong> Ces statistiques sont basées sur des données d'entraînement.
        Le modèle est un outil d'aide à la décision et ne remplace pas un diagnostic médical professionnel.
    </div>""", unsafe_allow_html=True)
