"""
Tableau de Bord — Modèle Cancer Cérébral (Deep Learning)
Statistiques, métriques et visualisations du modèle EfficientNet-B0
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from utils.core import section_label, DARK_LAYOUT

# ── Chemins ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "brain_tumer_deep" / "output"
HISTORY_PATH = MODEL_DIR / "history.json"
MODEL_PATH = MODEL_DIR / "best_model.pth"

# ── Classes ────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_LABELS_FR = {
    "glioma": "Gliome",
    "meningioma": "Méningiome",
    "notumor": "Pas de tumeur",
    "pituitary": "Tumeur hypophysaire",
}
CLASS_COLORS = {
    "glioma": "#e74c3c",
    "meningioma": "#f39c12",
    "notumor": "#27ae60",
    "pituitary": "#2980b9",
}


def load_training_history():
    """Charge l'historique d'entraînement depuis history.json"""
    if not HISTORY_PATH.exists():
        return None
    
    with open(HISTORY_PATH, 'r') as f:
        return json.load(f)


def load_model_info():
    """Charge les informations du modèle depuis best_model.pth"""
    if not MODEL_PATH.exists():
        return None
    
    try:
        import torch
        checkpoint = torch.load(str(MODEL_PATH), map_location='cpu')
        return checkpoint
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None


def render():
    """Affiche le tableau de bord du modèle de cancer cérébral"""
    
    # En-tête
    st.markdown("""
    <div style='text-align:center;padding:2rem;background:rgba(124,58,237,0.05);
                border-radius:15px;border:1px solid rgba(124,58,237,0.2);margin-bottom:2rem;'>
        <div style='font-size:3rem;margin-bottom:1rem;'>🧠</div>
        <h1 style='color:#f1f5f9;margin-bottom:0.5rem;'>Tableau de Bord — Cancer Cérébral</h1>
        <p style='color:#94a3b8;font-size:1.1rem;'>
        Statistiques et performances du modèle EfficientNet-B0
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Vérifier si le modèle existe
    if not MODEL_PATH.exists():
        st.warning("⚠️ Modèle non entraîné. Lancez d'abord l'entraînement.")
        st.code(
            "cd brain_tumer_deep\n"
            "python train.py --data_dir brain-tumor-mri-dataset --save_dir output",
            language="bash"
        )
        return
    
    # Charger les données
    history = load_training_history()
    model_info = load_model_info()
    
    if not history or not model_info:
        st.error("❌ Impossible de charger les données du modèle.")
        return
    
    # ── KPIs ───────────────────────────────────────────────────────────────────
    st.markdown("### 📊 Métriques Clés")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Récupérer les métriques
    val_acc = model_info.get('val_acc', 0)
    val_loss = model_info.get('val_loss', 0)
    best_epoch = model_info.get('epoch', 0)
    num_classes = len(CLASS_NAMES)
    
    col1.metric("🎯 Accuracy Validation", f"{val_acc*100:.2f}%")
    col2.metric("📉 Loss Validation", f"{val_loss:.4f}")
    col3.metric("🏆 Meilleure Époque", f"{best_epoch}")
    col4.metric("📋 Classes", f"{num_classes}")
    
    st.markdown("---")
    
    # ── Onglets ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Courbes d'Entraînement",
        "🎯 Matrice de Confusion",
        "📊 Distribution des Classes",
        "🔧 Configuration du Modèle"
    ])
    
    # ── Tab 1: Courbes d'Entraînement ──────────────────────────────────────────
    with tab1:
        section_label("📈 Évolution de l'Entraînement")
        
        if history:
            # Créer un DataFrame à partir de l'historique
            epochs = list(range(1, len(history['train_loss']) + 1))
            
            df_history = pd.DataFrame({
                'Époque': epochs,
                'Train Loss': history['train_loss'],
                'Val Loss': history['val_loss'],
                'Train Acc': [acc * 100 for acc in history['train_acc']],
                'Val Acc': [acc * 100 for acc in history['val_acc']]
            })
            
            # Graphique Loss
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📉 Loss (Perte)")
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=df_history['Époque'],
                    y=df_history['Train Loss'],
                    mode='lines+markers',
                    name='Train',
                    line=dict(color='#00d4ff', width=2),
                    marker=dict(size=6)
                ))
                fig_loss.add_trace(go.Scatter(
                    x=df_history['Époque'],
                    y=df_history['Val Loss'],
                    mode='lines+markers',
                    name='Validation',
                    line=dict(color='#7c3aed', width=2),
                    marker=dict(size=6)
                ))
                fig_loss.update_layout(
                    height=350,
                    xaxis_title="Époque",
                    yaxis_title="Loss",
                    **DARK_LAYOUT
                )
                st.plotly_chart(fig_loss, use_container_width=True)
            
            with col2:
                st.markdown("#### 🎯 Accuracy (Précision)")
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    x=df_history['Époque'],
                    y=df_history['Train Acc'],
                    mode='lines+markers',
                    name='Train',
                    line=dict(color='#10b981', width=2),
                    marker=dict(size=6)
                ))
                fig_acc.add_trace(go.Scatter(
                    x=df_history['Époque'],
                    y=df_history['Val Acc'],
                    mode='lines+markers',
                    name='Validation',
                    line=dict(color='#f59e0b', width=2),
                    marker=dict(size=6)
                ))
                fig_acc.update_layout(
                    height=350,
                    xaxis_title="Époque",
                    yaxis_title="Accuracy (%)",
                    **DARK_LAYOUT
                )
                st.plotly_chart(fig_acc, use_container_width=True)
            
            # Tableau récapitulatif
            st.markdown("#### 📋 Résumé par Époque")
            st.dataframe(
                df_history.style.format({
                    'Train Loss': '{:.4f}',
                    'Val Loss': '{:.4f}',
                    'Train Acc': '{:.2f}%',
                    'Val Acc': '{:.2f}%'
                }),
                use_container_width=True,
                hide_index=True
            )
    
    # ── Tab 2: Matrice de Confusion ────────────────────────────────────────────
    with tab2:
        section_label("🎯 Matrice de Confusion")
        
        # Vérifier si l'image existe
        confusion_matrix_path = MODEL_DIR / "confusion_matrix.png"
        
        if confusion_matrix_path.exists():
            from PIL import Image
            img = Image.open(confusion_matrix_path)
            st.image(img, caption="Matrice de Confusion sur l'ensemble de validation", use_column_width=True)
        else:
            st.info("📊 Matrice de confusion non disponible. Elle sera générée lors du prochain entraînement.")
        
        # Informations sur les classes
        st.markdown("#### 📋 Classes du Modèle")
        
        for cls in CLASS_NAMES:
            color = CLASS_COLORS[cls]
            label = CLASS_LABELS_FR[cls]
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:1rem;
                        margin-bottom:0.8rem;padding:0.8rem 1rem;
                        background:rgba(255,255,255,0.03);border-radius:8px;
                        border-left:3px solid {color};'>
                <div style='width:20px;height:20px;background:{color};border-radius:4px;'></div>
                <div style='flex:1;'>
                    <div style='color:#f1f5f9;font-weight:600;'>{label}</div>
                    <div style='color:#94a3b8;font-size:0.85rem;'>Classe: {cls}</div>
                </div>
            </div>""", unsafe_allow_html=True)
    
    # ── Tab 3: Distribution des Classes ────────────────────────────────────────
    with tab3:
        section_label("📊 Distribution des Classes dans le Dataset")
        
        # Informations sur le dataset (à adapter selon vos données)
        st.info("💡 Cette section affichera la distribution des classes dans votre dataset d'entraînement.")
        
        # Exemple de distribution (à remplacer par vos vraies données)
        # Vous pouvez lire ces informations depuis un fichier ou les calculer
        class_distribution = {
            "Gliome": 1321,
            "Méningiome": 1339,
            "Pas de tumeur": 1595,
            "Tumeur hypophysaire": 1457
        }
        
        df_dist = pd.DataFrame({
            'Classe': list(class_distribution.keys()),
            'Nombre': list(class_distribution.values())
        })
        
        # Graphique en barres
        fig_dist = px.bar(
            df_dist,
            x='Classe',
            y='Nombre',
            color='Nombre',
            color_continuous_scale=[[0,'#1e3a5f'],[0.5,'#7c3aed'],[1,'#00d4ff']],
            text='Nombre'
        )
        fig_dist.update_traces(textposition='outside')
        fig_dist.update_layout(
            height=400,
            coloraxis_showscale=False,
            **DARK_LAYOUT
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Graphique en camembert
        fig_pie = px.pie(
            df_dist,
            values='Nombre',
            names='Classe',
            hole=0.4,
            color_discrete_sequence=['#e74c3c', '#f39c12', '#27ae60', '#2980b9']
        )
        fig_pie.update_layout(
            height=400,
            **DARK_LAYOUT
        )
        fig_pie.update_traces(textfont_color='white')
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Statistiques
        total = sum(class_distribution.values())
        st.markdown("#### 📈 Statistiques")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total d'images", f"{total:,}")
        col2.metric("Classe majoritaire", max(class_distribution, key=class_distribution.get))
        col3.metric("Équilibre", f"{(min(class_distribution.values())/max(class_distribution.values())*100):.1f}%")
    
    # ── Tab 4: Configuration du Modèle ─────────────────────────────────────────
    with tab4:
        section_label("🔧 Configuration et Architecture")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏗️ Architecture")
            st.markdown("""
            - **Modèle**: EfficientNet-B0
            - **Type**: Transfer Learning
            - **Framework**: PyTorch
            - **Taille d'entrée**: 224x224x3
            - **Nombre de classes**: 4
            - **Fonction d'activation**: ReLU
            - **Dropout**: 0.4 (première couche), 0.2 (deuxième couche)
            """)
            
            st.markdown("#### 📦 Couches du Classifier")
            st.code("""
nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(256, num_classes)
)
            """, language="python")
        
        with col2:
            st.markdown("#### ⚙️ Hyperparamètres")
            
            # Informations du modèle
            if model_info:
                hyperparams = {
                    "Époque finale": model_info.get('epoch', 'N/A'),
                    "Accuracy validation": f"{model_info.get('val_acc', 0)*100:.2f}%",
                    "Loss validation": f"{model_info.get('val_loss', 0):.4f}",
                    "Optimizer": "Adam (probablement)",
                    "Learning rate": "Variable (scheduler)",
                    "Batch size": "32 (par défaut)",
                    "Augmentation": "Oui (rotation, flip, etc.)"
                }
                
                for key, value in hyperparams.items():
                    st.markdown(f"- **{key}**: {value}")
            
            st.markdown("#### 📊 Métriques de Performance")
            metrics_data = {
                "Métrique": ["Accuracy", "Loss", "Époque"],
                "Train": [
                    f"{history['train_acc'][-1]*100:.2f}%" if history else "N/A",
                    f"{history['train_loss'][-1]:.4f}" if history else "N/A",
                    f"{len(history['train_loss'])}" if history else "N/A"
                ],
                "Validation": [
                    f"{val_acc*100:.2f}%",
                    f"{val_loss:.4f}",
                    f"{best_epoch}"
                ]
            }
            
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        
        # Informations sur le fichier du modèle
        st.markdown("---")
        st.markdown("#### 📁 Informations du Fichier")
        
        if MODEL_PATH.exists():
            file_size = MODEL_PATH.stat().st_size / (1024 * 1024)  # MB
            st.markdown(f"""
            - **Chemin**: `{MODEL_PATH}`
            - **Taille**: {file_size:.2f} MB
            - **Format**: PyTorch (.pth)
            """)
        
        # Courbes ROC si disponibles
        roc_path = MODEL_DIR / "roc_curves.png"
        if roc_path.exists():
            st.markdown("#### 📈 Courbes ROC")
            from PIL import Image
            img = Image.open(roc_path)
            st.image(img, caption="Courbes ROC pour chaque classe", use_column_width=True)
    
    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;padding:1rem;background:rgba(0,212,255,0.05);
                border-radius:10px;border:1px solid rgba(0,212,255,0.15);'>
        <div style='color:#94a3b8;font-size:0.85rem;'>
            🧠 <strong>NovaClinic Dashboard</strong> — Modèle EfficientNet-B0
        </div>
        <div style='color:#64748b;font-size:0.75rem;margin-top:0.3rem;'>
            Outil d'aide à la décision médicale — Ne remplace pas un diagnostic professionnel
        </div>
    </div>
    """, unsafe_allow_html=True)
