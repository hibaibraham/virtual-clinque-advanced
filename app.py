"""
🏥 Clinique Virtuelle Intelligente — Diagnostic Thyroïdien
Application Streamlit multi-pages avec prédiction ML en temps réel.

Lancer avec : streamlit run app.py
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION DE LA PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🏥 Clinique Virtuelle — Diagnostic Thyroïdien",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS PERSONNALISÉ
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Palette médicale professionnelle */
    :root {
        --medical-blue: #1a73e8;
        --medical-green: #0d904f;
        --medical-red: #d93025;
        --bg-light: #f8f9fa;
    }

    .main-header {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        border-left: 4px solid #1a73e8;
    }

    .result-normal {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 5px solid #0d904f;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }

    .result-pathological {
        background: linear-gradient(135deg, #fce4ec 0%, #f8bbd0 100%);
        border-left: 5px solid #d93025;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }

    .stButton>button {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(26, 115, 232, 0.4);
    }

    .sidebar .sidebar-content {
        background: #f0f4f8;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e3f2fd 0%, #f8f9fa 100%);
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES RESSOURCES
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
DATA_PATH = os.path.join(BASE_DIR, 'thyroid.csv')
HISTORY_PATH = os.path.join(BASE_DIR, 'prediction_history.csv')


@st.cache_resource
def load_model():
    """Charge le modèle ML et le preprocessor sauvegardés."""
    model = joblib.load(os.path.join(SAVE_DIR, 'model.joblib'))
    preprocessor = joblib.load(os.path.join(SAVE_DIR, 'preprocessor.joblib'))
    with open(os.path.join(SAVE_DIR, 'feature_config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)
    return model, preprocessor, config


@st.cache_data
def load_dataset():
    """Charge le dataset thyroïdien pour le tableau de bord."""
    return pd.read_csv(DATA_PATH)


def compute_engineered_features(row):
    """Calcule les features dérivées identiques au pipeline d'entraînement."""
    row['TSH_abnormal'] = int((row.get('TSH', 0) < 0.4) or (row.get('TSH', 0) > 4.0))
    row['TT4_abnormal'] = int((row.get('TT4', 0) < 70) or (row.get('TT4', 0) > 180))
    row['T3_abnormal']  = int((row.get('T3', 0) < 1.2) or (row.get('T3', 0) > 3.1))
    row['FTI_abnormal'] = int((row.get('FTI', 0) < 70) or (row.get('FTI', 0) > 180))
    row['hormone_score'] = row['TSH_abnormal'] + row['TT4_abnormal'] + row['T3_abnormal'] + row['FTI_abnormal']
    row['T4U_TT4_ratio'] = row.get('T4U', 0) / (row.get('TT4', 0) + 1e-6)
    return row


def save_prediction(patient_data, prediction, probability):
    """Sauvegarde une prédiction dans l'historique CSV."""
    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'prediction': 'Pathologique' if prediction == 1 else 'Normal',
        'probability': f"{probability:.1%}",
        **{k: v for k, v in patient_data.items()}
    }
    df_record = pd.DataFrame([record])

    if os.path.exists(HISTORY_PATH):
        df_history = pd.read_csv(HISTORY_PATH)
        df_history = pd.concat([df_history, df_record], ignore_index=True)
    else:
        df_history = df_record

    df_history.to_csv(HISTORY_PATH, index=False)
    return df_history


# ══════════════════════════════════════════════════════════════════════════════
# BARRE LATÉRALE — NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏥 Navigation")
    page = st.radio(
        "Choisir une page :",
        ["🩺 Prédiction", "📊 Tableau de Bord", "📜 Historique", "ℹ️ À Propos"],
        index=0,
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; opacity:0.7; font-size:0.85rem;'>
        <p>🧬 Clinique Virtuelle v1.0</p>
        <p>Modèle : Random Forest Optimisé</p>
        <p>Dataset : Thyroid Disease</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 : PRÉDICTION
# ══════════════════════════════════════════════════════════════════════════════
if page == "🩺 Prédiction":
    st.markdown("""
    <div class="main-header">
        <h1>🩺 Diagnostic Thyroïdien Intelligent</h1>
        <p>Saisissez les données du patient pour obtenir une prédiction automatique</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        model, preprocessor, config = load_model()
    except Exception as e:
        st.error(f"❌ Erreur de chargement du modèle : {e}")
        st.info("💡 Veuillez d'abord exécuter `python train_and_save_model.py` pour entraîner et sauvegarder le modèle.")
        st.stop()

    col_form, col_result = st.columns([3, 2])

    with col_form:
        st.markdown("### 📋 Données du Patient")

        # Informations générales
        with st.expander("👤 Informations Générales", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Âge", 1, 100, 45, help="Âge du patient")
            with c2:
                sex = st.selectbox("Sexe", ["Féminin (F)", "Masculin (M)"], help="Sexe biologique")
                sex_val = 0 if sex.startswith("F") else 1

        # Résultats de laboratoire
        with st.expander("🔬 Résultats de Laboratoire", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                tsh = st.number_input("TSH (mU/L)", 0.0, 600.0, 2.0, 0.1,
                                      help="Plage normale : 0.4 – 4.0 mU/L")
                t3 = st.number_input("T3 (nmol/L)", 0.0, 15.0, 2.0, 0.1,
                                     help="Plage normale : 1.2 – 3.1 nmol/L")
            with c2:
                tt4 = st.number_input("TT4 (nmol/L)", 0.0, 600.0, 110.0, 1.0,
                                      help="Plage normale : 70 – 180 nmol/L")
                t4u = st.number_input("T4U", 0.0, 3.0, 1.0, 0.01,
                                      help="Thyroxine uptake")
            with c3:
                fti = st.number_input("FTI", 0.0, 900.0, 110.0, 1.0,
                                      help="Plage normale : 70 – 180")

        # Antécédents médicaux
        with st.expander("🏥 Antécédents Médicaux", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                on_thyroxine = int(st.checkbox("Sous thyroxine"))
                query_on_thyroxine = int(st.checkbox("Question sur thyroxine"))
                on_antithyroid = int(st.checkbox("Sous antithyroïdien"))
                sick = int(st.checkbox("Malade"))
                pregnant = int(st.checkbox("Enceinte"))
                thyroid_surgery = int(st.checkbox("Chirurgie thyroïdienne"))
                i131_treatment = int(st.checkbox("Traitement I131"))
            with c2:
                query_hypothyroid = int(st.checkbox("Question hypothyroïdie"))
                query_hyperthyroid = int(st.checkbox("Question hyperthyroïdie"))
                lithium = int(st.checkbox("Sous lithium"))
                goitre = int(st.checkbox("Goitre"))
                tumor = int(st.checkbox("Tumeur"))
                hypopituitary = int(st.checkbox("Hypopituitarisme"))
                psych = int(st.checkbox("Trouble psychiatrique"))

        # Bouton prédiction
        st.markdown("")
        predict_btn = st.button("🔮 Prédire le Diagnostic", use_container_width=True)

    with col_result:
        st.markdown("### 🎯 Résultat du Diagnostic")

        if predict_btn:
            # Construire le vecteur patient
            patient = {
                'age': age, 'sex': sex_val,
                'on_thyroxine': on_thyroxine,
                'query_on_thyroxine': query_on_thyroxine,
                'on_antithyroid_medication': on_antithyroid,
                'sick': sick, 'pregnant': pregnant,
                'thyroid_surgery': thyroid_surgery,
                'I131_treatment': i131_treatment,
                'query_hypothyroid': query_hypothyroid,
                'query_hyperthyroid': query_hyperthyroid,
                'lithium': lithium, 'goitre': goitre,
                'tumor': tumor, 'hypopituitary': hypopituitary,
                'psych': psych,
                'TSH': tsh, 'T3': t3, 'TT4': tt4, 'T4U': t4u, 'FTI': fti
            }

            # Feature engineering
            patient = compute_engineered_features(patient)

            # Ordonner les features selon la config
            all_features = config['all_features']
            X_input = pd.DataFrame([[patient.get(f, 0) for f in all_features]], columns=all_features)

            # Preprocessing
            X_processed = preprocessor.transform(X_input)

            # Prédiction
            prediction = model.predict(X_processed)[0]
            probabilities = model.predict_proba(X_processed)[0]
            prob_pathological = probabilities[1]

            # Sauvegarder dans l'historique
            save_prediction(patient, prediction, prob_pathological)

            # Affichage du résultat
            if prediction == 0:
                st.markdown("""
                <div class="result-normal">
                    <h2 style="color:#0d904f; margin:0;">✅ Normal</h2>
                    <p style="margin:0.5rem 0 0 0; font-size:1.1rem;">
                        Aucune pathologie thyroïdienne détectée
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-pathological">
                    <h2 style="color:#d93025; margin:0;">⚠️ Pathologique</h2>
                    <p style="margin:0.5rem 0 0 0; font-size:1.1rem;">
                        Pathologie thyroïdienne potentielle détectée
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Jauge de probabilité
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_pathological * 100,
                title={'text': "Probabilité de Pathologie (%)", 'font': {'size': 16}},
                number={'suffix': '%', 'font': {'size': 28}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#d93025" if prob_pathological > 0.5 else "#0d904f"},
                    'steps': [
                        {'range': [0, 30], 'color': '#e8f5e9'},
                        {'range': [30, 60], 'color': '#fff3e0'},
                        {'range': [60, 100], 'color': '#fce4ec'},
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=50, b=10, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Importance des features (IA explicable)
            st.markdown("### 🧠 Importance des Features")
            importances = config.get('feature_importances', {})
            if importances:
                imp_df = pd.DataFrame({
                    'Feature': list(importances.keys()),
                    'Importance': list(importances.values())
                }).sort_values('Importance', ascending=True).tail(10)

                fig_imp = px.bar(
                    imp_df, x='Importance', y='Feature',
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='Blues',
                    title="Top 10 Features les plus Influentes"
                )
                fig_imp.update_layout(
                    height=350,
                    showlegend=False,
                    margin=dict(t=40, b=10, l=10, r=10),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("👈 Remplissez le formulaire et cliquez sur **Prédire** pour obtenir un diagnostic.")

            # Afficher les plages normales
            st.markdown("### 📏 Plages Normales de Référence")
            ref_data = {
                'Marqueur': ['TSH', 'T3', 'TT4', 'FTI'],
                'Min Normal': [0.4, 1.2, 70, 70],
                'Max Normal': [4.0, 3.1, 180, 180],
                'Unité': ['mU/L', 'nmol/L', 'nmol/L', 'index']
            }
            st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 : TABLEAU DE BORD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Tableau de Bord":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Tableau de Bord — Données Thyroïdiennes</h1>
        <p>Statistiques globales et performance du modèle</p>
    </div>
    """, unsafe_allow_html=True)

    df_raw = load_dataset()

    # Métriques rapides
    try:
        _, _, config = load_model()
        model_loaded = True
    except Exception:
        config = {}
        model_loaded = False

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📊 Échantillons Totaux", f"{len(df_raw):,}")
    with c2:
        st.metric("📐 Nombre de Features", f"{df_raw.shape[1]}")
    with c3:
        if model_loaded:
            st.metric("🎯 Accuracy (Test)", f"{config.get('test_accuracy', 0):.1%}")
    with c4:
        if model_loaded:
            st.metric("📈 F1 Score (Test)", f"{config.get('test_f1', 0):.1%}")

    st.markdown("---")

    # Onglets du dashboard
    tab1, tab2, tab3 = st.tabs(["📈 Distribution", "🔗 Corrélations", "🤖 Modèle"])

    with tab1:
        st.markdown("### Distribution des Classes Thyroïdiennes")

        # Nettoyer la colonne class pour l'affichage
        import re
        def extract_label(val):
            if pd.isna(val):
                return 'Inconnu'
            match = re.match(r'^([^\[]+)', str(val))
            return match.group(1).strip() if match else str(val)

        df_display = df_raw.copy()
        df_display['class_label'] = df_display['class'].apply(extract_label)

        class_counts = df_display['class_label'].value_counts().reset_index()
        class_counts.columns = ['Classe', 'Nombre']

        fig_dist = px.bar(
            class_counts, x='Classe', y='Nombre',
            color='Nombre', color_continuous_scale='Viridis',
            title="Distribution des Diagnostics"
        )
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)

        # Distribution Normal vs Pathologique
        binary_counts = pd.DataFrame({
            'Statut': ['Normal (-)', 'Pathologique'],
            'Nombre': [
                int((df_display['class_label'] == '-').sum()),
                int((df_display['class_label'] != '-').sum())
            ]
        })

        fig_pie = px.pie(
            binary_counts, values='Nombre', names='Statut',
            color='Statut',
            color_discrete_map={'Normal (-)': '#4caf50', 'Pathologique': '#f44336'},
            title="Normal vs Pathologique",
            hole=0.4
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

        # Distribution de l'âge
        df_age = df_raw.copy()
        df_age['age'] = pd.to_numeric(df_age['age'], errors='coerce')
        fig_age = px.histogram(
            df_age.dropna(subset=['age']), x='age', nbins=30,
            title="Distribution de l'Âge des Patients",
            color_discrete_sequence=['#1a73e8']
        )
        fig_age.update_layout(height=350)
        st.plotly_chart(fig_age, use_container_width=True)

    with tab2:
        st.markdown("### Corrélations entre les Variables")

        # Préparer un sous-ensemble numérique
        df_corr = df_raw.copy()
        df_corr.replace('?', np.nan, inplace=True)
        num_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
        for c in num_cols:
            df_corr[c] = pd.to_numeric(df_corr[c], errors='coerce')

        corr_matrix = df_corr[num_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title="Matrice de Corrélation — Variables Numériques",
            aspect='auto'
        )
        fig_corr.update_layout(height=450)
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        st.markdown("### Performance du Modèle")

        if model_loaded:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📋 Métriques")
                metrics_df = pd.DataFrame({
                    'Métrique': ['Accuracy (Test)', 'F1 Score (Test)', 'F1 Score (CV)',
                                 'Échantillons Train', 'Échantillons Test'],
                    'Valeur': [
                        f"{config.get('test_accuracy', 0):.4f}",
                        f"{config.get('test_f1', 0):.4f}",
                        f"{config.get('cv_f1', 0):.4f}",
                        str(config.get('train_samples', 'N/A')),
                        str(config.get('test_samples', 'N/A'))
                    ]
                })
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            with col2:
                st.markdown("#### ⚙️ Hyperparamètres Optimaux")
                params = config.get('best_params', {})
                params_df = pd.DataFrame({
                    'Paramètre': list(params.keys()),
                    'Valeur': list(params.values())
                })
                st.dataframe(params_df, use_container_width=True, hide_index=True)

            # Feature importances
            importances = config.get('feature_importances', {})
            if importances:
                imp_df = pd.DataFrame({
                    'Feature': list(importances.keys()),
                    'Importance': list(importances.values())
                }).sort_values('Importance', ascending=False)

                fig_imp2 = px.bar(
                    imp_df, x='Feature', y='Importance',
                    color='Importance', color_continuous_scale='Viridis',
                    title="Importance de toutes les Features"
                )
                fig_imp2.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig_imp2, use_container_width=True)
        else:
            st.warning("⚠️ Modèle non chargé. Veuillez exécuter `python train_and_save_model.py` d'abord.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 : HISTORIQUE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📜 Historique":
    st.markdown("""
    <div class="main-header">
        <h1>📜 Historique des Prédictions</h1>
        <p>Consultez toutes les prédictions effectuées par le système</p>
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists(HISTORY_PATH):
        df_history = pd.read_csv(HISTORY_PATH)

        if len(df_history) > 0:
            # Statistiques rapides
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("📊 Total Prédictions", len(df_history))
            with c2:
                n_patho = (df_history['prediction'] == 'Pathologique').sum()
                st.metric("⚠️ Pathologiques", n_patho)
            with c3:
                n_normal = (df_history['prediction'] == 'Normal').sum()
                st.metric("✅ Normaux", n_normal)

            st.markdown("---")

            # Filtre
            filter_option = st.selectbox(
                "Filtrer par diagnostic :",
                ["Tous", "Normal", "Pathologique"]
            )

            if filter_option != "Tous":
                df_filtered = df_history[df_history['prediction'] == filter_option]
            else:
                df_filtered = df_history

            st.dataframe(df_filtered, use_container_width=True, hide_index=True)

            # Bouton de téléchargement
            csv_data = df_filtered.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger l'historique (CSV)",
                data=csv_data,
                file_name="historique_predictions.csv",
                mime="text/csv"
            )

            # Bouton d'effacement
            if st.button("🗑️ Effacer l'historique"):
                os.remove(HISTORY_PATH)
                st.success("Historique effacé !")
                st.rerun()
        else:
            st.info("📭 Aucune prédiction enregistrée pour le moment.")
    else:
        st.info("📭 Aucune prédiction enregistrée pour le moment. Allez sur la page **Prédiction** pour commencer.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 : À PROPOS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ À Propos":
    st.markdown("""
    <div class="main-header">
        <h1>ℹ️ À Propos de la Clinique Virtuelle</h1>
        <p>Informations sur le système de diagnostic intelligent</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🧬 Le Modèle

        Cette application utilise un modèle de **Random Forest optimisé** pour prédire
        la présence de pathologies thyroïdiennes à partir de données cliniques.

        **Pipeline ML :**
        1. **Nettoyage** : Remplacement des valeurs manquantes, conversion des types
        2. **Feature Engineering** : Création de variables dérivées (seuils cliniques)
        3. **Prétraitement** : StandardScaler (numériques) + SimpleImputer (binaires)
        4. **SMOTE** : Équilibrage des classes (26% → 50% pathologiques)
        5. **Optimisation** : RandomizedSearchCV avec 30 combinaisons de paramètres

        **Modèles testés :**
        - Logistic Regression (~89% F1)
        - Random Forest (~94% F1) ← **Sélectionné**
        - XGBoost (~94% F1)
        """)

    with col2:
        st.markdown("""
        ### 📏 Plages Normales de Référence

        | Marqueur | Min | Max | Unité |
        |----------|-----|-----|-------|
        | TSH | 0.4 | 4.0 | mU/L |
        | T3 | 1.2 | 3.1 | nmol/L |
        | TT4 | 70 | 180 | nmol/L |
        | FTI | 70 | 180 | index |

        ### ⚠️ Avertissement

        Ce système est un **outil d'aide à la décision** et ne remplace en aucun cas
        un diagnostic médical professionnel. Les résultats doivent toujours être
        interprétés par un professionnel de santé qualifié.

        ### 🛠️ Technologies

        - **Frontend** : Streamlit
        - **ML** : scikit-learn, Random Forest
        - **Data** : pandas, numpy
        - **Visualisation** : Plotly, Matplotlib
        """)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; padding:1rem; opacity:0.7;'>
        <p>🏥 Clinique Virtuelle Intelligente v1.0 — Diagnostic Thyroïdien</p>
        <p>Développé avec ❤️ en utilisant Streamlit & scikit-learn</p>
    </div>
    """, unsafe_allow_html=True)
