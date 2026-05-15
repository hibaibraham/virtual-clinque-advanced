import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.core import section_label, DARK_LAYOUT

@st.cache_data
def generate_mock_ptdm_data(n_samples=500):
    """Générer des données simulées pour la démonstration du dashboard PTDM"""
    np.random.seed(42)
    
    # Génération de données réalistes
    age = np.random.normal(50, 15, n_samples).clip(18, 80).astype(int)
    sexe = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    obesite = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    hta = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    
    # Marqueurs corrélés à l'obésité et à l'âge
    base_glycemie = 0.9 + (obesite * 0.2) + (age / 200)
    glycemie = np.random.normal(base_glycemie, 0.2, n_samples).clip(0.6, 3.0)
    
    base_hba1c = 5.0 + (glycemie - 0.9) * 2
    hba1c = np.random.normal(base_hba1c, 0.5, n_samples).clip(4.0, 12.0)
    
    duree_dialyse = np.random.exponential(3, n_samples).clip(0, 15)
    age_donneur = np.random.normal(45, 12, n_samples).clip(18, 80).astype(int)
    
    # Calcul de la probabilité PTDM
    risk_score = (
        (hba1c > 6.0) * 0.4 +
        (glycemie > 1.2) * 0.3 +
        obesite * 0.15 +
        (age > 50) * 0.1 +
        (duree_dialyse > 5) * 0.05
    )
    
    ptdm_class = (risk_score + np.random.normal(0, 0.2, n_samples) > 0.5).astype(int)
    
    df = pd.DataFrame({
        'age_receveur_TR': age,
        'sexe_receveur_M': sexe,
        'obésité_pre_TR_receveur': obesite,
        'HTA_pre_TR_receveur': hta,
        'glycémie_pre_TR_R': glycemie,
        'HbA1c_pre_TR_R': hba1c,
        'durée_dialyse_année': duree_dialyse,
        'age_donneur': age_donneur,
        'PTDM': ptdm_class
    })
    
    return df

def render():
    st.info("💡 **Mode Démonstration :** En l'absence du dataset original (Data1_2026.csv), ce tableau de bord affiche des données cliniques simulées.")
    
    df_raw = generate_mock_ptdm_data()

    try:
        from models.model_manager import ModelManager
        manager = ModelManager.get_cached_manager()
        model = manager.get_model('ptdm')
        if not model.loaded:
            manager.load_all_models()
        config = model.config if model.loaded else {}
        model_loaded = model.loaded
    except Exception:
        config = {}
        model_loaded = False

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Échantillons (Simulés)", f"{len(df_raw):,}")
    c2.metric("📐 Features", f"{df_raw.shape[1] - 1}")
    if model_loaded:
        c3.metric("🎯 Accuracy (Modèle)", f"{config.get('test_accuracy', 0):.1%}")
        c4.metric("📈 AUC Score", f"{config.get('test_auc', 0):.1%}")

    st.markdown("---")

    # Filtres dashboard
    with st.expander("🔍 Filtres Cohorte", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        age_range = fc1.slider("Âge Receveur", 18, 80, (18, 80))
        hba1c_max = fc2.slider("HbA1c max (%)", 4.0, 12.0, 12.0)
        sexe_filter = fc3.multiselect("Sexe (1=M, 0=F)", [0, 1], default=[0, 1])
        
        df_filtered = df_raw[
            (df_raw['age_receveur_TR'].between(*age_range)) &
            (df_raw['HbA1c_pre_TR_R'] <= hba1c_max) &
            (df_raw['sexe_receveur_M'].isin(sexe_filter))
        ]

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Prévalence", "🔗 Corrélations", "📊 Démographie", "🤖 Modèle"])

    # ── Tab 1: Prévalence ─────────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns(2)

        binary_counts = pd.DataFrame({
            'Diagnostic': ['Sans PTDM', 'Avec PTDM'],
            'Nombre': [int((df_filtered['PTDM'] == 0).sum()),
                       int((df_filtered['PTDM'] == 1).sum())]
        })
        
        fig_pie = px.pie(binary_counts, values='Nombre', names='Diagnostic', hole=0.55,
                         color='Diagnostic',
                         color_discrete_map={'Sans PTDM': '#10b981', 'Avec PTDM': '#ef4444'})
        fig_pie.update_layout(height=380, **DARK_LAYOUT)
        fig_pie.update_traces(textfont_color='white')
        c1.plotly_chart(fig_pie, width='stretch')

        # Distribution de la glycémie par classe
        fig_hist = px.histogram(df_filtered, x='glycémie_pre_TR_R', color='PTDM',
                               barmode='overlay', opacity=0.7,
                               color_discrete_map={0: '#10b981', 1: '#ef4444'},
                               title="Distribution de la Glycémie")
        fig_hist.update_layout(height=380, **DARK_LAYOUT)
        c2.plotly_chart(fig_hist, width='stretch')

    # ── Tab 2: Corrélations ─────────────────────────────────────────────
    with tab2:
        num_cols = ['age_receveur_TR', 'glycémie_pre_TR_R', 'HbA1c_pre_TR_R', 'durée_dialyse_année', 'age_donneur']
        corr_matrix = df_filtered[num_cols + ['PTDM']].corr()

        fig_corr = px.imshow(corr_matrix, text_auto='.2f',
                             color_continuous_scale='RdBu_r', aspect='auto')
        fig_corr.update_layout(height=450, **DARK_LAYOUT)
        st.plotly_chart(fig_corr, width='stretch')

        # Scatter interactif
        section_label("📉 Scatter — Explorer les relations cliniques")
        sc1, sc2, sc3 = st.columns(3)
        x_ax = sc1.selectbox("Axe X", num_cols, index=1) # Glycémie
        y_ax = sc2.selectbox("Axe Y", num_cols, index=2) # HbA1c
        color = sc3.selectbox("Couleur", ['PTDM', 'obésité_pre_TR_receveur', 'HTA_pre_TR_receveur'], index=0)

        df_scatter = df_filtered.copy()
        # Convertir en chaîne pour la coloration catégorielle
        df_scatter[color] = df_scatter[color].astype(str)
        
        fig_scatter = px.scatter(df_scatter, x=x_ax, y=y_ax, color=color,
                                 opacity=0.6, marginal_x='histogram', marginal_y='box',
                                 color_discrete_sequence=['#10b981','#ef4444','#00d4ff','#f59e0b'])
        fig_scatter.update_layout(height=480, **DARK_LAYOUT)
        st.plotly_chart(fig_scatter, width='stretch')

    # ── Tab 3: Démographie ─────────────────────────────────────────────
    with tab3:
        c1, c2 = st.columns(2)

        fig_age = px.histogram(df_filtered, x='age_receveur_TR', color='PTDM', nbins=20,
                               color_discrete_map={0: '#10b981', 1: '#ef4444'},
                               title="Âge du receveur et risque PTDM")
        fig_age.update_layout(height=350, **DARK_LAYOUT)
        c1.plotly_chart(fig_age, width='stretch')

        # Box plots
        df_filtered['statut'] = df_filtered['PTDM'].apply(lambda x: 'PTDM' if x == 1 else 'Normal')
        marker_box = c2.selectbox("Marqueur Biologique", ['HbA1c_pre_TR_R', 'glycémie_pre_TR_R', 'durée_dialyse_année'])
        fig_box = px.box(df_filtered, x='statut', y=marker_box,
                         color='statut',
                         color_discrete_map={'Normal': '#10b981', 'PTDM': '#ef4444'})
        fig_box.update_layout(height=350, showlegend=False, **DARK_LAYOUT)
        c2.plotly_chart(fig_box, width='stretch')

    # ── Tab 4: Modèle ─────────────────────────────────────────────
    with tab4:
        if not model_loaded:
            st.warning("Modèle non chargé.")
            return

        c1, c2 = st.columns(2)
        metrics_df = pd.DataFrame({
            'Métrique': ['Accuracy', 'AUC Score', 'Modèle type', 'Source des données'],
            'Valeur':   [f"{config.get('test_accuracy',0):.4f}",
                         f"{config.get('test_auc',0):.4f}",
                         "Random Forest / SVM",
                         "Simulées (Manque Data1_2026.csv)"]
        })
        c1.dataframe(metrics_df, width='stretch', hide_index=True)

        importances = config.get('feature_importances', {})
        if importances:
            imp_df = (pd.DataFrame({'Feature': list(importances.keys()),
                                    'Importance': list(importances.values())})
                      .sort_values('Importance', ascending=False))
            fig_imp = px.bar(imp_df, x='Feature', y='Importance',
                             color='Importance',
                             color_continuous_scale=[[0,'#1e3a5f'],[0.5,'#7c3aed'],[1,'#00d4ff']])
            fig_imp.update_layout(height=400, xaxis_tickangle=-45,
                                  coloraxis_showscale=False, **DARK_LAYOUT)
            st.plotly_chart(fig_imp, width='stretch')
