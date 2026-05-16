import os
import pandas as pd
import plotly.express as px
import streamlit as st
from utils.core import HISTORY_PATH, section_label, DARK_LAYOUT
from utils.database import get_predictions_collection, is_mongodb_available


def render():
    # Get current user
    username = st.session_state.get("auth_username", "anonymous")
    
    # Try MongoDB first
    df = None
    data_source = "local"
    
    if is_mongodb_available():
        collection = get_predictions_collection()
        if collection is not None:
            try:
                predictions = list(collection.find({}))
                if predictions:
                    # Remove MongoDB _id
                    for pred in predictions:
                        pred.pop('_id', None)
                    df = pd.DataFrame(predictions)
                    data_source = "mongodb"
            except Exception as e:
                st.warning(f"⚠️ Erreur MongoDB, utilisation du CSV local: {e}")
    
    # Fallback to local CSV
    if df is None or len(df) == 0:
        if not os.path.exists(HISTORY_PATH):
            st.markdown("""
            <div style='text-align:center;padding:3rem;color:#94a3b8;'>
                <div style='font-size:3rem;margin-bottom:1rem;'>📭</div>
                <div style='font-size:1.1rem;color:#f1f5f9;'>Aucune prédiction enregistrée</div>
                <div style='margin-top:0.5rem;font-size:0.85rem;'>Effectuez des analyses pour voir l'historique</div>
            </div>""", unsafe_allow_html=True)
            return
        
        df = pd.read_csv(HISTORY_PATH)
        data_source = "local"
    
    if len(df) == 0:
        st.info("Aucune donnée.")
        return
    
    # Add model type if not present
    if 'model_type' not in df.columns:
        df['model_type'] = 'thyroid'  # Default for old data
    
    # Data source indicator
    if data_source == "mongodb":
        st.success("☁️ Données chargées depuis MongoDB")
    else:
        st.info("💾 Données chargées depuis le stockage local (CSV)")

    # Tabs for different models
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Vue Globale", "🦋 Thyroïde", "🧠 Brain Cancer", "🩸 PTDM"])
    
    # ── Tab 1: Vue Globale ────────────────────────────────────────────────────
    with tab1:
        # KPIs globaux
        col1, col2, col3, col4 = st.columns(4)
        
        total_predictions = len(df)
        thyroid_count = len(df[df['model_type'] == 'thyroid'])
        brain_count = len(df[df['model_type'] == 'brain_cancer'])
        ptdm_count = len(df[df['model_type'] == 'ptdm'])
        
        col1.metric("📊 Total Prédictions", total_predictions)
        col2.metric("🦋 Thyroïde", thyroid_count)
        col3.metric("🧠 Brain Cancer", brain_count)
        col4.metric("🩸 PTDM", ptdm_count)
        
        st.markdown("---")
        
        # Distribution par modèle
        section_label("📈 Distribution par Modèle")
        model_counts = df['model_type'].value_counts().reset_index()
        model_counts.columns = ['Modèle', 'Nombre']
        
        # Map model names
        model_names = {
            'thyroid': '🦋 Thyroïde',
            'brain_cancer': '🧠 Brain Cancer',
            'ptdm': '🩸 PTDM'
        }
        model_counts['Modèle'] = model_counts['Modèle'].map(model_names)
        
        fig_models = px.bar(model_counts, x='Modèle', y='Nombre',
                           color='Modèle',
                           color_discrete_map={
                               '🦋 Thyroïde': '#8b5cf6',
                               '🧠 Brain Cancer': '#00d4ff',
                               '🩸 PTDM': '#ef4444'
                           })
        fig_models.update_layout(height=300, showlegend=False, **DARK_LAYOUT)
        st.plotly_chart(fig_models, use_container_width=True)
        
        # Tendance temporelle globale
        if 'timestamp' in df.columns:
            section_label("📅 Évolution Temporelle")
            df_time = df.dropna(subset=['timestamp']).copy()
            if len(df_time) > 1:
                df_time['timestamp'] = pd.to_datetime(df_time['timestamp'], errors='coerce')
                df_time = df_time.dropna(subset=['timestamp'])
                if len(df_time) > 1:
                    df_time['date'] = df_time['timestamp'].dt.date
                    trend = df_time.groupby(['date', 'model_type']).size().reset_index(name='count')
                    trend['model_type'] = trend['model_type'].map(model_names)
                    
                    fig_trend = px.line(trend, x='date', y='count', color='model_type',
                                       markers=True,
                                       color_discrete_map={
                                           '🦋 Thyroïde': '#8b5cf6',
                                           '🧠 Brain Cancer': '#00d4ff',
                                           '🩸 PTDM': '#ef4444'
                                       })
                    fig_trend.update_layout(height=280, **DARK_LAYOUT)
                    st.plotly_chart(fig_trend, use_container_width=True)
    
    # ── Tab 2: Thyroïde ───────────────────────────────────────────────────────
    with tab2:
        df_thyroid = df[df['model_type'] == 'thyroid']
        
        if len(df_thyroid) == 0:
            st.info("Aucune prédiction thyroïde enregistrée.")
        else:
            # KPIs
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📊 Total", len(df_thyroid))
            patho_count = (df_thyroid['prediction'] == 'Pathologique').sum() if 'prediction' in df_thyroid.columns else 0
            normal_count = (df_thyroid['prediction'] == 'Normal').sum() if 'prediction' in df_thyroid.columns else 0
            c2.metric("⚠️ Pathologiques", patho_count)
            c3.metric("✅ Normaux", normal_count)
            rate = patho_count / len(df_thyroid) if len(df_thyroid) > 0 else 0
            c4.metric("📈 Taux Patho", f"{rate:.1%}")
            
            st.markdown("---")
            
            # Courbe d'évolution temporelle
            section_label("📅 Évolution Temporelle")
            if 'timestamp' in df_thyroid.columns:
                df_time = df_thyroid.dropna(subset=['timestamp']).copy()
                if len(df_time) > 1:
                    df_time['timestamp'] = pd.to_datetime(df_time['timestamp'], errors='coerce')
                    df_time = df_time.dropna(subset=['timestamp'])
                    if len(df_time) > 1:
                        df_time['date'] = df_time['timestamp'].dt.date
                        
                        # Convertir les prédictions numériques en texte
                        if df_time['prediction'].dtype in ['int64', 'float64']:
                            df_time['prediction'] = df_time['prediction'].apply(
                                lambda x: 'Pathologique' if x == 1 else 'Normal'
                            )
                        
                        trend = df_time.groupby(['date', 'prediction']).size().reset_index(name='count')
                        fig_trend = px.line(trend, x='date', y='count', color='prediction',
                                          color_discrete_map={'Normal': '#10b981', 'Pathologique': '#ef4444'},
                                          markers=True,
                                          title="Nombre de prédictions par jour")
                        fig_trend.update_layout(height=280, **DARK_LAYOUT)
                        st.plotly_chart(fig_trend, use_container_width=True)
            
            st.markdown("---")
            
            # Filtres
            section_label("🔍 Filtres de Recherche")
            fc1, fc2, fc3 = st.columns(3)
            
            filter_diag = fc1.selectbox("Diagnostic", ["Tous", "Normal", "Pathologique"], key="filter_thyroid")
            
            # Sélecteur de date avec calendrier
            from datetime import date, timedelta
            date_debut = fc2.date_input("📅 Du (date)", value=None, key="date_debut_thyroid", help="Date de début de la période")
            date_fin = fc3.date_input("📅 Au (date)", value=None, key="date_fin_thyroid", help="Date de fin de la période")
            
            df_filtered = df_thyroid.copy()
            
            # Filtre par diagnostic
            if filter_diag != "Tous" and 'prediction' in df_filtered.columns:
                # Convertir les prédictions numériques en texte pour le filtre
                if df_filtered['prediction'].dtype in ['int64', 'float64']:
                    df_filtered['prediction_text'] = df_filtered['prediction'].apply(
                        lambda x: 'Pathologique' if x == 1 else 'Normal'
                    )
                    df_filtered = df_filtered[df_filtered['prediction_text'] == filter_diag]
                else:
                    df_filtered = df_filtered[df_filtered['prediction'] == filter_diag]
            
            # Filtre par date
            if 'timestamp' in df_filtered.columns and (date_debut or date_fin):
                df_filtered['timestamp_dt'] = pd.to_datetime(df_filtered['timestamp'], errors='coerce')
                if date_debut:
                    df_filtered = df_filtered[df_filtered['timestamp_dt'].dt.date >= date_debut]
                if date_fin:
                    df_filtered = df_filtered[df_filtered['timestamp_dt'].dt.date <= date_fin]
            
            section_label(f"📋 Registre Thyroïde — {len(df_filtered)} entrée(s)")
            
            # Display columns avec nom du patient
            cols_show = ['timestamp', 'patient_name', 'prediction', 'probability', 'age', 'TSH', 'T3', 'TT4', 'FTI']
            
            # Format prediction column for display
            if 'prediction' in df_filtered.columns:
                df_display = df_filtered.copy()
                # Convert numeric predictions to text if needed
                if df_display['prediction'].dtype in ['int64', 'float64']:
                    df_display['prediction'] = df_display['prediction'].apply(
                        lambda x: 'Pathologique' if x == 1 else 'Normal'
                    )
            else:
                df_display = df_filtered
            
            cols_avail = [c for c in cols_show if c in df_display.columns]
            
            if len(cols_avail) > 0:
                st.dataframe(df_display[cols_avail], use_container_width=True, hide_index=True)
            else:
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # Export
            st.markdown("---")
            csv_data = df_filtered.to_csv(index=False)
            st.download_button("📥 Exporter les données (CSV)", csv_data,
                             file_name="historique_thyroide.csv", mime="text/csv",
                             use_container_width=True)
    
    # ── Tab 3: Brain Cancer ───────────────────────────────────────────────────
    with tab3:
        df_brain = df[df['model_type'] == 'brain_cancer']
        
        if len(df_brain) == 0:
            st.info("Aucune prédiction brain cancer enregistrée.")
        else:
            # KPIs
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📊 Total", len(df_brain))
            
            # Count by class
            if 'prediction' in df_brain.columns:
                class_counts = df_brain['prediction'].value_counts()
                c2.metric("🔴 Glioma", class_counts.get('glioma', 0))
                c3.metric("🟢 Meningioma", class_counts.get('meningioma', 0))
                c4.metric("🔵 Pituitary", class_counts.get('pituitary', 0))
            
            st.markdown("---")
            
            # Courbe d'évolution temporelle
            section_label("📅 Évolution Temporelle")
            if 'timestamp' in df_brain.columns:
                df_time = df_brain.dropna(subset=['timestamp']).copy()
                if len(df_time) > 1:
                    df_time['timestamp'] = pd.to_datetime(df_time['timestamp'], errors='coerce')
                    df_time = df_time.dropna(subset=['timestamp'])
                    if len(df_time) > 1:
                        df_time['date'] = df_time['timestamp'].dt.date
                        trend = df_time.groupby(['date', 'prediction']).size().reset_index(name='count')
                        
                        fig_trend = px.line(trend, x='date', y='count', color='prediction',
                                          markers=True,
                                          title="Nombre d'analyses IRM par jour",
                                          color_discrete_map={
                                              'glioma': '#ef4444',
                                              'meningioma': '#10b981',
                                              'pituitary': '#3b82f6',
                                              'notumor': '#94a3b8'
                                          })
                        fig_trend.update_layout(height=280, **DARK_LAYOUT)
                        st.plotly_chart(fig_trend, use_container_width=True)
            
            st.markdown("---")
            
            # Filtres
            section_label("🔍 Filtres de Recherche")
            fc1, fc2, fc3 = st.columns(3)
            
            classes = ["Tous"] + df_brain['prediction'].unique().tolist() if 'prediction' in df_brain.columns else ["Tous"]
            filter_class = fc1.selectbox("Type de tumeur", classes, key="filter_brain")
            
            # Sélecteur de date avec calendrier
            from datetime import date, timedelta
            date_debut = fc2.date_input("📅 Du (date)", value=None, key="date_debut_brain", help="Date de début de la période")
            date_fin = fc3.date_input("📅 Au (date)", value=None, key="date_fin_brain", help="Date de fin de la période")
            
            df_filtered = df_brain.copy()
            
            # Filtre par classe
            if filter_class != "Tous" and 'prediction' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['prediction'] == filter_class]
            
            # Filtre par date
            if 'timestamp' in df_filtered.columns and (date_debut or date_fin):
                df_filtered['timestamp_dt'] = pd.to_datetime(df_filtered['timestamp'], errors='coerce')
                if date_debut:
                    df_filtered = df_filtered[df_filtered['timestamp_dt'].dt.date >= date_debut]
                if date_fin:
                    df_filtered = df_filtered[df_filtered['timestamp_dt'].dt.date <= date_fin]
            
            section_label(f"📋 Registre Brain Cancer — {len(df_filtered)} entrée(s)")
            
            # Display columns avec nom du patient
            cols_show = ['timestamp', 'patient_name', 'prediction', 'confidence', 'image_name']
            cols_avail = [c for c in cols_show if c in df_filtered.columns]
            
            if len(cols_avail) > 0:
                st.dataframe(df_filtered[cols_avail], use_container_width=True, hide_index=True)
            else:
                st.dataframe(df_filtered, use_container_width=True, hide_index=True)
            
            # Export
            st.markdown("---")
            csv_data = df_filtered.to_csv(index=False)
            st.download_button("📥 Exporter les données (CSV)", csv_data,
                             file_name="historique_brain_cancer.csv", mime="text/csv",
                             use_container_width=True)
    
    # ── Tab 4: PTDM ───────────────────────────────────────────────────────────
    with tab4:
        df_ptdm = df[df['model_type'] == 'ptdm']
        
        if len(df_ptdm) == 0:
            st.info("Aucune prédiction PTDM enregistrée.")
        else:
            # KPIs
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📊 Total", len(df_ptdm))
            
            if 'prediction' in df_ptdm.columns:
                risque_eleve = (df_ptdm['prediction'] == 1).sum() if df_ptdm['prediction'].dtype in ['int64', 'float64'] else 0
                risque_faible = (df_ptdm['prediction'] == 0).sum() if df_ptdm['prediction'].dtype in ['int64', 'float64'] else 0
                c2.metric("⚠️ Risque Élevé", risque_eleve)
                c3.metric("✅ Risque Faible", risque_faible)
                rate = risque_eleve / len(df_ptdm) if len(df_ptdm) > 0 else 0
                c4.metric("📈 Taux Risque", f"{rate:.1%}")
            
            st.markdown("---")
            
            # Courbe d'évolution temporelle
            section_label("📅 Évolution Temporelle")
            if 'timestamp' in df_ptdm.columns:
                df_time = df_ptdm.dropna(subset=['timestamp']).copy()
                if len(df_time) > 1:
                    df_time['timestamp'] = pd.to_datetime(df_time['timestamp'], errors='coerce')
                    df_time = df_time.dropna(subset=['timestamp'])
                    if len(df_time) > 1:
                        df_time['date'] = df_time['timestamp'].dt.date
                        
                        # Convertir les prédictions numériques en texte
                        if df_time['prediction'].dtype in ['int64', 'float64']:
                            df_time['prediction_text'] = df_time['prediction'].apply(
                                lambda x: 'Risque Élevé' if x == 1 else 'Risque Faible'
                            )
                        else:
                            df_time['prediction_text'] = df_time['prediction']
                        
                        trend = df_time.groupby(['date', 'prediction_text']).size().reset_index(name='count')
                        fig_trend = px.line(trend, x='date', y='count', color='prediction_text',
                                          color_discrete_map={'Risque Faible': '#10b981', 'Risque Élevé': '#ef4444'},
                                          markers=True,
                                          title="Nombre d'évaluations PTDM par jour")
                        fig_trend.update_layout(height=280, **DARK_LAYOUT)
                        st.plotly_chart(fig_trend, use_container_width=True)
            
            st.markdown("---")
            
            # Filtres
            section_label("🔍 Filtres de Recherche")
            fc1, fc2, fc3 = st.columns(3)
            
            filter_risk = fc1.selectbox("Niveau de risque", ["Tous", "Risque Élevé", "Risque Faible"], key="filter_ptdm")
            
            # Sélecteur de date avec calendrier
            from datetime import date, timedelta
            date_debut = fc2.date_input("📅 Du (date)", value=None, key="date_debut_ptdm", help="Date de début de la période")
            date_fin = fc3.date_input("📅 Au (date)", value=None, key="date_fin_ptdm", help="Date de fin de la période")
            
            df_filtered = df_ptdm.copy()
            
            # Filtre par risque
            if filter_risk != "Tous" and 'prediction' in df_filtered.columns:
                if filter_risk == "Risque Élevé":
                    df_filtered = df_filtered[df_filtered['prediction'] == 1]
                else:
                    df_filtered = df_filtered[df_filtered['prediction'] == 0]
            
            # Filtre par date
            if 'timestamp' in df_filtered.columns and (date_debut or date_fin):
                df_filtered['timestamp_dt'] = pd.to_datetime(df_filtered['timestamp'], errors='coerce')
                if date_debut:
                    df_filtered = df_filtered[df_filtered['timestamp_dt'].dt.date >= date_debut]
                if date_fin:
                    df_filtered = df_filtered[df_filtered['timestamp_dt'].dt.date <= date_fin]
            
            section_label(f"📋 Registre PTDM — {len(df_filtered)} entrée(s)")
            
            # Display columns avec nom du patient
            cols_show = ['timestamp', 'patient_name', 'prediction', 'probability', 'age_receveur_TR', 'HbA1c_pre_TR_R', 'glycémie_pre_TR_R']
            
            # Format prediction column for display
            if 'prediction' in df_filtered.columns:
                df_display = df_filtered.copy()
                # Convert numeric predictions to text
                if df_display['prediction'].dtype in ['int64', 'float64']:
                    df_display['prediction'] = df_display['prediction'].apply(
                        lambda x: 'Risque Élevé' if x == 1 else 'Risque Faible'
                    )
            else:
                df_display = df_filtered
            
            cols_avail = [c for c in cols_show if c in df_display.columns]
            
            if len(cols_avail) > 0:
                st.dataframe(df_display[cols_avail], use_container_width=True, hide_index=True)
            else:
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # Export
            st.markdown("---")
            csv_data = df_filtered.to_csv(index=False)
            st.download_button("📥 Exporter les données (CSV)", csv_data,
                             file_name="historique_ptdm.csv", mime="text/csv",
                             use_container_width=True)
    
    # Actions globales
    st.markdown("---")
    st.markdown("### ⚙️ Actions Globales")
    
    col1, col2 = st.columns(2)
    
    # Export all
    with col1:
        csv_all = df.to_csv(index=False)
        st.download_button("📥 Exporter Tout", csv_all,
                          file_name="historique_complet_novaclinic.csv", 
                          mime="text/csv",
                          use_container_width=True,
                          help="Télécharger l'historique complet de tous les modèles")
    
    # Delete history
    with col2:
        if st.button("🗑️ Effacer Tout l'Historique", use_container_width=True, type="secondary"):
            if data_source == "mongodb":
                collection = get_predictions_collection()
                if collection is not None:
                    collection.delete_many({})
            if os.path.exists(HISTORY_PATH):
                os.remove(HISTORY_PATH)
            st.success("✅ Historique effacé avec succès !")
            st.rerun()
