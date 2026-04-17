import os
import pandas as pd
import plotly.express as px
import streamlit as st
from utils.core import HISTORY_PATH, section_label, DARK_LAYOUT
from utils.firebase import get_user_predictions_firestore, get_all_predictions_firestore, delete_prediction_firestore, is_firebase_enabled


def render():
    # Get current user
    username = st.session_state.get("auth_username", "anonymous")
    
    # Get predictions from Firebase or local CSV
    if is_firebase_enabled():
        # Use Firebase
        predictions = get_user_predictions_firestore(username, limit=100)
        if not predictions:
            st.markdown("""
            <div style='text-align:center;padding:3rem;color:#94a3b8;'>
                <div style='font-size:3rem;margin-bottom:1rem;'>📭</div>
                <div style='font-size:1.1rem;color:#f1f5f9;'>Aucune prédiction enregistrée</div>
                <div style='margin-top:0.5rem;font-size:0.85rem;'>Allez sur la page Prédiction pour commencer</div>
            </div>""", unsafe_allow_html=True)
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(predictions)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        if 'created_at' in df.columns:
            df['timestamp'] = pd.to_datetime(df['created_at'], errors='coerce')
            
        # Add Firebase indicator
        st.info("📡 Données chargées depuis Firebase Firestore")
    else:
        # Fallback to local CSV
        if not os.path.exists(HISTORY_PATH):
            st.markdown("""
            <div style='text-align:center;padding:3rem;color:#94a3b8;'>
                <div style='font-size:3rem;margin-bottom:1rem;'>📭</div>
                <div style='font-size:1.1rem;color:#f1f5f9;'>Aucune prédiction enregistrée</div>
                <div style='margin-top:0.5rem;font-size:0.85rem;'>Allez sur la page Prédiction pour commencer</div>
            </div>""", unsafe_allow_html=True)
            return

        df = pd.read_csv(HISTORY_PATH)
        if len(df) == 0:
            st.info("Aucune donnée.")
            return
            
        # Add local indicator
        st.info("💾 Données chargées depuis le stockage local")

	# KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Total",        len(df))
    c2.metric("⚠️ Pathologiques", (df['prediction'] == 'Pathologique').sum())
    c3.metric("✅ Normaux",       (df['prediction'] == 'Normal').sum())
    rate = (df['prediction'] == 'Pathologique').mean()
    c4.metric("📈 Taux Patho",   f"{rate:.1%}")

    st.markdown("---")

	# Tendance temporelle
    section_label("📅 Évolution Temporelle")
    if 'timestamp' in df.columns:
        df_time = df.dropna(subset=['timestamp']).copy()
        if len(df_time) > 1:
            df_time['date'] = df_time['timestamp'].dt.date
            trend = (df_time.groupby(['date', 'prediction'])
                     .size().reset_index(name='count'))
            fig_trend = px.line(trend, x='date', y='count', color='prediction',
                                color_discrete_map={'Normal': '#10b981', 'Pathologique': '#ef4444'},
                                markers=True)
            fig_trend.update_layout(height=280, **DARK_LAYOUT)
            st.plotly_chart(fig_trend, width='stretch')

	# Filtres
    fc1, fc2 = st.columns(2)
    filter_diag = fc1.selectbox("Filtrer par diagnostic", ["Tous", "Normal", "Pathologique"])
    search_date = fc2.text_input("Rechercher par date (YYYY-MM-DD)", "")

    df_filtered = df.copy()
    if filter_diag != "Tous":
        df_filtered = df_filtered[df_filtered['prediction'] == filter_diag]
    if search_date and 'timestamp' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['timestamp'].astype(str).str.startswith(search_date)]

    section_label(f"📋 Registre — {len(df_filtered)} entrée(s)")
    cols_show = ['timestamp', 'prediction', 'probability', 'age', 'TSH', 'T3', 'TT4', 'FTI']
    if 'patient_data' in df_filtered.columns:
        # Extract patient data from nested dict
        for col in ['age', 'TSH', 'T3', 'TT4', 'FTI']:
            df_filtered[col] = df_filtered['patient_data'].apply(lambda x: x.get(col, '') if isinstance(x, dict) else '')
    
    cols_avail = [c for c in cols_show if c in df_filtered.columns]
    st.dataframe(df_filtered[cols_avail], width='stretch', hide_index=True)

	# Actions
    c1, c2 = st.columns(2)
    csv_data = df_filtered.to_csv(index=False)
    c1.download_button("📥 Exporter CSV", csv_data,
                       file_name="historique_predictions.csv", mime="text/csv")
    
    if is_firebase_enabled():
        # Firebase delete option
        if c2.button("🗑️ Effacer l'historique Firebase", type="secondary"):
            st.warning("Cette action supprimera toutes vos prédictions de Firebase. Continuer?")
            if st.button("Oui, supprimer définitivement"):
                # Get all user predictions
                preds = get_user_predictions_firestore(username)
                for pred in preds:
                    if 'id' in pred:
                        delete_prediction_firestore(pred['id'])
                st.success("Historique Firebase effacé.")
                st.rerun()
    else:
        # Local CSV delete option
        if c2.button("🗑️ Effacer l'historique"):
            os.remove(HISTORY_PATH)
            st.success("Historique effacé.")
            st.rerun()
