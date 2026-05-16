import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from utils.core import (
	confidence_badge, input_indicator, section_label, DARK_LAYOUT
)

def render():
    try:
        from models.model_manager import ModelManager
        manager = ModelManager.get_cached_manager()
        
        # Charger uniquement le modèle PTDM, pas tous les modèles
        model = manager.get_model('ptdm')
        if not model.loaded:
            model.load()  # Charger seulement PTDM
            
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle PTDM : {e}")
        st.info("💡 Le modèle PTDM utilise un algorithme de Machine Learning simple qui ne nécessite pas de dépendances lourdes.")
        return

    col_form, col_result = st.columns([3, 2], gap="large")

    # ── Formulaire ─────────────────────────────────────────────
    with col_form:
        section_label("📋 Dossier Patient (Transplantation)")

        with st.expander("👤 Informations Générales Receveur", expanded=True):
            c1, c2 = st.columns(2)
            age = c1.slider("Âge Receveur", 18, 80, 45)
            sex_str = c2.selectbox("Sexe Receveur", ["Masculin (M)", "Féminin (F)"])
            sex_val = 1 if sex_str.startswith("M") else 0

        with st.expander("🩺 Paramètres Cliniques", expanded=True):
            c1, c2 = st.columns(2)
            obesite_str = c1.selectbox("Obésité Pré-TR", ["Non", "Oui"])
            obesite_val = 1 if obesite_str == "Oui" else 0
            
            hta_str = c2.selectbox("HTA Pré-TR", ["Non", "Oui"])
            hta_val = 1 if hta_str == "Oui" else 0
            
            duree_dialyse = st.number_input("Durée Dialyse (années)", 0.0, 15.0, 2.0, 0.5)

        with st.expander("🔬 Marqueurs Biologiques", expanded=True):
            c1, c2 = st.columns(2)
            glycemie = c1.number_input("Glycémie Pré-TR (g/L)", 0.5, 3.0, 1.0, 0.05)
            hba1c = c2.number_input("HbA1c Pré-TR (%)", 4.0, 12.0, 5.5, 0.1)

            # Indicateurs visuels
            markers = {'Glycémie': glycemie, 'HbA1c': hba1c}
            indicator_html = " &nbsp; ".join(
                f"<b style='color:#94a3b8'>{k}</b>{input_indicator(v, k)}"
                for k, v in markers.items()
            )
            st.markdown(
                f"<div style='margin-top:0.5rem;font-size:0.85rem;'>{indicator_html}</div>",
                unsafe_allow_html=True
            )

        with st.expander("👤 Donneur", expanded=False):
            age_donneur = st.slider("Âge Donneur", 18, 80, 40)

        st.markdown("")
        predict_btn = st.button("🔬 Évaluer le Risque PTDM", width='stretch')

    # ── Résultats ─────────────────────────────────────────────
    with col_result:
        section_label("🎯 Analyse & Résultat")

        if predict_btn:
            patient = {
                'age_receveur_TR': age,
                'sexe_receveur_M': sex_val,
                'obésité_pre_TR_receveur': obesite_val,
                'HTA_pre_TR_receveur': hta_val,
                'glycémie_pre_TR_R': glycemie,
                'HbA1c_pre_TR_R': hba1c,
                'durée_dialyse_année': duree_dialyse,
                'age_donneur': age_donneur
            }

            with st.spinner("Analyse du risque..."):
                prediction, prob_patho, patient_data = model.predict(patient)

            badge = confidence_badge(prob_patho)

            if prediction == 0:
                st.markdown(f"""
                <div class="result-normal">
                    <h2>✅ Risque Faible</h2>
                    <p>Faible probabilité de développer un diabète post-transplantation</p>
                    {badge}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-pathological">
                    <h2>⚠️ Risque Élevé (PTDM)</h2>
                    <p>Probabilité significative de développer un diabète post-transplantation</p>
                    {badge}
                </div>""", unsafe_allow_html=True)

            # Jauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_patho * 100,
                title={'text': "Probabilité de PTDM", 'font': {'size': 13, 'color': '#94a3b8'}},
                number={'suffix': '%', 'font': {'size': 30, 'color': '#f1f5f9'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#94a3b8', 'tickfont': {'color': '#94a3b8'}},
                    'bar': {'color': "#ef4444" if prob_patho > 0.5 else "#10b981", 'thickness': 0.25},
                    'bgcolor': 'rgba(255,255,255,0.03)',
                    'bordercolor': 'rgba(0,212,255,0.15)',
                    'steps': [
                        {'range': [0,  30], 'color': 'rgba(16,185,129,0.12)'},
                        {'range': [30, 60], 'color': 'rgba(245,158,11,0.10)'},
                        {'range': [60,100], 'color': 'rgba(239,68,68,0.12)'},
                    ],
                    'threshold': {'line': {'color': '#00d4ff', 'width': 2}, 'thickness': 0.75, 'value': 50}
                }
            ))
            fig_gauge.update_layout(height=240, margin=dict(t=50,b=5,l=20,r=20), **DARK_LAYOUT)
            st.plotly_chart(fig_gauge, width='stretch')

            # Radar — marqueurs cliniques (Normalisé)
            section_label("📡 Profil de Risque")
            radar_markers = ['HbA1c', 'Glycémie', 'Âge Rec', 'Durée Dialyse', 'Âge Donneur']
            radar_vals = [
                min(hba1c / 5.7, 2.0),
                min(glycemie / 1.1, 2.0),
                min(age / 50.0, 2.0),
                min(duree_dialyse / 5.0, 2.0),
                min(age_donneur / 50.0, 2.0)
            ]
            radar_norm = [1.0, 1.0, 1.0, 1.0, 1.0]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_norm + [radar_norm[0]],
                theta=radar_markers + [radar_markers[0]],
                fill='toself', name='Référence',
                line=dict(color='#10b981', width=1),
                fillcolor='rgba(16,185,129,0.08)'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_vals + [radar_vals[0]],
                theta=radar_markers + [radar_markers[0]],
                fill='toself', name='Patient',
                line=dict(color='#00d4ff', width=2),
                fillcolor='rgba(0,212,255,0.12)'
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0,2], color='#94a3b8', gridcolor='rgba(255,255,255,0.06)'),
                    angularaxis=dict(color='#94a3b8', gridcolor='rgba(255,255,255,0.06)')
                ),
                height=280, margin=dict(t=20,b=20,l=20,r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color='#94a3b8'), bgcolor='rgba(0,0,0,0)')
            )
            st.plotly_chart(fig_radar, width='stretch')

            # Export PDF-like rapport texte
            section_label("📄 Rapport")
            rapport = f"""RAPPORT DE PRÉDICTION PTDM — NovaClinic
==========================================
Date       : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Patient    : Âge {age} ans, {'Masculin' if sex_val==1 else 'Féminin'}

RÉSULTAT   : {'RISQUE ÉLEVÉ' if prediction==1 else 'FAIBLE RISQUE'}
Probabilité: {prob_patho:.1%}

PARAMÈTRES CLINIQUES
  Obésité        : {'Oui' if obesite_val==1 else 'Non'}
  HTA            : {'Oui' if hta_val==1 else 'Non'}
  Durée Dialyse  : {duree_dialyse} ans
  Glycémie       : {glycemie} g/L (norme < 1.1)
  HbA1c          : {hba1c} % (norme < 5.7)
  Âge Donneur    : {age_donneur} ans

AVERTISSEMENT : Outil d'aide à la décision uniquement.
"""
            st.download_button("📥 Télécharger le rapport", rapport,
                               file_name="rapport_ptdm.txt", mime="text/plain")

        else:
            st.markdown("""
            <div style='background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.15);
                        border-radius:12px;padding:2rem;text-align:center;color:#94a3b8;'>
                <div style='font-size:2.5rem;margin-bottom:0.8rem;'>🩸</div>
                <div style='font-weight:600;color:#f1f5f9;margin-bottom:0.4rem;'>Évaluation PTDM</div>
                <div style='font-size:0.85rem;'>Remplissez le dossier de transplantation et lancez l'évaluation</div>
            </div>""", unsafe_allow_html=True)
