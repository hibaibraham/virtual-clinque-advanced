import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from utils.core import (
	load_model, compute_engineered_features, save_prediction,
	confidence_badge, input_indicator, section_label, NORMAL_RANGES, DARK_LAYOUT
)


def render():
	try:
		model, preprocessor, config = load_model()
	except Exception as e:
		st.error(f"❌ Modèle introuvable : {e}")
		st.info("Exécutez `python train_and_save_model.py` pour générer le modèle.")
		return

	col_form, col_result = st.columns([3, 2], gap="large")

	# ── Formulaire ─────────────────────────────────────────────
	with col_form:
		section_label("📋 Dossier Patient")

		with st.expander("👤 Informations Générales", expanded=True):
			c1, c2 = st.columns(2)
			age     = c1.slider("Âge", 1, 100, 45)
			sex_str = c2.selectbox("Sexe", ["Féminin (F)", "Masculin (M)"])
			sex_val = 0 if sex_str.startswith("F") else 1

		with st.expander("🔬 Résultats de Laboratoire", expanded=True):
			c1, c2, c3 = st.columns(3)
			tsh = c1.number_input("TSH (mU/L)",  0.0, 600.0,  2.0,  0.1)
			t3  = c1.number_input("T3 (nmol/L)", 0.0,  15.0,  2.0,  0.1)
			tt4 = c2.number_input("TT4 (nmol/L)",0.0, 600.0,110.0,  1.0)
			t4u = c2.number_input("T4U",         0.0,   3.0,  1.0, 0.01)
			fti = c3.number_input("FTI",         0.0, 900.0,110.0,  1.0)

			# Indicateurs visuels en temps réel
			markers = {'TSH': tsh, 'T3': t3, 'TT4': tt4, 'T4U': t4u, 'FTI': fti}
			indicator_html = " &nbsp; ".join(
				f"<b style='color:#94a3b8'>{k}</b>{input_indicator(v, k)}"
				for k, v in markers.items()
			)
			st.markdown(
				f"<div style='margin-top:0.5rem;font-size:0.85rem;'>{indicator_html}</div>",
				unsafe_allow_html=True
			)

		with st.expander("🏥 Antécédents Médicaux", expanded=False):
			c1, c2 = st.columns(2)
			on_thyroxine       = int(c1.checkbox("Sous thyroxine"))
			query_on_thyroxine = int(c1.checkbox("Question thyroxine"))
			on_antithyroid     = int(c1.checkbox("Sous antithyroïdien"))
			sick               = int(c1.checkbox("Malade"))
			pregnant           = int(c1.checkbox("Enceinte"))
			thyroid_surgery    = int(c1.checkbox("Chirurgie thyroïdienne"))
			i131_treatment     = int(c1.checkbox("Traitement I131"))
			query_hypothyroid  = int(c2.checkbox("Question hypothyroïdie"))
			query_hyperthyroid = int(c2.checkbox("Question hyperthyroïdie"))
			lithium            = int(c2.checkbox("Sous lithium"))
			goitre             = int(c2.checkbox("Goitre"))
			tumor              = int(c2.checkbox("Tumeur"))
			hypopituitary      = int(c2.checkbox("Hypopituitarisme"))
			psych              = int(c2.checkbox("Trouble psychiatrique"))

		st.markdown("")
		predict_btn = st.button("🔬 Lancer l'Analyse Diagnostique", width='stretch')

	# ── Résultats ─────────────────────────────────────────────
	with col_result:
		section_label("🎯 Analyse & Résultat")

		if predict_btn:
			patient = {
				'age': age, 'sex': sex_val,
				'on_thyroxine': on_thyroxine, 'query_on_thyroxine': query_on_thyroxine,
				'on_antithyroid_medication': on_antithyroid, 'sick': sick,
				'pregnant': pregnant, 'thyroid_surgery': thyroid_surgery,
				'I131_treatment': i131_treatment, 'query_hypothyroid': query_hypothyroid,
				'query_hyperthyroid': query_hyperthyroid, 'lithium': lithium,
				'goitre': goitre, 'tumor': tumor, 'hypopituitary': hypopituitary,
				'psych': psych, 'TSH': tsh, 'T3': t3, 'TT4': tt4, 'T4U': t4u, 'FTI': fti
			}

			with st.spinner("Analyse en cours..."):
				patient       = compute_engineered_features(patient)
				all_features  = config['all_features']
				X_input       = pd.DataFrame([[patient.get(f, 0) for f in all_features]], columns=all_features)
				X_proc        = preprocessor.transform(X_input)
				prediction    = model.predict(X_proc)[0]
				proba         = model.predict_proba(X_proc)[0]
				prob_patho    = proba[1]

			save_prediction(patient, prediction, prob_patho)

			badge = confidence_badge(prob_patho)

			if prediction == 0:
				st.markdown(f"""
				<div class="result-normal">
					<h2>✅ Résultat Normal</h2>
					<p>Profil hormonal dans les normes cliniques</p>
					{badge}
				</div>""", unsafe_allow_html=True)
			else:
				st.markdown(f"""
				<div class="result-pathological">
					<h2>⚠️ Pathologie Détectée</h2>
					<p>Anomalie thyroïdienne potentielle — consultation recommandée</p>
					{badge}
				</div>""", unsafe_allow_html=True)

			# Jauge
			fig_gauge = go.Figure(go.Indicator(
				mode="gauge+number",
				value=prob_patho * 100,
				title={'text': "Probabilité de Pathologie", 'font': {'size': 13, 'color': '#94a3b8'}},
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

			# Radar — marqueurs biologiques vs normes
			section_label("📡 Radar Biologique")
			radar_markers = ['TSH', 'T3', 'TT4', 'FTI', 'T4U']
			radar_vals, radar_norm = [], []
			for m in radar_markers:
				lo, hi  = NORMAL_RANGES[m]
				val     = markers[m]
				mid     = (lo + hi) / 2
				radar_vals.append(round(min(val / hi, 2.0), 3))
				radar_norm.append(1.0)

			fig_radar = go.Figure()
			fig_radar.add_trace(go.Scatterpolar(
				r=radar_norm + [radar_norm[0]],
				theta=radar_markers + [radar_markers[0]],
				fill='toself', name='Zone normale',
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

			# Top features (XAI)
			section_label("🧠 IA Explicable")
			importances = config.get('feature_importances', {})
			if importances:
				imp_df = (pd.DataFrame({'Feature': list(importances.keys()),
										'Importance': list(importances.values())})
						  .sort_values('Importance', ascending=True).tail(8))
				fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
								 color='Importance',
								 color_continuous_scale=[[0,'#1e3a5f'],[0.5,'#0ea5e9'],[1,'#00d4ff']])
				fig_imp.update_layout(height=280, margin=dict(t=5,b=5,l=5,r=5),
									  coloraxis_showscale=False, **DARK_LAYOUT)
				st.plotly_chart(fig_imp, width='stretch')

			# Export PDF-like rapport texte
			section_label("📄 Rapport")
			rapport = f"""RAPPORT DE DIAGNOSTIC — MedAI Thyroid v3.0
==========================================
Date       : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Patient    : Âge {age} ans, {'Féminin' if sex_val==0 else 'Masculin'}

RÉSULTAT   : {'NORMAL' if prediction==0 else 'PATHOLOGIQUE'}
Probabilité: {prob_patho:.1%}

MARQUEURS BIOLOGIQUES
  TSH  : {tsh}  mU/L  (norme 0.4–4.0)  {'✓' if 0.4<=tsh<=4.0 else '✗'}
  T3   : {t3}  nmol/L (norme 1.2–3.1)  {'✓' if 1.2<=t3<=3.1 else '✗'}
  TT4  : {tt4} nmol/L (norme 70–180)   {'✓' if 70<=tt4<=180 else '✗'}
  FTI  : {fti}        (norme 70–180)   {'✓' if 70<=fti<=180 else '✗'}
  T4U  : {t4u}        (norme 0.7–1.3)  {'✓' if 0.7<=t4u<=1.3 else '✗'}

AVERTISSEMENT : Outil d'aide à la décision uniquement.
"""
			st.download_button("📥 Télécharger le rapport", rapport,
							   file_name="rapport_diagnostic.txt", mime="text/plain")

		else:
			st.markdown("""
			<div style='background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.15);
						border-radius:12px;padding:2rem;text-align:center;color:#94a3b8;'>
				<div style='font-size:2.5rem;margin-bottom:0.8rem;'>🔬</div>
				<div style='font-weight:600;color:#f1f5f9;margin-bottom:0.4rem;'>En attente d'analyse</div>
				<div style='font-size:0.85rem;'>Remplissez le formulaire et lancez le diagnostic</div>
			</div>""", unsafe_allow_html=True)

			st.markdown("<br>", unsafe_allow_html=True)
			section_label("📏 Valeurs de Référence")
			ref = pd.DataFrame({
				'Marqueur': ['TSH', 'T3', 'TT4', 'FTI', 'T4U'],
				'Min':  [0.4, 1.2,  70,  70, 0.7],
				'Max':  [4.0, 3.1, 180, 180, 1.3],
				'Unité':['mU/L','nmol/L','nmol/L','index','ratio']
			})
			st.dataframe(ref, width='stretch', hide_index=True)
