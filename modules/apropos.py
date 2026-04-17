import streamlit as st
from utils.core import footer, section_label


def render():
		c1, c2 = st.columns(2)

		with c1:
				section_label("🧬 Architecture ML")
				st.markdown("""
				<div style='background:#111827;border:1px solid rgba(0,212,255,0.15);border-radius:12px;padding:1.5rem;line-height:1.8;color:#cbd5e1;font-size:0.9rem;'>
				Le système utilise un <strong style='color:#00d4ff'>Random Forest optimisé</strong> entraîné
				sur le dataset UCI Thyroid Disease (>7000 patients).<br><br>
				<strong style='color:#94a3b8'>Pipeline :</strong><br>
				1. Nettoyage &amp; imputation des valeurs manquantes<br>
				2. Feature engineering (seuils cliniques, ratios)<br>
				3. StandardScaler + SimpleImputer<br>
				4. SMOTE (rééquilibrage 26% → 50%)<br>
				5. RandomizedSearchCV (30 combinaisons)<br><br>
				<strong style='color:#94a3b8'>Modèles évalués :</strong><br>
				Logistic Regression (~89% F1)<br>
				<span style='color:#10b981'>✓ Random Forest (~94% F1) — Sélectionné</span><br>
				XGBoost (~94% F1)
				</div>""", unsafe_allow_html=True)

				st.markdown("<br>", unsafe_allow_html=True)
				section_label("🛠️ Stack Technique")
				st.markdown("""
				<div style='background:#111827;border:1px solid rgba(0,212,255,0.15);border-radius:12px;padding:1.5rem;color:#cbd5e1;font-size:0.9rem;'>
				<b style='color:#00d4ff'>Frontend</b> : Streamlit + CSS custom<br>
				<b style='color:#00d4ff'>ML</b> : scikit-learn, imbalanced-learn (SMOTE)<br>
				<b style='color:#00d4ff'>Data</b> : pandas, numpy<br>
				<b style='color:#00d4ff'>Visualisation</b> : Plotly Express &amp; Graph Objects<br>
				<b style='color:#00d4ff'>Export</b> : rapport texte téléchargeable
				</div>""", unsafe_allow_html=True)

		with c2:
				section_label("📏 Valeurs de Référence")
				st.markdown("""
				<div style='background:#111827;border:1px solid rgba(0,212,255,0.15);border-radius:12px;padding:1.5rem;'>
				<table style='width:100%;border-collapse:collapse;color:#cbd5e1;font-size:0.88rem;'>
					<tr style='border-bottom:1px solid rgba(0,212,255,0.15);color:#94a3b8;'>
						<th style='padding:0.5rem;text-align:left'>Marqueur</th>
						<th style='padding:0.5rem;text-align:center'>Min</th>
						<th style='padding:0.5rem;text-align:center'>Max</th>
						<th style='padding:0.5rem;text-align:left'>Unité</th>
					</tr>
					<tr><td style='padding:0.5rem'>TSH</td><td style='text-align:center'>0.4</td><td style='text-align:center'>4.0</td><td>mU/L</td></tr>
					<tr style='background:rgba(0,212,255,0.03)'><td style='padding:0.5rem'>T3</td><td style='text-align:center'>1.2</td><td style='text-align:center'>3.1</td><td>nmol/L</td></tr>
					<tr><td style='padding:0.5rem'>TT4</td><td style='text-align:center'>70</td><td style='text-align:center'>180</td><td>nmol/L</td></tr>
					<tr style='background:rgba(0,212,255,0.03)'><td style='padding:0.5rem'>FTI</td><td style='text-align:center'>70</td><td style='text-align:center'>180</td><td>index</td></tr>
					<tr><td style='padding:0.5rem'>T4U</td><td style='text-align:center'>0.7</td><td style='text-align:center'>1.3</td><td>ratio</td></tr>
				</table>
				</div>""", unsafe_allow_html=True)

				st.markdown("<br>", unsafe_allow_html=True)
				section_label("⚠️ Avertissement")
				st.markdown("""
				<div style='background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.3);border-radius:12px;padding:1.5rem;color:#fcd34d;font-size:0.88rem;line-height:1.7;'>
				Ce système est un <strong>outil d'aide à la décision</strong> et ne remplace
				en aucun cas un diagnostic médical professionnel. Les résultats doivent
				toujours être interprétés par un professionnel de santé qualifié.
				</div>""", unsafe_allow_html=True)

		st.markdown("<br>", unsafe_allow_html=True)
		footer()
