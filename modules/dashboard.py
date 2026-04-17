import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.core import load_model, load_dataset, extract_label, section_label, DARK_LAYOUT


def render():
	df_raw = load_dataset()

	try:
		_, _, config = load_model()
		model_loaded = True
	except Exception:
		config = {}
		model_loaded = False

	# KPI row
	c1, c2, c3, c4 = st.columns(4)
	c1.metric("📊 Échantillons", f"{len(df_raw):,}")
	c2.metric("📐 Features",     f"{df_raw.shape[1]}")
	if model_loaded:
		c3.metric("🎯 Accuracy", f"{config.get('test_accuracy', 0):.1%}")
		c4.metric("📈 F1 Score", f"{config.get('test_f1', 0):.1%}")

	st.markdown("---")

	# Filtres dashboard
	df_raw_copy = df_raw.copy()
	df_raw_copy.replace('?', np.nan, inplace=True)
	df_raw_copy['age'] = pd.to_numeric(df_raw_copy['age'], errors='coerce')
	df_raw_copy['class_label'] = df_raw_copy['class'].apply(extract_label)

	with st.expander("🔍 Filtres", expanded=False):
		fc1, fc2 = st.columns(2)
		age_range = fc1.slider("Plage d'âge", 0, 100, (0, 100))
		sex_filter = fc2.multiselect("Sexe", ["M", "F", "?"], default=["M", "F", "?"])
		df_filtered = df_raw_copy[
			(df_raw_copy['age'].fillna(50).between(*age_range)) &
			(df_raw_copy['sex'].isin(sex_filter))
		]

	tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribution", "🔗 Corrélations", "📊 Démographie", "🤖 Modèle"])

	# ── Tab 1: Distribution ─────────────────────────────────────────────
	with tab1:
		c1, c2 = st.columns(2)

		class_counts = df_filtered['class_label'].value_counts().reset_index()
		class_counts.columns = ['Classe', 'Nombre']
		fig_dist = px.bar(class_counts, x='Classe', y='Nombre', color='Nombre',
						  color_continuous_scale=[[0,'#1e3a5f'],[0.5,'#0ea5e9'],[1,'#00d4ff']])
		fig_dist.update_layout(height=380, coloraxis_showscale=False, **DARK_LAYOUT)
		c1.plotly_chart(fig_dist, width='stretch')

		binary_counts = pd.DataFrame({
			'Statut': ['Normal', 'Pathologique'],
			'Nombre': [int((df_filtered['class_label'] == '-').sum()),
					   int((df_filtered['class_label'] != '-').sum())]
		})
		fig_pie = px.pie(binary_counts, values='Nombre', names='Statut', hole=0.55,
						 color='Statut',
						 color_discrete_map={'Normal': '#10b981', 'Pathologique': '#ef4444'})
		fig_pie.update_layout(height=380, **DARK_LAYOUT)
		fig_pie.update_traces(textfont_color='white')
		c2.plotly_chart(fig_pie, width='stretch')

	# ── Tab 2: Corrélations ─────────────────────────────────────────────
	with tab2:
		num_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
		df_corr  = df_filtered[num_cols].apply(pd.to_numeric, errors='coerce')
		corr_matrix = df_corr.corr()

		fig_corr = px.imshow(corr_matrix, text_auto='.2f',
							 color_continuous_scale='RdBu_r', aspect='auto')
		fig_corr.update_layout(height=450, **DARK_LAYOUT)
		st.plotly_chart(fig_corr, width='stretch')

		# Scatter interactif
		section_label("📉 Scatter — Choisir les axes")
		sc1, sc2, sc3 = st.columns(3)
		x_ax  = sc1.selectbox("Axe X",     num_cols, index=0)
		y_ax  = sc2.selectbox("Axe Y",     num_cols, index=2)
		color = sc3.selectbox("Couleur",   ['class_label', 'sex'], index=0)

		df_scatter = df_filtered.dropna(subset=[x_ax, y_ax])
		fig_scatter = px.scatter(df_scatter, x=x_ax, y=y_ax, color=color,
								 opacity=0.6, marginal_x='histogram', marginal_y='box',
								 color_discrete_sequence=['#00d4ff','#ef4444','#10b981','#f59e0b','#7c3aed'])
		fig_scatter.update_layout(height=480, **DARK_LAYOUT)
		st.plotly_chart(fig_scatter, width='stretch')

	# ── Tab 3: Démographie ─────────────────────────────────────────────
	with tab3:
		c1, c2 = st.columns(2)

		fig_age = px.histogram(df_filtered.dropna(subset=['age']), x='age', nbins=30,
							   color_discrete_sequence=['#0ea5e9'])
		fig_age.update_layout(height=350, **DARK_LAYOUT)
		c1.plotly_chart(fig_age, width='stretch')

		# Box plots marqueurs par statut
		df_filtered['statut'] = df_filtered['class_label'].apply(lambda x: 'Normal' if x == '-' else 'Pathologique')
		marker_box = c2.selectbox("Marqueur", ['TSH', 'T3', 'TT4', 'FTI'], key='box_marker')
		df_filtered[marker_box] = pd.to_numeric(df_filtered[marker_box], errors='coerce')
		fig_box = px.box(df_filtered.dropna(subset=[marker_box]), x='statut', y=marker_box,
						 color='statut',
						 color_discrete_map={'Normal': '#10b981', 'Pathologique': '#ef4444'})
		fig_box.update_layout(height=350, showlegend=False, **DARK_LAYOUT)
		c2.plotly_chart(fig_box, width='stretch')

		# Tendance âge vs marqueurs
		df_trend = df_filtered.dropna(subset=['age', 'TSH'])
		df_trend['age_bin'] = pd.cut(df_trend['age'], bins=10)
		df_trend['age_mid'] = df_trend['age_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan)
		df_agg = df_trend.groupby('age_mid')[['TSH','T3','TT4']].median().reset_index()
		fig_trend = px.line(df_agg, x='age_mid', y=['TSH','T3','TT4'],
							color_discrete_sequence=['#00d4ff','#10b981','#f59e0b'],
							title="Médiane des marqueurs par tranche d'âge")
		fig_trend.update_layout(height=350, **DARK_LAYOUT)
		st.plotly_chart(fig_trend, width='stretch')

	# ── Tab 4: Modèle ─────────────────────────────────────────────
	with tab4:
		if not model_loaded:
			st.warning("Modèle non chargé.")
			return

		c1, c2 = st.columns(2)
		metrics_df = pd.DataFrame({
			'Métrique': ['Accuracy', 'F1 Score', 'F1 CV', 'Train samples', 'Test samples'],
			'Valeur':   [f"{config.get('test_accuracy',0):.4f}",
						 f"{config.get('test_f1',0):.4f}",
						 f"{config.get('cv_f1',0):.4f}",
						 str(config.get('train_samples','N/A')),
						 str(config.get('test_samples','N/A'))]
		})
		c1.dataframe(metrics_df, width='stretch', hide_index=True)

		params = config.get('best_params', {})
		c2.dataframe(pd.DataFrame({'Paramètre': list(params.keys()),
								   'Valeur': list(params.values())}),
					 width='stretch', hide_index=True)

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
