"""
Dashboard Analyse Dentaire - Statistiques et Visualisations
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta

# Chemin du fichier CSV
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tooth_predictions.csv')

# Configuration des classes
CLASS_INFO = {
    'Cavity': {'icon': '🦷', 'name': 'Carie', 'color': '#ef4444'},
    'Fillings': {'icon': '🔧', 'name': 'Plombage', 'color': '#3b82f6'},
    'Impacted Tooth': {'icon': '⚠️', 'name': 'Dent Incluse', 'color': '#f59e0b'},
    'Implant': {'icon': '🦾', 'name': 'Implant', 'color': '#10b981'},
    'Normal': {'icon': '✅', 'name': 'Saine', 'color': '#22c55e'}
}

@st.cache_data(ttl=30)
def load_predictions():
    """Charge les prédictions depuis le CSV."""
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        return df
    return pd.DataFrame()

def render():
    """Interface principale du dashboard dentaire."""
    
    # Charger les données
    df = load_predictions()
    
    if df.empty:
        st.info("""
        ### 📊 Aucune Donnée Disponible
        
        Le dashboard affichera les statistiques une fois que des analyses dentaires auront été effectuées.
        
        **Pour commencer:**
        1. Accédez à la page "🦷 Analyse Dentaire"
        2. Téléchargez une radiographie dentaire
        3. Effectuez une analyse
        4. Revenez ici pour voir les statistiques
        """)
        return
    
    # En-tête avec statistiques globales
    st.markdown("""
    <div style='text-align:center;padding:1.5rem;background:linear-gradient(135deg, rgba(0,212,255,0.1), rgba(124,58,237,0.1));
                border-radius:12px;border:1px solid rgba(0,212,255,0.2);margin-bottom:2rem;'>
        <h2 style='color:#f1f5f9;margin:0;'>📊 Dashboard Analyse Dentaire</h2>
        <p style='color:#94a3b8;margin:0.5rem 0 0;'>Statistiques et Visualisations des Analyses</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Métriques principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_analyses = len(df)
    unique_patients = df['patient_id'].nunique() if 'patient_id' in df.columns else 0
    today_analyses = len(df[df['date'] == datetime.now().date()])
    
    # Calculer la distribution
    most_common = df['prediction'].value_counts().index[0] if len(df) > 0 else "N/A"
    most_common_info = CLASS_INFO.get(most_common, {'icon': '❓', 'name': 'N/A'})
    
    with col1:
        st.metric("📊 Total Analyses", f"{total_analyses:,}")
    
    with col2:
        st.metric("👥 Patients", f"{unique_patients:,}")
    
    with col3:
        st.metric("📅 Aujourd'hui", f"{today_analyses}")
    
    with col4:
        avg_confidence = df['confidence'].str.rstrip('%').astype(float).mean()
        st.metric("🎯 Confiance Moy.", f"{avg_confidence:.1f}%")
    
    with col5:
        st.metric(f"{most_common_info['icon']} Plus Fréquent", most_common_info['name'])
    
    st.markdown("---")
    
    # Filtres
    st.markdown("### 🔍 Filtres")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filtre par période
        period = st.selectbox(
            "Période",
            ["Tout", "Aujourd'hui", "7 derniers jours", "30 derniers jours", "Personnalisé"]
        )
        
        if period == "Aujourd'hui":
            df_filtered = df[df['date'] == datetime.now().date()]
        elif period == "7 derniers jours":
            df_filtered = df[df['date'] >= (datetime.now().date() - timedelta(days=7))]
        elif period == "30 derniers jours":
            df_filtered = df[df['date'] >= (datetime.now().date() - timedelta(days=30))]
        elif period == "Personnalisé":
            col_start, col_end = st.columns(2)
            with col_start:
                start_date = st.date_input("Date début", value=df['date'].min())
            with col_end:
                end_date = st.date_input("Date fin", value=df['date'].max())
            df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        else:
            df_filtered = df
    
    with col2:
        # Filtre par classe
        selected_classes = st.multiselect(
            "Classes",
            options=list(CLASS_INFO.keys()),
            default=list(CLASS_INFO.keys())
        )
        if selected_classes:
            df_filtered = df_filtered[df_filtered['prediction'].isin(selected_classes)]
    
    with col3:
        # Filtre par médecin
        if 'medecin' in df.columns:
            medecins = ['Tous'] + list(df['medecin'].unique())
            selected_medecin = st.selectbox("Médecin", medecins)
            if selected_medecin != 'Tous':
                df_filtered = df_filtered[df_filtered['medecin'] == selected_medecin]
    
    st.markdown("---")
    
    # Graphiques
    if len(df_filtered) == 0:
        st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
        return
    
    # Ligne 1: Distribution et Évolution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Distribution des Diagnostics")
        
        # Compter les prédictions
        pred_counts = df_filtered['prediction'].value_counts()
        
        # Créer le graphique en donut
        colors = [CLASS_INFO[pred]['color'] for pred in pred_counts.index]
        labels = [f"{CLASS_INFO[pred]['icon']} {CLASS_INFO[pred]['name']}" 
                 for pred in pred_counts.index]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=pred_counts.values,
            hole=0.4,
            marker=dict(colors=colors),
            textinfo='label+percent',
            textposition='outside',
            hovertemplate='<b>%{label}</b><br>Nombre: %{value}<br>Pourcentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            showlegend=True,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f1f5f9', size=12),
            margin=dict(t=20, b=20, l=20, r=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Évolution Temporelle")
        
        # Grouper par date
        daily_counts = df_filtered.groupby(['date', 'prediction']).size().reset_index(name='count')
        
        fig = px.line(
            daily_counts,
            x='date',
            y='count',
            color='prediction',
            color_discrete_map={pred: CLASS_INFO[pred]['color'] for pred in CLASS_INFO.keys()},
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Nombre d'Analyses",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f1f5f9'),
            legend=dict(title="Diagnostic", orientation="v", x=1.02, y=1),
            hovermode='x unified',
            margin=dict(t=20, b=20, l=20, r=20)
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Ligne 2: Confiance et Distribution horaire
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Distribution de la Confiance")
        
        # Extraire les valeurs de confiance
        df_filtered['confidence_value'] = df_filtered['confidence'].str.rstrip('%').astype(float)
        
        fig = go.Figure()
        
        for pred in df_filtered['prediction'].unique():
            pred_data = df_filtered[df_filtered['prediction'] == pred]['confidence_value']
            fig.add_trace(go.Box(
                y=pred_data,
                name=f"{CLASS_INFO[pred]['icon']} {CLASS_INFO[pred]['name']}",
                marker_color=CLASS_INFO[pred]['color'],
                boxmean='sd'
            ))
        
        fig.update_layout(
            yaxis_title="Confiance (%)",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f1f5f9'),
            showlegend=True,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### ⏰ Distribution Horaire")
        
        # Grouper par heure
        hourly_counts = df_filtered.groupby('hour').size().reset_index(name='count')
        
        fig = go.Figure(data=[
            go.Bar(
                x=hourly_counts['hour'],
                y=hourly_counts['count'],
                marker=dict(
                    color=hourly_counts['count'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Analyses")
                ),
                hovertemplate='<b>Heure: %{x}h</b><br>Analyses: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            xaxis_title="Heure de la Journée",
            yaxis_title="Nombre d'Analyses",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f1f5f9'),
            margin=dict(t=20, b=20, l=20, r=20)
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', dtick=1)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Statistiques détaillées par classe
    st.markdown("### 📋 Statistiques Détaillées par Classe")
    
    stats_data = []
    for pred in CLASS_INFO.keys():
        pred_df = df_filtered[df_filtered['prediction'] == pred]
        if len(pred_df) > 0:
            stats_data.append({
                'Classe': f"{CLASS_INFO[pred]['icon']} {CLASS_INFO[pred]['name']}",
                'Nombre': len(pred_df),
                'Pourcentage': f"{len(pred_df)/len(df_filtered)*100:.1f}%",
                'Confiance Moy.': f"{pred_df['confidence_value'].mean():.1f}%",
                'Confiance Min.': f"{pred_df['confidence_value'].min():.1f}%",
                'Confiance Max.': f"{pred_df['confidence_value'].max():.1f}%"
            })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Afficher le tableau avec style
    st.dataframe(
        stats_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Classe': st.column_config.TextColumn('Classe', width='medium'),
            'Nombre': st.column_config.NumberColumn('Nombre', width='small'),
            'Pourcentage': st.column_config.TextColumn('Pourcentage', width='small'),
            'Confiance Moy.': st.column_config.TextColumn('Confiance Moy.', width='small'),
            'Confiance Min.': st.column_config.TextColumn('Confiance Min.', width='small'),
            'Confiance Max.': st.column_config.TextColumn('Confiance Max.', width='small')
        }
    )
    
    st.markdown("---")
    
    # Tableau des dernières analyses
    st.markdown("### 📜 Dernières Analyses")
    
    # Préparer les données pour l'affichage
    recent_df = df_filtered.sort_values('timestamp', ascending=False).head(10).copy()
    recent_df['Diagnostic'] = recent_df['prediction'].apply(
        lambda x: f"{CLASS_INFO[x]['icon']} {CLASS_INFO[x]['name']}"
    )
    recent_df['Date'] = recent_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    display_df = recent_df[['Date', 'patient_id', 'patient_name', 'Diagnostic', 'confidence', 'medecin']]
    display_df.columns = ['Date', 'ID Patient', 'Nom Patient', 'Diagnostic', 'Confiance', 'Médecin']
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Date': st.column_config.TextColumn('Date', width='medium'),
            'ID Patient': st.column_config.TextColumn('ID Patient', width='medium'),
            'Nom Patient': st.column_config.TextColumn('Nom Patient', width='medium'),
            'Diagnostic': st.column_config.TextColumn('Diagnostic', width='medium'),
            'Confiance': st.column_config.TextColumn('Confiance', width='small'),
            'Médecin': st.column_config.TextColumn('Médecin', width='medium')
        }
    )
    
    # Export des données
    st.markdown("---")
    st.markdown("### 💾 Export des Données")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        # Export CSV
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name=f"tooth_analyses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Export Excel
        try:
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_filtered.to_excel(writer, index=False, sheet_name='Analyses')
            excel_data = output.getvalue()
            
            st.download_button(
                label="📥 Télécharger Excel",
                data=excel_data,
                file_name=f"tooth_analyses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except ImportError:
            st.info("📦 Installez openpyxl pour l'export Excel")
    
    with col3:
        st.info(f"📊 {len(df_filtered)} analyses dans la sélection actuelle")

if __name__ == "__main__":
    render()
