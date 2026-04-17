"""
MedAI Thyroid v3.0 — Diagnostic Thyroïdien Intelligent
Lancer : streamlit run app.py
"""
import streamlit as st
from utils.core import inject_css, inject_bg, page_header
from utils.auth import require_auth

st.set_page_config(
    page_title="MedAI — Diagnostic Thyroïdien",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS + animations
inject_css()
inject_bg()

# ── Authentification 2FA (bloque l'app si non connecté) ──────────────────────
require_auth()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">🧬</span>
        <h2>MedAI Thyroid</h2>
        <p>Système de Diagnostic Intelligent</p>
    </div>""", unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state.page = "🩺 Prédiction"

    pages = ["🩺 Prédiction", "📊 Tableau de Bord", "📜 Historique", "ℹ️ À Propos"]
    page  = st.radio("nav", pages,
                     index=pages.index(st.session_state.page),
                     label_visibility="collapsed")
    st.session_state.page = page

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding:1rem;background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.12);border-radius:10px;font-size:0.78rem;color:#94a3b8;'>
        <div style='margin-bottom:0.4rem;'><span class='status-dot'></span> Système opérationnel</div>
        <div style='margin-bottom:0.3rem;'>🤖 Modèle : Random Forest</div>
        <div style='margin-bottom:0.3rem;'>📦 Dataset : Thyroid Disease</div>
        <div>🔖 Version : 3.0</div>
    </div>""", unsafe_allow_html=True)

    # Déconnexion
    st.markdown("<br>", unsafe_allow_html=True)
    user = st.session_state.get("auth_username", "")
    st.markdown(f"<div style='text-align:center;font-size:0.78rem;color:#94a3b8;margin-bottom:0.5rem;'>👤 Connecté : <b style='color:#00d4ff'>{user}</b></div>", unsafe_allow_html=True)
    if st.button("🔓 Déconnexion", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.auth_step = "login"
        st.rerun()

# ── Router ────────────────────────────────────────────────────────────────────
if page == "🩺 Prédiction":
    page_header("🩺 Module Clinique",
                "Diagnostic Thyroïdien par IA",
                "Renseignez les paramètres biologiques pour une analyse prédictive en temps réel")
    from modules.prediction import render
    render()

elif page == "📊 Tableau de Bord":
    page_header("📊 Analytics",
                "Tableau de Bord Clinique",
                "Statistiques interactives du dataset et performance du modèle")
    from modules.dashboard import render
    render()

elif page == "📜 Historique":
    page_header("📜 Registre Médical",
                "Historique des Prédictions",
                "Consultez, filtrez et exportez l'ensemble des analyses effectuées")
    from modules.historique import render
    render()

elif page == "ℹ️ À Propos":
    page_header("ℹ️ Documentation",
                "À Propos de MedAI Thyroid",
                "Architecture, pipeline ML et informations cliniques de référence")
    from modules.apropos import render
    render()
