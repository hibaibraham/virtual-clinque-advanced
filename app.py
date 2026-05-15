"""
MedAI Thyroid v3.0 — Diagnostic Thyroïdien Intelligent
Lancer : streamlit run app.py
"""
import streamlit as st
from utils.core import inject_css, inject_bg, page_header
from utils.auth import require_auth, get_user_role

st.set_page_config(
    page_title="MedAI — Diagnostic Clinique Intelligent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS + animations
inject_css()
inject_bg()

# ── Authentification 2FA (bloque l'app si non connecté) ──────────────────────
require_auth()

# Récupérer le rôle de l'utilisateur
username = st.session_state.get("auth_username", "")
user_role = get_user_role(username)

# Vérifier que l'utilisateur a un rôle autorisé (médecin ou secrétaire uniquement)
if user_role not in ["medecin", "secretaire"]:
    st.error("🚫 Accès Refusé")
    st.warning("Cette plateforme est réservée au personnel médical et administratif.")
    st.info("Seuls les médecins et secrétaires peuvent accéder à cette application.")
    if st.button("🔓 Déconnexion"):
        st.session_state.authenticated = False
        st.session_state.auth_step = "select_role"
        st.session_state.page = None
        st.session_state.selected_role = ""
        st.rerun()
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">🧬</span>
        <h2>MedAI Thyroid</h2>
        <p>Système de Diagnostic Intelligent</p>
    </div>""", unsafe_allow_html=True)

    # Afficher le rôle de l'utilisateur
    role_icons = {"medecin": "👨‍⚕️", "secretaire": "📋"}
    role_names = {"medecin": "Médecin", "secretaire": "Secrétaire"}
    st.markdown(f"""
    <div style='text-align:center;padding:0.8rem;background:rgba(0,212,255,0.08);
                border-radius:10px;margin-bottom:1rem;border:1px solid rgba(0,212,255,0.15);'>
        <div style='font-size:2rem;margin-bottom:0.3rem;'>{role_icons.get(user_role, "👤")}</div>
        <div style='color:#00d4ff;font-weight:600;font-size:0.9rem;'>{role_names.get(user_role, "Utilisateur")}</div>
        <div style='color:#94a3b8;font-size:0.75rem;margin-top:0.2rem;'>{username}</div>
    </div>""", unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state.page = None

    # Pages selon le rôle
    if user_role == "medecin":
        pages = ["👨‍⚕️ Interface Médecin", "🦋 Analyse Thyroïde", "🧠 Tumeur Cérébrale", "📊 Dashboard Thyroïde", "📊 Dashboard Cancer", "📜 Historique", "ℹ️ À Propos"]
    elif user_role == "secretaire":
        pages = ["📋 Interface Secrétaire", "👥 Gestion Patients", "📅 Rendez-vous", "📊 Statistiques", "ℹ️ À Propos"]
    else:
        pages = ["ℹ️ À Propos"]
    
    # Initialiser la page par défaut
    if st.session_state.page is None or st.session_state.page not in pages:
        st.session_state.page = pages[0]
    
    page = st.radio("nav", pages,
                   index=pages.index(st.session_state.page),
                   label_visibility="collapsed")
    st.session_state.page = page

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding:1rem;background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.12);border-radius:10px;font-size:0.78rem;color:#94a3b8;'>
        <div style='margin-bottom:0.4rem;'><span class='status-dot'></span> Système opérationnel</div>
        <div style='margin-bottom:0.3rem;'>🤖 Thyroïde : Random Forest</div>
        <div style='margin-bottom:0.3rem;'>🧠 IRM : EfficientNet-B0</div>
        <div style='margin-bottom:0.3rem;'>📦 Dataset : Thyroid + Brain MRI</div>
        <div>🔖 Version : 3.1</div>
    </div>""", unsafe_allow_html=True)

    # Déconnexion
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔓 Déconnexion", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.auth_step = "select_role"
        st.session_state.page = None
        st.session_state.selected_role = ""
        st.rerun()

# ── Router ────────────────────────────────────────────────────────────────────

# Pages Médecin
if page == "👨‍⚕️ Interface Médecin":
    from modules.medecin import render
    render()

# Pages Secrétaire
elif page == "📋 Interface Secrétaire":
    from modules.secretaire import render
    render()

elif page == "👥 Gestion Patients":
    from modules.secretaire import render
    render()

elif page == "📅 Rendez-vous":
    page_header("📅 Gestion des Rendez-vous",
                "Calendrier et Planification",
                "Organisation des consultations")
    st.info("🚧 Module en développement")

elif page == "📊 Statistiques":
    from modules.secretaire import render
    render()

# Pages communes
elif page == "🦋 Analyse Thyroïde":
    page_header("🦋 Analyse Thyroïde",
                "Diagnostic Thyroïdien par IA",
                "Renseignez les paramètres biologiques pour une analyse prédictive en temps réel")
    from modules.prediction import render
    render()

elif page == "🧠 Tumeur Cérébrale":
    page_header("🧠 Module IRM",
                "Diagnostic Tumeur Cérébrale",
                "Analysez une image IRM cérébrale — classification en 4 classes par EfficientNet-B0")
    from modules.brain_tumor import render
    render()

elif page == "📊 Dashboard Thyroïde":
    page_header("📊 Analytics Thyroïde",
                "Tableau de Bord Thyroïde",
                "Statistiques interactives du dataset et performance du modèle Random Forest")
    from modules.dashboard import render
    render()

elif page == "📊 Dashboard Cancer":
    page_header("📊 Analytics Cancer Cérébral",
                "Tableau de Bord Cancer Cérébral",
                "Statistiques et performances du modèle EfficientNet-B0")
    from modules.brain_tumor_dashboard import render
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
