"""
NovaClinic v4.0 — Plateforme de Diagnostic Médical Intelligent
Lancer : streamlit run app.py
"""
import streamlit as st
from utils.core import inject_css, inject_bg, page_header
from utils.auth import require_auth, get_user_role

st.set_page_config(
    page_title="NovaClinic",
    page_icon="🏥",
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
        <span class="logo-icon">🏥</span>
        <h2>NovaClinic</h2>
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
        pages = ["👨‍⚕️ Interface Médecin", "🦋 Analyse Thyroïde", "🧠 Tumeur Cérébrale", "🩸 Analyse PTDM", "📊 Dashboard Thyroïde", "📊 Dashboard Cancer", "📊 Dashboard PTDM", "📜 Historique", "ℹ️ À Propos"]
    elif user_role == "secretaire":
        pages = ["📋 Accueil", "➕ Nouveau Patient", "👥 Liste Patients", "📅 Rendez-vous", "📊 Statistiques"]
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
        <div style='margin-bottom:0.3rem;'>🩸 PTDM : SVM / RF</div>
        <div style='margin-bottom:0.3rem;'>📦 Dataset : Thyroid + Brain + PTDM</div>
        <div>🔖 Version : 4.0 - NovaClinic</div>
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
elif page == "📋 Accueil":
    # Barre de recherche en haut
    st.markdown("""
    <style>
    .search-bar {
        background: rgba(236,72,153,0.05);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(236,72,153,0.2);
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="search-bar">', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    with col1:
        search_term = st.text_input("🔍 Recherche rapide", 
                                   placeholder="Rechercher un patient par nom, prénom, téléphone, ID...",
                                   key="quick_search",
                                   label_visibility="collapsed")
    with col2:
        search_btn = st.button("Rechercher", use_container_width=True, key="btn_quick_search")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Afficher les résultats de recherche si recherche effectuée
    if search_btn and search_term:
        from utils.patients import search_patients
        results = search_patients(search_term)
        if results:
            st.success(f"📊 {len(results)} patient(s) trouvé(s)")
            
            for patient in results:
                status_icon = {
                    "en_attente": "⏳",
                    "en_cours": "🔄",
                    "complete": "✅"
                }.get(patient.get('status', ''), "❓")
                
                with st.expander(f"{status_icon} {patient.get('prenom', '')} {patient.get('nom', '')} - {patient.get('patient_id', '')}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Informations Personnelles:**")
                        st.text(f"ID: {patient.get('patient_id', '')}")
                        st.text(f"Nom: {patient.get('nom', '')}")
                        st.text(f"Prénom: {patient.get('prenom', '')}")
                        st.text(f"Âge: {patient.get('age', '')}")
                        st.text(f"Téléphone: {patient.get('telephone', '')}")
                    
                    with col2:
                        st.markdown("**Informations Médicales:**")
                        st.text(f"Statut: {patient.get('status', '').replace('_', ' ').title()}")
                        st.text(f"Motif: {patient.get('motif_consultation', '')}")
                        st.text(f"Créé le: {patient.get('created_at', '').split('T')[0] if patient.get('created_at') else ''}")
            
            st.markdown("---")
        else:
            st.info("Aucun patient trouvé avec ces critères.")
            st.markdown("---")
    
    # Contenu de la page d'accueil
    st.markdown("""
    <div style='text-align:center;padding:2rem;background:rgba(236,72,153,0.05);
                border-radius:12px;border:1px solid rgba(236,72,153,0.2);margin-bottom:2rem;'>
        <div style='font-size:3rem;margin-bottom:0.5rem;'>📋</div>
        <h1 style='color:#f1f5f9;margin-bottom:0.5rem;'>Espace Secrétaire</h1>
        <p style='color:#94a3b8;font-size:1.1rem;'>
        Bienvenue {username} - Gestion Administrative
        </p>
    </div>
    """.format(username=username), unsafe_allow_html=True)
    
    from utils.patients import get_all_patients
    from datetime import datetime
    
    all_patients = get_all_patients()
    today = datetime.now().date()
    today_patients = len([p for p in all_patients if p.get('created_at', '').startswith(str(today))])
    en_attente = len([p for p in all_patients if p.get('status') == 'en_attente'])
    
    # Statistiques rapides
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Total Patients", len(all_patients))
    with col2:
        st.metric("📅 Aujourd'hui", today_patients)
    with col3:
        st.metric("⏳ En Attente", en_attente)
    with col4:
        complets = len([p for p in all_patients if p.get('status') == 'complete'])
        st.metric("✅ Complets", complets)
    
    st.markdown("---")
    
    # Actions rapides
    st.subheader("🚀 Actions Rapides")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➕ Nouveau Patient", use_container_width=True):
            st.session_state.page = "➕ Nouveau Patient"
            st.rerun()
    
    with col2:
        if st.button("👥 Voir Patients", use_container_width=True):
            st.session_state.page = "👥 Liste Patients"
            st.rerun()
    
    with col3:
        if st.button("🔍 Rechercher", use_container_width=True):
            st.session_state.page = "🔍 Rechercher"
            st.rerun()

elif page == "➕ Nouveau Patient":
    # Barre de recherche en haut
    st.markdown("""
    <style>
    .search-bar {
        background: rgba(236,72,153,0.05);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(236,72,153,0.2);
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="search-bar">', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    with col1:
        search_term = st.text_input("🔍 Recherche rapide", 
                                   placeholder="Rechercher un patient par nom, prénom, téléphone, ID...",
                                   key="quick_search_new",
                                   label_visibility="collapsed")
    with col2:
        if st.button("Rechercher", use_container_width=True, key="btn_quick_search_new"):
            if search_term:
                from utils.patients import search_patients
                results = search_patients(search_term)
                if results:
                    st.success(f"📊 {len(results)} patient(s) trouvé(s)")
                    for patient in results:
                        status_icon = {"en_attente": "⏳", "en_cours": "🔄", "complete": "✅"}.get(patient.get('status', ''), "❓")
                        st.info(f"{status_icon} {patient.get('prenom', '')} {patient.get('nom', '')} - {patient.get('patient_id', '')}")
                else:
                    st.info("Aucun patient trouvé.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    from modules.secretaire import render_new_patient
    render_new_patient()

elif page == "👥 Liste Patients":
    # Barre de recherche en haut
    st.markdown("""
    <style>
    .search-bar {
        background: rgba(236,72,153,0.05);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(236,72,153,0.2);
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="search-bar">', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    with col1:
        search_term = st.text_input("🔍 Recherche rapide", 
                                   placeholder="Rechercher un patient par nom, prénom, téléphone, ID...",
                                   key="quick_search_list",
                                   label_visibility="collapsed")
    with col2:
        if st.button("Rechercher", use_container_width=True, key="btn_quick_search_list"):
            if search_term:
                from utils.patients import search_patients
                results = search_patients(search_term)
                if results:
                    st.success(f"📊 {len(results)} patient(s) trouvé(s)")
                    for patient in results:
                        status_icon = {"en_attente": "⏳", "en_cours": "🔄", "complete": "✅"}.get(patient.get('status', ''), "❓")
                        st.info(f"{status_icon} {patient.get('prenom', '')} {patient.get('nom', '')} - {patient.get('patient_id', '')}")
                else:
                    st.info("Aucun patient trouvé.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    from modules.secretaire import render_list_patients
    render_list_patients()

elif page == "📅 Rendez-vous":
    # Barre de recherche en haut
    st.markdown("""
    <style>
    .search-bar {
        background: rgba(236,72,153,0.05);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(236,72,153,0.2);
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="search-bar">', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    with col1:
        search_term = st.text_input("🔍 Recherche rapide", 
                                   placeholder="Rechercher un patient par nom, prénom, téléphone, ID...",
                                   key="quick_search_rdv",
                                   label_visibility="collapsed")
    with col2:
        if st.button("Rechercher", use_container_width=True, key="btn_quick_search_rdv"):
            if search_term:
                from utils.patients import search_patients
                results = search_patients(search_term)
                if results:
                    st.success(f"📊 {len(results)} patient(s) trouvé(s)")
                    for patient in results:
                        status_icon = {"en_attente": "⏳", "en_cours": "🔄", "complete": "✅"}.get(patient.get('status', ''), "❓")
                        st.info(f"{status_icon} {patient.get('prenom', '')} {patient.get('nom', '')} - {patient.get('patient_id', '')}")
                else:
                    st.info("Aucun patient trouvé.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    from modules.secretaire import render_appointments
    render_appointments()

elif page == "📊 Statistiques":
    # Barre de recherche en haut
    st.markdown("""
    <style>
    .search-bar {
        background: rgba(236,72,153,0.05);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(236,72,153,0.2);
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="search-bar">', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    with col1:
        search_term = st.text_input("🔍 Recherche rapide", 
                                   placeholder="Rechercher un patient par nom, prénom, téléphone, ID...",
                                   key="quick_search_stats",
                                   label_visibility="collapsed")
    with col2:
        if st.button("Rechercher", use_container_width=True, key="btn_quick_search_stats"):
            if search_term:
                from utils.patients import search_patients
                results = search_patients(search_term)
                if results:
                    st.success(f"📊 {len(results)} patient(s) trouvé(s)")
                    for patient in results:
                        status_icon = {"en_attente": "⏳", "en_cours": "🔄", "complete": "✅"}.get(patient.get('status', ''), "❓")
                        st.info(f"{status_icon} {patient.get('prenom', '')} {patient.get('nom', '')} - {patient.get('patient_id', '')}")
                else:
                    st.info("Aucun patient trouvé.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    from modules.secretaire import render_statistics
    render_statistics()

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

elif page == "🩸 Analyse PTDM":
    page_header("🩸 Module PTDM",
                "Diagnostic Diabète Post-Transplantation",
                "Évaluez le risque de diabète à partir des paramètres cliniques du patient et du donneur")
    from modules.ptdm_prediction import render
    render()

elif page == "📊 Dashboard PTDM":
    page_header("📊 Analytics PTDM",
                "Tableau de Bord PTDM",
                "Statistiques sur les transplantations et la prévalence du risque PTDM")
    from modules.ptdm_dashboard import render
    render()

elif page == "📜 Historique":
    page_header("📜 Registre Médical",
                "Historique des Prédictions",
                "Consultez, filtrez et exportez l'ensemble des analyses effectuées")
    from modules.historique import render
    render()

elif page == "ℹ️ À Propos":
    page_header("ℹ️ Documentation",
                "À Propos de NovaClinic",
                "Architecture, pipeline ML et informations cliniques de référence")
    from modules.apropos import render
    render()
