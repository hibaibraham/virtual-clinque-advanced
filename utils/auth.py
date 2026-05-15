"""
Authentification — login + 2FA TOTP (stockage JSON local)
"""
import io
import json
import os
import bcrypt
import pyotp
import qrcode
import streamlit as st
from PIL import Image

USERS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'users.json')
APP_NAME   = "MedAI Thyroid"


# ── Persistance ──────────────────────────────────────────────────────────────

def _load_users() -> dict:
    if not os.path.exists(USERS_PATH):
        return {}
    with open(USERS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save_users(users: dict):
    with open(USERS_PATH, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2)


# ── Gestion des comptes (Firestore ou local) ─────────────────────────────────

def create_user(username: str, password: str) -> str:
    """Crée un utilisateur et retourne le secret TOTP."""
    # Vérifier existence
    if user_exists(username):
        raise ValueError(f"L'utilisateur '{username}' existe déjà.")
    # Hash password
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    secret = pyotp.random_base32()
    # Stockage JSON local
    users = _load_users()
    users[username] = {"password": hashed, "totp_secret": secret, "totp_verified": False}
    _save_users(users)
    return secret


def verify_password(username: str, password: str) -> bool:
    users = _load_users()
    if username not in users:
        return False
    return bcrypt.checkpw(password.encode(), users[username]["password"].encode())


def verify_totp(username: str, code: str) -> bool:
    secret = get_totp_secret(username)
    if not secret:
        return False
    totp  = pyotp.TOTP(secret)
    valid = totp.verify(code, valid_window=1)
    if valid and not is_totp_verified(username):
        users = _load_users()
        users[username]["totp_verified"] = True
        _save_users(users)
    return valid


def get_totp_secret(username: str) -> str:
    users = _load_users()
    return users.get(username, {}).get("totp_secret")


def is_totp_verified(username: str) -> bool:
    users = _load_users()
    return users.get(username, {}).get("totp_verified", False)


def user_exists(username: str) -> bool:
    return username in _load_users()


def get_user_role(username: str) -> str:
    """Retourne le rôle de l'utilisateur (medecin, patient, secretaire)."""
    users = _load_users()
    return users.get(username, {}).get("role", "patient")


# ── QR Code ──────────────────────────────────────────────────────────────────

def generate_qr_image(username: str) -> Image.Image:
    secret = get_totp_secret(username)
    uri    = pyotp.totp.TOTP(secret).provisioning_uri(name=username, issuer_name=APP_NAME)
    qr     = qrcode.QRCode(box_size=6, border=3)
    qr.add_data(uri)
    qr.make(fit=True)
    return qr.make_image(fill_color="#00d4ff", back_color="#0a0e1a")


# ── UI Streamlit ──────────────────────────────────────────────────────────────

def _auth_styles():
    st.markdown("""
    <style>
    /* Conteneur principal plus compact */
    .auth-wrapper {
        max-width: 380px;
        margin: 2rem auto;
        padding: 1.8rem 2rem;
        background: #111827;
        border: 1px solid rgba(0,212,255,0.15);
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3), 0 4px 16px rgba(0,212,255,0.05);
    }
    
    /* Logo plus sobre */
    .auth-logo {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .auth-logo .icon {
        font-size: 2.2rem;
        display: block;
        margin-bottom: 0.4rem;
        color: #00d4ff;
    }
    .auth-logo h1 {
        margin: 0;
        font-size: 1.3rem;
        font-weight: 600;
        color: #f1f5f9;
    }
    .auth-logo p {
        margin: 0.3rem 0 0;
        font-size: 0.8rem;
        color: #94a3b8;
    }
    
    /* Champs de formulaire plus compacts */
    .stTextInput > div > div > input,
    .stTextInput > div > div > input:focus {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        color: #f1f5f9;
        font-size: 0.88rem;
        padding: 0.5rem 0.8rem;
        height: 38px;
    }
    
    /* Labels plus petits */
    .stTextInput > label {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-bottom: 0.25rem;
        font-weight: 500;
    }
    
    /* Boutons principaux plus compacts */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1rem;
        font-weight: 500;
        font-size: 0.88rem;
        transition: all 0.2s;
        height: 40px;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,212,255,0.3);
    }
    
    /* Boutons de sélection de rôle - plus compacts */
    .role-button {
        display: inline-block;
        width: 100%;
        padding: 1.2rem 0.8rem;
        background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(124,58,237,0.08));
        border: 2px solid rgba(0,212,255,0.25);
        border-radius: 12px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
        margin-bottom: 0.8rem;
    }
    .role-button:hover {
        background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(124,58,237,0.15));
        border: 2px solid rgba(0,212,255,0.4);
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,212,255,0.2);
    }
    .role-button .icon {
        font-size: 2rem;
        display: block;
        margin-bottom: 0.4rem;
    }
    .role-button .title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #f1f5f9;
    }
    
    /* QR Code container */
    .qr-container {
        background: #0a0e1a;
        border: 1px solid rgba(0,212,255,0.1);
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
        margin: 0.8rem 0;
    }
    
    /* Secret box */
    .secret-box {
        background: rgba(0,212,255,0.05);
        border: 1px dashed rgba(0,212,255,0.2);
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        font-family: 'Courier New', monospace;
        font-size: 0.78rem;
        color: #00d4ff;
        text-align: center;
        margin: 0.5rem 0;
        word-break: break-all;
    }
    
    /* Messages d'erreur/succès */
    .stAlert {
        border-radius: 8px;
        padding: 0.5rem 0.7rem;
        font-size: 0.82rem;
        margin: 0.5rem 0;
    }
    
    /* Ligne de séparation */
    .auth-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #334155, transparent);
        margin: 1rem 0;
    }
    
    /* Ajustement des colonnes pour les boutons */
    div[data-testid="column"] {
        padding: 0 0.25rem;
    }
    
    /* Espacement entre éléments */
    .stTextInput {
        margin-bottom: 0.8rem;
    }
    
    /* Responsive */
    @media (max-width: 500px) {
        .auth-wrapper {
            max-width: 95%;
            padding: 1.5rem 1.3rem;
            margin: 1rem auto;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def require_auth():
    """
    Appeler en tête de app.py.
    Bloque l'app et affiche le flow login → 2FA si non authentifié.
    """
    _auth_styles()

    # État de session
    for key, default in [
        ("authenticated", False),
        ("auth_step", "select_role"),   # select_role | login | totp | register
        ("auth_username", ""),
        ("selected_role", ""),           # medecin ou secretaire
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    if st.session_state.authenticated:
        return  # accès accordé

    # ── Étape 0 : Sélection du Rôle ──────────────────────────────────────────
    if st.session_state.auth_step == "select_role":
        # Forcer une largeur réduite avec colonnes vides
        _, col_center, _ = st.columns([1, 2, 1])
        
        with col_center:
            # Logo
            st.markdown("""
            <div style='text-align:center;margin-bottom:1.5rem;'>
                <div style='font-size:2.2rem;margin-bottom:0.4rem;color:#00d4ff;'>🏥</div>
                <h1 style='margin:0;font-size:1.3rem;font-weight:600;color:#f1f5f9;'>Clinique Virtuelle</h1>
                <p style='margin:0.3rem 0 0;font-size:0.8rem;color:#94a3b8;'>Sélectionnez votre espace</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
            
            # Boutons de sélection
            col1, col2 = st.columns(2, gap="small")
            
            with col1:
                st.markdown("""
                <div style='padding:1rem 0.6rem;background:linear-gradient(135deg, rgba(0,212,255,0.08), rgba(124,58,237,0.08));
                            border:2px solid rgba(0,212,255,0.25);border-radius:12px;text-align:center;margin-bottom:0.6rem;'>
                    <div style='font-size:1.8rem;margin-bottom:0.4rem;'>👨‍⚕️</div>
                    <div style='font-size:0.9rem;font-weight:600;color:#f1f5f9;'>Médecin</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Accéder", use_container_width=True, key="btn_medecin"):
                    st.session_state.selected_role = "medecin"
                    st.session_state.auth_step = "login"
                    st.rerun()
            
            with col2:
                st.markdown("""
                <div style='padding:1rem 0.6rem;background:linear-gradient(135deg, rgba(0,212,255,0.08), rgba(124,58,237,0.08));
                            border:2px solid rgba(0,212,255,0.25);border-radius:12px;text-align:center;margin-bottom:0.6rem;'>
                    <div style='font-size:1.8rem;margin-bottom:0.4rem;'>📋</div>
                    <div style='font-size:0.9rem;font-weight:600;color:#f1f5f9;'>Secrétaire</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Accéder", use_container_width=True, key="btn_secretaire"):
                    st.session_state.selected_role = "secretaire"
                    st.session_state.auth_step = "login"
                    st.rerun()
        
        st.stop()

    # ── Étape 1 : Login ──────────────────────────────────────────────────────
    if st.session_state.auth_step == "login":
        selected_role = st.session_state.get("selected_role", "")
        role_names = {"medecin": "Médecin", "secretaire": "Secrétaire"}
        role_icons = {"medecin": "👨‍⚕️", "secretaire": "📋"}
        
        # Forcer une largeur réduite avec colonnes vides
        _, col_center, _ = st.columns([1.5, 2, 1.5])
        
        with col_center:
            # Logo avec rôle sélectionné
            st.markdown(f"""
            <div style='text-align:center;margin-bottom:1.5rem;'>
                <div style='font-size:2.2rem;margin-bottom:0.4rem;color:#00d4ff;'>{role_icons.get(selected_role, "🔐")}</div>
                <h1 style='margin:0;font-size:1.3rem;font-weight:600;color:#f1f5f9;'>Connexion {role_names.get(selected_role, "")}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # Champs de formulaire
            username = st.text_input("Nom d'utilisateur", placeholder="Votre identifiant")
            password = st.text_input("Mot de passe", type="password", placeholder="Votre mot de passe")
            
            # Bouton de connexion
            if st.button("Se connecter", use_container_width=True):
                if not username or not password:
                    st.error("Veuillez remplir tous les champs.")
                elif not user_exists(username):
                    st.error("Utilisateur introuvable.")
                elif not verify_password(username, password):
                    st.error("Mot de passe incorrect.")
                else:
                    # Vérifier que le rôle correspond
                    user_role = get_user_role(username)
                    if user_role != selected_role:
                        st.error(f"❌ Ce compte n'est pas un compte {role_names.get(selected_role, '')}.")
                        st.warning(f"Vous essayez d'accéder à l'espace {role_names.get(selected_role, '')} mais votre compte est de type {role_names.get(user_role, '')}.")
                    else:
                        st.session_state.auth_username = username
                        st.session_state.auth_step = "totp"
                        st.rerun()
            
            # Ligne de séparation
            st.markdown('<div style="height:1px;background:linear-gradient(90deg, transparent, #334155, transparent);margin:1rem 0;"></div>', unsafe_allow_html=True)
            
            # Boutons de navigation
            col1, col2 = st.columns(2, gap="small")
            with col1:
                if st.button("← Retour", use_container_width=True):
                    st.session_state.auth_step = "select_role"
                    st.session_state.selected_role = ""
                    st.rerun()
            with col2:
                if st.button("Créer un compte", use_container_width=True, type="secondary"):
                    st.session_state.auth_step = "register"
                    st.rerun()

        st.stop()

    # ── Étape 2 : TOTP ───────────────────────────────────────────────────────
    elif st.session_state.auth_step == "totp":
        username = st.session_state.auth_username

        # Forcer une largeur réduite avec colonnes vides
        _, col_center, _ = st.columns([1.5, 2, 1.5])
        
        with col_center:
            # Premier login : afficher QR code
            if not is_totp_verified(username):
                st.markdown("""
                <div style='text-align:center;margin-bottom:1.5rem;'>
                    <div style='font-size:2.2rem;margin-bottom:0.4rem;color:#00d4ff;'>📱</div>
                    <h1 style='margin:0;font-size:1.3rem;font-weight:600;color:#f1f5f9;'>Configuration 2FA</h1>
                </div>""", unsafe_allow_html=True)
                
                st.markdown("""
                <div style='font-size:0.85rem;color:#94a3b8;margin-bottom:1rem;line-height:1.6;'>
                Scannez ce QR code avec votre application d'authentification.
                </div>""", unsafe_allow_html=True)

                qr_img = generate_qr_image(username)
                buf    = io.BytesIO()
                qr_img.save(buf, format="PNG")
                st.markdown('<div class="qr-container">', unsafe_allow_html=True)
                st.image(buf.getvalue(), width=180)
                st.markdown('</div>', unsafe_allow_html=True)

                secret = get_totp_secret(username)
                st.markdown(f'<div class="secret-box">{secret}</div>', unsafe_allow_html=True)
                st.markdown("<div style='text-align:center;font-size:0.75rem;color:#94a3b8;margin-bottom:1rem;'>Clé manuelle</div>", unsafe_allow_html=True)

            else:
                st.markdown("""
                <div style='text-align:center;margin-bottom:1.5rem;'>
                    <div style='font-size:2.2rem;margin-bottom:0.4rem;color:#00d4ff;'>🔐</div>
                    <h1 style='margin:0;font-size:1.3rem;font-weight:600;color:#f1f5f9;'>Vérification 2FA</h1>
                </div>""", unsafe_allow_html=True)

            code = st.text_input("Code à 6 chiffres", placeholder="000000", max_chars=6)
            st.caption("Code de votre application d'authentification")

            if st.button("Vérifier", use_container_width=True):
                if len(code) != 6 or not code.isdigit():
                    st.error("Le code doit contenir 6 chiffres.")
                elif verify_totp(username, code):
                    st.session_state.authenticated = True
                    st.session_state.auth_step = "login"
                    st.success(f"Bienvenue, {username}")
                    st.rerun()
                else:
                    st.error("Code incorrect ou expiré.")

            st.markdown('<div style="height:1px;background:linear-gradient(90deg, transparent, #334155, transparent);margin:1rem 0;"></div>', unsafe_allow_html=True)
            if st.button("← Retour", use_container_width=True):
                st.session_state.auth_step = "select_role"
                st.session_state.selected_role = ""
                st.rerun()

        st.stop()

    # ── Inscription ───────────────────────────────────────────────────────────
    elif st.session_state.auth_step == "register":
        selected_role = st.session_state.get("selected_role", "medecin")
        role_names = {"medecin": "Médecin", "secretaire": "Secrétaire"}
        role_icons = {"medecin": "👨‍⚕️", "secretaire": "📋"}
        
        # Forcer une largeur réduite avec colonnes vides
        _, col_center, _ = st.columns([1.5, 2, 1.5])
        
        with col_center:
            st.markdown(f"""
            <div style='text-align:center;margin-bottom:1.5rem;'>
                <div style='font-size:2.2rem;margin-bottom:0.4rem;color:#00d4ff;'>{role_icons.get(selected_role, "🧬")}</div>
                <h1 style='margin:0;font-size:1.3rem;font-weight:600;color:#f1f5f9;'>Créer un compte {role_names.get(selected_role, "")}</h1>
            </div>""", unsafe_allow_html=True)

            new_user = st.text_input("Nom d'utilisateur", placeholder="Votre identifiant", key="reg_user")
            new_pass = st.text_input("Mot de passe", type="password", placeholder="Min. 8 caractères", key="reg_pass")
            new_pass2 = st.text_input("Confirmer le mot de passe", type="password", placeholder="••••••••", key="reg_pass2")

            if st.button("Créer le compte", use_container_width=True):
                if not new_user or not new_pass:
                    st.error("Tous les champs sont requis.")
                elif len(new_pass) < 8:
                    st.error("Le mot de passe doit contenir au moins 8 caractères.")
                elif new_pass != new_pass2:
                    st.error("Les mots de passe ne correspondent pas.")
                elif user_exists(new_user):
                    st.error("Ce nom d'utilisateur est déjà pris.")
                else:
                    secret = create_user(new_user, new_pass)
                    # Ajouter le rôle sélectionné
                    users = _load_users()
                    users[new_user]["role"] = selected_role
                    _save_users(users)
                    st.session_state.auth_username = new_user
                    st.session_state.auth_step = "totp"
                    st.success("Compte créé !")
                    st.rerun()

            st.markdown('<div style="height:1px;background:linear-gradient(90deg, transparent, #334155, transparent);margin:1rem 0;"></div>', unsafe_allow_html=True)
            if st.button("← Retour", use_container_width=True):
                st.session_state.auth_step = "login"
                st.rerun()

        st.stop()
