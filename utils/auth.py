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
        padding: 1.8rem 2rem 1.5rem;
        background: #111827;
        border: 1px solid rgba(0,212,255,0.15);
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3), 0 4px 16px rgba(0,212,255,0.05);
    }
    
    /* Logo plus sobre */
    .auth-logo {
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .auth-logo .icon {
        font-size: 2.2rem;
        display: block;
        margin-bottom: 0.3rem;
        color: #00d4ff;
    }
    .auth-logo h1 {
        margin: 0;
        font-size: 1.2rem;
        font-weight: 600;
        color: #f1f5f9;
    }
    .auth-logo p {
        margin: 0.2rem 0 0;
        font-size: 0.75rem;
        color: #94a3b8;
    }
    
    /* Indicateur d'étape discret */
    .auth-step {
        display: block;
        text-align: center;
        font-size: 0.7rem;
        color: #00d4ff;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(0,212,255,0.1);
    }
    
    /* Champs de formulaire */
    .stTextInput > div > div > input,
    .stTextInput > div > div > input:focus {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        color: #f1f5f9;
        font-size: 0.9rem;
        padding: 0.5rem 0.75rem;
    }
    
    /* Boutons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,212,255,0.2);
    }
    
    /* QR Code container */
    .qr-container {
        background: #0a0e1a;
        border: 1px solid rgba(0,212,255,0.1);
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Secret box */
    .secret-box {
        background: rgba(0,212,255,0.05);
        border: 1px dashed rgba(0,212,255,0.2);
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: #00d4ff;
        text-align: center;
        margin: 0.5rem 0;
        word-break: break-all;
    }
    
    /* Messages d'erreur/succès */
    .stAlert {
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        font-size: 0.85rem;
        margin: 0.5rem 0;
    }
    
    /* Ligne de séparation */
    .auth-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #334155, transparent);
        margin: 1rem 0;
    }
    
    /* Texte de lien/action secondaire */
    .auth-secondary {
        text-align: center;
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.8rem;
    }
    
    /* Responsive */
    @media (max-width: 400px) {
        .auth-wrapper {
            max-width: 320px;
            padding: 1.5rem 1.2rem 1.2rem;
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
        ("auth_step", "login"),       # login | totp | setup_totp
        ("auth_username", ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    if st.session_state.authenticated:
        return  # accès accordé

    # ── Étape 1 : Login ──────────────────────────────────────────────────────
    if st.session_state.auth_step == "login":
        st.markdown('<div class="auth-wrapper">', unsafe_allow_html=True)
        
        # Logo simplifié
        st.markdown("""
        <div class="auth-logo">
            <span class="icon">🔐</span>
            <h1>Connexion</h1>
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
                st.session_state.auth_username = username
                st.session_state.auth_step = "totp"
                st.rerun()
        
        # Ligne de séparation
        st.markdown('<div class="auth-divider"></div>', unsafe_allow_html=True)
        
        # Lien vers l'inscription
        if st.button("Créer un compte", use_container_width=True, type="secondary"):
            st.session_state.auth_step = "register"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    # ── Étape 2 : TOTP ───────────────────────────────────────────────────────
    elif st.session_state.auth_step == "totp":
        username = st.session_state.auth_username

        st.markdown('<div class="auth-wrapper">', unsafe_allow_html=True)
        
        # Premier login : afficher QR code
        if not is_totp_verified(username):
            st.markdown("""
            <div class="auth-logo">
                <span class="icon">📱</span>
                <h1>Configuration 2FA</h1>
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
            <div class="auth-logo">
                <span class="icon">🔐</span>
                <h1>Vérification 2FA</h1>
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

        st.markdown('<div class="auth-divider"></div>', unsafe_allow_html=True)
        if st.button("← Retour", use_container_width=True):
            st.session_state.auth_step = "login"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    # ── Inscription ───────────────────────────────────────────────────────────
    elif st.session_state.auth_step == "register":
        st.markdown('<div class="auth-wrapper">', unsafe_allow_html=True)
        st.markdown("""
        <div class="auth-logo">
            <span class="icon">🧬</span>
            <h1>Créer un compte</h1>
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
                create_user(new_user, new_pass)
                st.session_state.auth_username = new_user
                st.session_state.auth_step = "totp"
                st.success("Compte créé !")
                st.rerun()

        st.markdown('<div class="auth-divider"></div>', unsafe_allow_html=True)
        if st.button("← Retour", use_container_width=True):
            st.session_state.auth_step = "login"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()
