"""
Authentification — login + 2FA TOTP (Firestore ou local)
"""
import io
import json
import os
import bcrypt
import pyotp
import qrcode
import streamlit as st
from PIL import Image
from utils.firebase import (
    firebase_ready,
    create_user_firestore,
    get_user_firestore,
    update_user_firestore
)

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
    # Firebase
    if firebase_ready:
        create_user_firestore(username, hashed, secret)
    # Local fallback
    else:
        users = _load_users()
        users[username] = {"password": hashed, "totp_secret": secret, "totp_verified": False}
        _save_users(users)
    return secret


def verify_password(username: str, password: str) -> bool:
    if firebase_ready:
        user = get_user_firestore(username)
        if not user:
            return False
        return bcrypt.checkpw(password.encode(), user["password_hash"].encode())
    else:
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
        if firebase_ready:
            update_user_firestore(username, totp_verified=True)
        else:
            users = _load_users()
            users[username]["totp_verified"] = True
            _save_users(users)
    return valid


def get_totp_secret(username: str) -> str:
    if firebase_ready:
        user = get_user_firestore(username)
        return user.get("totp_secret") if user else None
    else:
        users = _load_users()
        return users.get(username, {}).get("totp_secret")


def is_totp_verified(username: str) -> bool:
    if firebase_ready:
        user = get_user_firestore(username)
        return user.get("totp_verified", False) if user else False
    else:
        users = _load_users()
        return users.get(username, {}).get("totp_verified", False)


def user_exists(username: str) -> bool:
    if firebase_ready:
        return get_user_firestore(username) is not None
    else:
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
    .auth-wrapper {
        max-width: 420px;
        margin: 4rem auto;
        padding: 2.5rem 2.5rem 2rem;
        background: #111827;
        border: 1px solid rgba(0,212,255,0.18);
        border-radius: 20px;
        box-shadow: 0 0 60px rgba(0,212,255,0.07), 0 20px 60px rgba(0,0,0,0.5);
        animation: authPop 0.5s cubic-bezier(0.34,1.56,0.64,1) both;
    }
    @keyframes authPop {
        from { transform: scale(0.92) translateY(30px); opacity: 0; }
        to   { transform: scale(1)    translateY(0);    opacity: 1; }
    }
    .auth-logo {
        text-align: center;
        margin-bottom: 1.8rem;
    }
    .auth-logo .icon {
        font-size: 3rem;
        display: block;
        margin-bottom: 0.4rem;
        animation: iconGlow 3s ease-in-out infinite alternate;
    }
    @keyframes iconGlow {
        0%   { filter: drop-shadow(0 0 8px rgba(0,212,255,0.3)); }
        100% { filter: drop-shadow(0 0 22px rgba(0,212,255,0.8)); }
    }
    .auth-logo h1 {
        margin: 0;
        font-size: 1.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .auth-logo p {
        margin: 0.3rem 0 0;
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .auth-step {
        display: inline-block;
        background: rgba(0,212,255,0.1);
        border: 1px solid rgba(0,212,255,0.25);
        border-radius: 20px;
        padding: 0.2rem 0.9rem;
        font-size: 0.72rem;
        color: #00d4ff;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
        animation: stepPulse 2.5s ease-in-out infinite;
    }
    @keyframes stepPulse {
        0%,100% { box-shadow: none; }
        50%      { box-shadow: 0 0 10px rgba(0,212,255,0.25); }
    }
    .auth-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,212,255,0.2), transparent);
        margin: 1.2rem 0;
    }
    .qr-container {
        background: #0a0e1a;
        border: 1px solid rgba(0,212,255,0.2);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .secret-box {
        background: rgba(0,212,255,0.06);
        border: 1px dashed rgba(0,212,255,0.3);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-family: monospace;
        font-size: 1rem;
        color: #00d4ff;
        text-align: center;
        letter-spacing: 0.15em;
        margin: 0.5rem 0;
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
        st.markdown("""
        <div class="auth-logo">
            <span class="icon">🧬</span>
            <h1>MedAI Thyroid</h1>
            <p>Accès Sécurisé — Authentification</p>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="auth-step">Étape 1 / 2 — Identifiants</div>', unsafe_allow_html=True)

        username = st.text_input("Nom d'utilisateur", placeholder="ex: docteur.martin")
        password = st.text_input("Mot de passe", type="password", placeholder="••••••••")

        if st.button("Continuer →", use_container_width=True):
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

        st.markdown('<div class="auth-divider"></div>', unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;font-size:0.78rem;color:#94a3b8;'>Pas encore de compte ?</div>", unsafe_allow_html=True)
        if st.button("Créer un compte", use_container_width=True):
            st.session_state.auth_step = "register"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    # ── Étape 2 : TOTP ───────────────────────────────────────────────────────
    elif st.session_state.auth_step == "totp":
        username = st.session_state.auth_username

        # Premier login : afficher QR code
        if not is_totp_verified(username):
            st.markdown('<div class="auth-wrapper">', unsafe_allow_html=True)
            st.markdown("""
            <div class="auth-logo">
                <span class="icon">📱</span>
                <h1>Configuration 2FA</h1>
                <p>Première connexion — Scanner le QR code</p>
            </div>""", unsafe_allow_html=True)
            st.markdown('<div class="auth-step">Configuration initiale</div>', unsafe_allow_html=True)

            st.markdown("""
            <div style='font-size:0.85rem;color:#94a3b8;margin-bottom:1rem;line-height:1.6;'>
            Scannez ce QR code avec <b style='color:#f1f5f9'>Google Authenticator</b>,
            <b style='color:#f1f5f9'>Authy</b> ou toute app TOTP compatible.
            </div>""", unsafe_allow_html=True)

            qr_img = generate_qr_image(username)
            buf    = io.BytesIO()
            qr_img.save(buf, format="PNG")
            st.markdown('<div class="qr-container">', unsafe_allow_html=True)
            st.image(buf.getvalue(), width=200)
            st.markdown('</div>', unsafe_allow_html=True)

            secret = get_totp_secret(username)
            st.markdown(f'<div class="secret-box">{secret}</div>', unsafe_allow_html=True)
            st.markdown("<div style='text-align:center;font-size:0.75rem;color:#94a3b8;margin-bottom:1rem;'>Clé manuelle si scan impossible</div>", unsafe_allow_html=True)
            st.markdown('<div class="auth-divider"></div>', unsafe_allow_html=True)

        else:
            st.markdown('<div class="auth-wrapper">', unsafe_allow_html=True)
            st.markdown("""
            <div class="auth-logo">
                <span class="icon">🔐</span>
                <h1>Vérification 2FA</h1>
                <p>Code à usage unique</p>
            </div>""", unsafe_allow_html=True)
            st.markdown('<div class="auth-step">Étape 2 / 2 — Code TOTP</div>', unsafe_allow_html=True)

        code = st.text_input("Code à 6 chiffres", placeholder="000000",
                             max_chars=6, label_visibility="collapsed" if not is_totp_verified(username) else "visible")
        st.caption("Code généré par votre application d'authentification (expire toutes les 30s)")

        if st.button("✅ Vérifier le code", use_container_width=True):
            if len(code) != 6 or not code.isdigit():
                st.error("Le code doit contenir exactement 6 chiffres.")
            elif verify_totp(username, code):
                st.session_state.authenticated = True
                st.session_state.auth_step = "login"
                st.success(f"Bienvenue, {username} 👋")
                st.rerun()
            else:
                st.error("Code incorrect ou expiré. Réessayez.")

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
            <p>MedAI Thyroid</p>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="auth-step">Nouveau compte</div>', unsafe_allow_html=True)

        new_user = st.text_input("Nom d'utilisateur", placeholder="ex: docteur.martin", key="reg_user")
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
                st.success("Compte créé ! Configurez maintenant votre 2FA.")
                st.rerun()

        st.markdown('<div class="auth-divider"></div>', unsafe_allow_html=True)
        if st.button("← Retour à la connexion", use_container_width=True):
            st.session_state.auth_step = "login"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()
