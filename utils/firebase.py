"""
Wrapper Firebase — Firestore + Auth
"""
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore, auth
from firebase_admin.exceptions import FirebaseError
from datetime import datetime
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, 'firebase_config.json')

# ── Initialisation ────────────────────────────────────────────────────────────

def _init_firebase():
    if not firebase_admin._apps:
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"Fichier de config Firebase introuvable : {CONFIG_PATH}")
        cred = credentials.Certificate(CONFIG_PATH)
        firebase_admin.initialize_app(cred)
    return firestore.client()


try:
    db = _init_firebase()
    firebase_ready = True
except Exception as e:
    print(f"⚠️ Firebase non configuré : {e}")
    db = None
    firebase_ready = False


# ── Firestore — Users ────────────────────────────────────────────────────────

def create_user_firestore(username: str, password_hash: str, totp_secret: str):
    if not firebase_ready:
        return
    user_ref = db.collection('users').document(username)
    user_ref.set({
        'username': username,
        'password_hash': password_hash,
        'totp_secret': totp_secret,
        'totp_verified': False,
        'created_at': datetime.utcnow(),
        'last_login': None
    })


def get_user_firestore(username: str) -> dict:
    if not firebase_ready:
        return None
    doc = db.collection('users').document(username).get()
    return doc.to_dict() if doc.exists else None


def update_user_firestore(username: str, **kwargs):
    if not firebase_ready:
        return
    db.collection('users').document(username).update(kwargs)


# ── Firestore — Predictions History ───────────────────────────────────────────

def save_prediction_firestore(patient_data: dict, prediction: int, probability: float):
    if not firebase_ready:
        return None
    username = st.session_state.get("auth_username", "anonymous")
    pred_ref = db.collection('predictions').document()
    record = {
        'id': pred_ref.id,
        'username': username,
        'timestamp': datetime.utcnow(),
        'prediction': 'Pathologique' if prediction == 1 else 'Normal',
        'probability': probability,
        'patient_data': patient_data,
        'created_at': datetime.utcnow()
    }
    pred_ref.set(record)
    return record


def get_user_predictions_firestore(username: str, limit: int = 100):
    if not firebase_ready:
        return []
    preds = (db.collection('predictions')
                .where('username', '==', username)
                .order_by('timestamp', direction=firestore.Query.DESCENDING)
                .limit(limit)
                .stream())
    return [doc.to_dict() for doc in preds]


def get_all_predictions_firestore(limit: int = 200):
    if not firebase_ready:
        return []
    preds = (db.collection('predictions')
                .order_by('timestamp', direction=firestore.Query.DESCENDING)
                .limit(limit)
                .stream())
    return [doc.to_dict() for doc in preds]


def delete_prediction_firestore(prediction_id: str):
    if not firebase_ready:
        return False
    db.collection('predictions').document(prediction_id).delete()
    return True


# ── Firebase Auth (optionnel) ────────────────────────────────────────────────

def create_firebase_auth_user(email: str, password: str):
    if not firebase_ready:
        return None
    try:
        user = auth.create_user(email=email, password=password)
        return user.uid
    except FirebaseError as e:
        print(f"Erreur création utilisateur Firebase Auth: {e}")
        return None


def verify_firebase_token(id_token: str):
    if not firebase_ready:
        return None
    try:
        decoded = auth.verify_id_token(id_token)
        return decoded
    except FirebaseError:
        return None


# ── Fallback local (si Firebase non configuré) ────────────────────────────────

def is_firebase_enabled():
    return firebase_ready


def get_firebase_status():
    return {
        'enabled': firebase_ready,
        'config_exists': os.path.exists(CONFIG_PATH),
        'project_id': json.load(open(CONFIG_PATH, 'r')).get('project_id', 'N/A') if os.path.exists(CONFIG_PATH) else 'N/A'
    }
