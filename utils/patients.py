"""
Gestion des patients - Stockage et récupération des données patients
Supporte MongoDB avec fallback vers JSON/CSV
"""
import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from utils.database import get_patients_collection, is_mongodb_available

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATIENTS_PATH = os.path.join(BASE_DIR, 'patients.json')
PATIENTS_CSV_PATH = os.path.join(BASE_DIR, 'patients.csv')

# ── Fonctions JSON (fallback) ────────────────────────────────────────────────

def _load_patients() -> dict:
    """Charge les patients depuis le fichier JSON."""
    if not os.path.exists(PATIENTS_PATH):
        return {}
    with open(PATIENTS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def _save_patients(patients: dict):
    """Sauvegarde les patients dans le fichier JSON."""
    with open(PATIENTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(patients, f, indent=2, ensure_ascii=False)

def _load_patients_csv() -> pd.DataFrame:
    """Charge les patients depuis le fichier CSV."""
    if not os.path.exists(PATIENTS_CSV_PATH):
        return pd.DataFrame()
    return pd.read_csv(PATIENTS_CSV_PATH)

def _save_patients_csv(df: pd.DataFrame):
    """Sauvegarde les patients dans le fichier CSV."""
    df.to_csv(PATIENTS_CSV_PATH, index=False)

def create_patient(patient_data: dict) -> str:
    """
    Crée un nouveau patient avec les informations de base.
    Retourne l'ID du patient.
    """
    # Invalider le cache après création
    get_all_patients.clear()
    get_patients_by_status.clear()
    
    # Générer un ID unique
    patient_id = f"PAT{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Ajouter des métadonnées
    patient_record = {
        "patient_id": patient_id,
        "created_at": datetime.now().isoformat(),
        "created_by": "secretaire",  # À remplacer par l'utilisateur connecté
        "status": "en_attente",  # en_attente, en_cours, complete
        **patient_data
    }
    
    # Essayer MongoDB d'abord
    if is_mongodb_available():
        collection = get_patients_collection()
        if collection is not None:
            try:
                collection.insert_one(patient_record.copy())
                return patient_id
            except Exception as e:
                print(f"⚠️ Erreur MongoDB, fallback vers JSON: {e}")
    
    # Fallback vers JSON/CSV
    patients = _load_patients()
    patients[patient_id] = patient_record
    _save_patients(patients)
    
    # Sauvegarder aussi en CSV pour compatibilité
    df_patients = _load_patients_csv()
    new_row = {
        "patient_id": patient_id,
        "created_at": datetime.now().isoformat(),
        "status": "en_attente",
        **{k: v for k, v in patient_data.items() if not isinstance(v, (dict, list))}
    }
    df_new = pd.DataFrame([new_row])
    if df_patients.empty:
        df_patients = df_new
    else:
        df_patients = pd.concat([df_patients, df_new], ignore_index=True)
    _save_patients_csv(df_patients)
    
    return patient_id

@st.cache_data(ttl=10)  # Cache pendant 10 secondes
def get_patient(patient_id: str) -> dict:
    """Récupère les informations d'un patient par son ID."""
    # Essayer MongoDB d'abord
    if is_mongodb_available():
        collection = get_patients_collection()
        if collection is not None:
            try:
                patient = collection.find_one({"patient_id": patient_id})
                if patient:
                    patient.pop('_id', None)  # Retirer l'ID MongoDB
                    return patient
            except Exception as e:
                print(f"⚠️ Erreur MongoDB, fallback vers JSON: {e}")
    
    # Fallback vers JSON
    patients = _load_patients()
    return patients.get(patient_id, {})

def update_patient(patient_id: str, updates: dict):
    """Met à jour les informations d'un patient."""
    # Invalider le cache après mise à jour
    get_patient.clear()
    get_all_patients.clear()
    get_patients_by_status.clear()
    
    updates["updated_at"] = datetime.now().isoformat()
    updates["updated_by"] = "medecin"  # À remplacer par l'utilisateur connecté
    
    # Déterminer le nouveau statut
    if updates.get("medical_data"):
        updates["status"] = "complete"
    elif "status" not in updates:
        updates["status"] = "en_cours"
    
    # Essayer MongoDB d'abord
    if is_mongodb_available():
        collection = get_patients_collection()
        if collection is not None:
            try:
                collection.update_one(
                    {"patient_id": patient_id},
                    {"$set": updates}
                )
                return
            except Exception as e:
                print(f"⚠️ Erreur MongoDB, fallback vers JSON: {e}")
    
    # Fallback vers JSON/CSV
    patients = _load_patients()
    if patient_id in patients:
        patients[patient_id].update(updates)
        _save_patients(patients)
        
        # Mettre à jour le CSV aussi
        df_patients = _load_patients_csv()
        if not df_patients.empty and "patient_id" in df_patients.columns:
            mask = df_patients["patient_id"] == patient_id
            for key, value in updates.items():
                if key in df_patients.columns and not isinstance(value, (dict, list)):
                    df_patients.loc[mask, key] = value
            _save_patients_csv(df_patients)

@st.cache_data(ttl=10)  # Cache pendant 10 secondes
def get_patients_by_status(status: str = "en_attente") -> list:
    """Récupère la liste des patients par statut."""
    # Essayer MongoDB d'abord
    if is_mongodb_available():
        collection = get_patients_collection()
        if collection is not None:
            try:
                patients = list(collection.find({"status": status}))
                # Retirer les _id MongoDB
                for patient in patients:
                    patient.pop('_id', None)
                return patients
            except Exception as e:
                print(f"⚠️ Erreur MongoDB, fallback vers JSON: {e}")
    
    # Fallback vers JSON
    patients = _load_patients()
    return [patient for patient in patients.values() if patient.get("status") == status]

@st.cache_data(ttl=10)  # Cache pendant 10 secondes
def get_all_patients() -> list:
    """Récupère tous les patients."""
    # Essayer MongoDB d'abord
    if is_mongodb_available():
        collection = get_patients_collection()
        if collection is not None:
            try:
                patients = list(collection.find({}))
                # Retirer les _id MongoDB
                for patient in patients:
                    patient.pop('_id', None)
                return patients
            except Exception as e:
                print(f"⚠️ Erreur MongoDB, fallback vers JSON: {e}")
    
    # Fallback vers JSON
    patients = _load_patients()
    return list(patients.values())

@st.cache_data(ttl=5)  # Cache pendant 5 secondes
def search_patients(search_term: str) -> list:
    """Recherche des patients par nom, prénom, ID, etc."""
    search_term = search_term.lower()
    
    # Essayer MongoDB d'abord
    if is_mongodb_available():
        collection = get_patients_collection()
        if collection is not None:
            try:
                # Recherche avec regex MongoDB (plus performant)
                query = {
                    "$or": [
                        {"nom": {"$regex": search_term, "$options": "i"}},
                        {"prenom": {"$regex": search_term, "$options": "i"}},
                        {"patient_id": {"$regex": search_term, "$options": "i"}},
                        {"telephone": {"$regex": search_term, "$options": "i"}},
                        {"email": {"$regex": search_term, "$options": "i"}}
                    ]
                }
                patients = list(collection.find(query))
                # Retirer les _id MongoDB
                for patient in patients:
                    patient.pop('_id', None)
                return patients
            except Exception as e:
                print(f"⚠️ Erreur MongoDB, fallback vers JSON: {e}")
    
    # Fallback vers JSON
    patients = _load_patients()
    results = []
    
    for patient in patients.values():
        # Recherche dans les champs textuels
        for field in ["nom", "prenom", "patient_id", "telephone", "email"]:
            if field in patient and search_term in str(patient[field]).lower():
                results.append(patient)
                break
    
    return results