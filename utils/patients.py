"""
Gestion des patients - Stockage et récupération des données patients
"""
import os
import json
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATIENTS_PATH = os.path.join(BASE_DIR, 'patients.json')
PATIENTS_CSV_PATH = os.path.join(BASE_DIR, 'patients.csv')

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
    patients = _load_patients()
    
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

def get_patient(patient_id: str) -> dict:
    """Récupère les informations d'un patient par son ID."""
    patients = _load_patients()
    return patients.get(patient_id, {})

def update_patient(patient_id: str, updates: dict):
    """Met à jour les informations d'un patient."""
    patients = _load_patients()
    if patient_id in patients:
        patients[patient_id].update(updates)
        patients[patient_id]["updated_at"] = datetime.now().isoformat()
        patients[patient_id]["updated_by"] = "medecin"  # À remplacer par l'utilisateur connecté
        patients[patient_id]["status"] = "complete" if updates.get("medical_data") else "en_cours"
        _save_patients(patients)
        
        # Mettre à jour le CSV aussi
        df_patients = _load_patients_csv()
        if not df_patients.empty and "patient_id" in df_patients.columns:
            mask = df_patients["patient_id"] == patient_id
            for key, value in updates.items():
                if key in df_patients.columns and not isinstance(value, (dict, list)):
                    df_patients.loc[mask, key] = value
            df_patients.loc[mask, "status"] = "complete" if updates.get("medical_data") else "en_cours"
            _save_patients_csv(df_patients)

def get_patients_by_status(status: str = "en_attente") -> list:
    """Récupère la liste des patients par statut."""
    patients = _load_patients()
    return [patient for patient in patients.values() if patient.get("status") == status]

def get_all_patients() -> list:
    """Récupère tous les patients."""
    patients = _load_patients()
    return list(patients.values())

def search_patients(search_term: str) -> list:
    """Recherche des patients par nom, prénom, ID, etc."""
    patients = _load_patients()
    results = []
    search_term = search_term.lower()
    
    for patient in patients.values():
        # Recherche dans les champs textuels
        for field in ["nom", "prenom", "patient_id", "telephone", "email"]:
            if field in patient and search_term in str(patient[field]).lower():
                results.append(patient)
                break
    
    return results