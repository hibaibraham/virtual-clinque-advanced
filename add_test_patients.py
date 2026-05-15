"""
Script pour ajouter des patients de test au système
"""
import json
import os
from datetime import datetime, timedelta
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATIENTS_PATH = os.path.join(BASE_DIR, 'patients.json')

def load_patients():
    """Charge les patients depuis le fichier JSON."""
    if not os.path.exists(PATIENTS_PATH):
        return {}
    with open(PATIENTS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_patients(patients):
    """Sauvegarde les patients dans le fichier JSON."""
    with open(PATIENTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(patients, f, indent=2, ensure_ascii=False)

def add_test_patients():
    """Ajoute des patients de test au système."""
    patients = load_patients()
    
    # Patients de test
    test_patients = [
        {
            "nom": "Dupont",
            "prenom": "Jean",
            "age": 45,
            "sexe": "Homme",
            "telephone": "0612345678",
            "email": "jean.dupont@email.com",
            "adresse": "123 Rue de Paris, 75001 Paris",
            "profession": "Ingénieur",
            "date_naissance": "1979-03-15",
            "poids": 78.5,
            "taille": 175.0,
            "groupe_sanguin": "A+",
            "antecedents": {
                "familiaux": "Diabète type 2 (père)",
                "personnels": "Hypertension légère",
                "allergies": "Pénicilline",
                "medicaments": "Amlodipine 5mg"
            },
            "habitudes_vie": {
                "tabagisme": "Ancien fumeur (arrêté il y a 5 ans)",
                "alcool": "Occasionnel",
                "activite_physique": "2-3 fois/semaine"
            },
            "motif_consultation": "Contrôle tension artérielle et fatigue",
            "status": "complete",
            "medical_data": {
                "examen_clinique": {
                    "tension_arterielle": "135/85",
                    "frequence_cardiaque": 72,
                    "temperature": 36.8,
                    "saturation_o2": 98,
                    "poids_confirme": 78.5,
                    "taille_confirmee": 175.0,
                    "imc": 25.6
                },
                "laboratoire": {
                    "hematologie": {
                        "hemoglobine": 14.5,
                        "leucocytes": 7.2,
                        "plaquettes": 245
                    },
                    "biochimie": {
                        "glycemie": 102,
                        "creatinine": 0.95,
                        "cholesterol": 195
                    },
                    "autres": {
                        "crp": 3.2,
                        "vs": 18,
                        "autres_analyses": "TSH: 2.1 mUI/L"
                    }
                },
                "diagnostic": "Hypertension artérielle stade 1, fatigue liée au stress professionnel",
                "traitement": "Amlodipine 5mg 1cp/jour, conseils hygiéno-diététiques, repos",
                "recommandations": "Contrôle tensionnel mensuel, régime hyposodé, activité physique régulière",
                "notes_medecin": "Patient coopératif, à surveiller",
                "medecin": "Dr. Martin"
            }
        },
        {
            "nom": "Martin",
            "prenom": "Sophie",
            "age": 32,
            "sexe": "Femme",
            "telephone": "0698765432",
            "email": "sophie.martin@email.com",
            "adresse": "456 Avenue des Champs, 69002 Lyon",
            "profession": "Enseignante",
            "date_naissance": "1992-08-22",
            "poids": 62.0,
            "taille": 165.0,
            "groupe_sanguin": "O+",
            "antecedents": {
                "familiaux": "Cancer du sein (mère)",
                "personnels": "Asthme léger",
                "allergies": "Acariens, pollens",
                "medicaments": "Ventoline au besoin"
            },
            "habitudes_vie": {
                "tabagisme": "Non-fumeur",
                "alcool": "Rarement",
                "activite_physique": "1 fois/semaine"
            },
            "motif_consultation": "Douleurs abdominales récurrentes",
            "status": "en_cours"
        },
        {
            "nom": "Bernard",
            "prenom": "Pierre",
            "age": 58,
            "sexe": "Homme",
            "telephone": "0677889900",
            "email": "pierre.bernard@email.com",
            "adresse": "789 Boulevard Maritime, 13008 Marseille",
            "profession": "Retraité",
            "date_naissance": "1966-11-30",
            "poids": 85.0,
            "taille": 180.0,
            "groupe_sanguin": "B+",
            "antecedents": {
                "familiaux": "Maladie cardiaque (père)",
                "personnels": "Diabète type 2, hypercholestérolémie",
                "allergies": "Aucune",
                "medicaments": "Metformine 850mg, Atorvastatine 20mg"
            },
            "habitudes_vie": {
                "tabagisme": "Fumeur (20 paquets-années)",
                "alcool": "Modéré",
                "activite_physique": "Sédentaire"
            },
            "motif_consultation": "Suivi diabète et bilan cardiologique",
            "status": "en_attente"
        },
        {
            "nom": "Petit",
            "prenom": "Marie",
            "age": 28,
            "sexe": "Femme",
            "telephone": "0655443322",
            "email": "marie.petit@email.com",
            "adresse": "321 Rue du Commerce, 31000 Toulouse",
            "profession": "Infirmière",
            "date_naissance": "1996-04-12",
            "poids": 58.0,
            "taille": 168.0,
            "groupe_sanguin": "AB+",
            "antecedents": {
                "familiaux": "Aucun",
                "personnels": "Migraines",
                "allergies": "Iode",
                "medicaments": "Sumatriptan au besoin"
            },
            "habitudes_vie": {
                "tabagisme": "Non-fumeur",
                "alcool": "Occasionnel",
                "activite_physique": "4-5 fois/semaine"
            },
            "motif_consultation": "Consultation pré-conceptionnelle",
            "status": "en_attente"
        },
        {
            "nom": "Leroy",
            "prenom": "Thomas",
            "age": 40,
            "sexe": "Homme",
            "telephone": "0633221144",
            "email": "thomas.leroy@email.com",
            "adresse": "654 Rue de la République, 59000 Lille",
            "profession": "Commercial",
            "date_naissance": "1984-07-05",
            "poids": 92.0,
            "taille": 182.0,
            "groupe_sanguin": "A-",
            "antecedents": {
                "familiaux": "Obésité (parents)",
                "personnels": "Syndrome métabolique",
                "allergies": "Aucune",
                "medicaments": "Aucun"
            },
            "habitudes_vie": {
                "tabagisme": "Fumeur (10 cigarettes/jour)",
                "alcool": "Régulier",
                "activite_physique": "Sédentaire"
            },
            "motif_consultation": "Perte de poids et rééquilibrage alimentaire",
            "status": "complete",
            "medical_data": {
                "examen_clinique": {
                    "tension_arterielle": "142/88",
                    "frequence_cardiaque": 78,
                    "temperature": 36.7,
                    "saturation_o2": 97,
                    "poids_confirme": 92.0,
                    "taille_confirmee": 182.0,
                    "imc": 27.8
                },
                "laboratoire": {
                    "hematologie": {
                        "hemoglobine": 15.2,
                        "leucocytes": 8.1,
                        "plaquettes": 265
                    },
                    "biochimie": {
                        "glycemie": 118,
                        "creatinine": 1.05,
                        "cholesterol": 235
                    },
                    "autres": {
                        "crp": 5.8,
                        "vs": 22,
                        "autres_analyses": "Triglycérides: 180 mg/dL"
                    }
                },
                "diagnostic": "Syndrome métabolique, obésité abdominale, pré-diabète",
                "traitement": "Programme de perte de poids supervisé, activité physique progressive, conseils nutritionnels",
                "recommandations": "Suivi nutritionnel mensuel, activité physique 3 fois/semaine, arrêt du tabac",
                "notes_medecin": "Motivation variable, besoin de soutien régulier",
                "medecin": "Dr. Martin"
            }
        }
    ]
    
    # Ajouter les patients avec des IDs et dates différents
    base_date = datetime.now() - timedelta(days=30)
    
    for i, patient_data in enumerate(test_patients):
        patient_id = f"TEST{i+1:03d}"
        
        # Date de création différente pour chaque patient
        created_at = (base_date + timedelta(days=i*5)).isoformat()
        
        patient_record = {
            "patient_id": patient_id,
            "created_at": created_at,
            "created_by": "secretaire",
            "status": patient_data.pop("status"),
            **patient_data
        }
        
        # Ajouter la date de mise à jour si le patient est complet
        if patient_record["status"] == "complete":
            patient_record["updated_at"] = (datetime.fromisoformat(created_at) + timedelta(days=2)).isoformat()
            patient_record["updated_by"] = "medecin"
        
        patients[patient_id] = patient_record
    
    save_patients(patients)
    print(f"✅ {len(test_patients)} patients de test ajoutés avec succès !")
    print("Patients ajoutés :")
    for patient_id in patients:
        if patient_id.startswith("TEST"):
            patient = patients[patient_id]
            print(f"  - {patient['prenom']} {patient['nom']} (ID: {patient_id}, Statut: {patient['status']})")

if __name__ == "__main__":
    add_test_patients()