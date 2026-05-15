"""
Module de connexion et gestion MongoDB
"""
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os
from datetime import datetime

# Configuration MongoDB
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DATABASE_NAME = "clinique_virtuelle"

# Client MongoDB global
_client = None
_db = None

def get_database():
    """Retourne la connexion à la base de données MongoDB."""
    global _client, _db
    
    if _db is None:
        try:
            _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            # Tester la connexion
            _client.admin.command('ping')
            _db = _client[DATABASE_NAME]
            print(f"✅ Connecté à MongoDB: {DATABASE_NAME}")
        except ConnectionFailure as e:
            print(f"❌ Erreur de connexion MongoDB: {e}")
            print("⚠️ Utilisation du mode local (JSON/CSV)")
            return None
        except Exception as e:
            print(f"❌ Erreur MongoDB: {e}")
            return None
    
    return _db

def is_mongodb_available():
    """Vérifie si MongoDB est disponible."""
    db = get_database()
    return db is not None

# Collections
def get_users_collection():
    """Retourne la collection des utilisateurs."""
    db = get_database()
    return db.users if db is not None else None

def get_patients_collection():
    """Retourne la collection des patients."""
    db = get_database()
    return db.patients if db is not None else None

def get_appointments_collection():
    """Retourne la collection des rendez-vous."""
    db = get_database()
    return db.appointments if db is not None else None

def get_predictions_collection():
    """Retourne la collection des prédictions."""
    db = get_database()
    return db.predictions if db is not None else None

def get_consultations_collection():
    """Retourne la collection des consultations."""
    db = get_database()
    return db.consultations if db is not None else None

# Fonctions utilitaires
def create_indexes():
    """Crée les index pour optimiser les recherches."""
    db = get_database()
    if db is None:
        return
    
    try:
        # Index pour les patients
        db.patients.create_index("patient_id", unique=True)
        db.patients.create_index("nom")
        db.patients.create_index("prenom")
        db.patients.create_index("telephone")
        db.patients.create_index("status")
        db.patients.create_index("created_at")
        
        # Index pour les utilisateurs
        db.users.create_index("username", unique=True)
        db.users.create_index("role")
        
        # Index pour les rendez-vous
        db.appointments.create_index("patient_id")
        db.appointments.create_index("date")
        db.appointments.create_index("status")
        
        # Index pour les prédictions
        db.predictions.create_index("timestamp")
        db.predictions.create_index("username")
        
        print("✅ Index MongoDB créés avec succès")
    except Exception as e:
        print(f"⚠️ Erreur lors de la création des index: {e}")

def close_connection():
    """Ferme la connexion MongoDB."""
    global _client
    if _client:
        _client.close()
        print("✅ Connexion MongoDB fermée")
