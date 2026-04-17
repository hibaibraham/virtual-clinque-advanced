#!/usr/bin/env python3
"""
Script de migration des données locales vers Firebase
"""
import os
import sys
import json
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def migrate_users():
    """Migrer les utilisateurs de users.json vers Firestore"""
    users_path = os.path.join(os.path.dirname(__file__), 'users.json')
    if not os.path.exists(users_path):
        print("⚠️  Aucun fichier users.json trouvé")
        return 0
    
    try:
        from utils.firebase import create_user_firestore, firebase_ready
        
        if not firebase_ready:
            print("❌ Firebase non configuré")
            return 0
            
        with open(users_path, 'r', encoding='utf-8') as f:
            users = json.load(f)
        
        migrated = 0
        for username, data in users.items():
            try:
                create_user_firestore(
                    username=username,
                    password_hash=data.get('password'),
                    totp_secret=data.get('totp_secret', '')
                )
                migrated += 1
                print(f"✅ Utilisateur migré: {username}")
            except Exception as e:
                print(f"❌ Erreur migration {username}: {e}")
        
        return migrated
    except ImportError:
        print("❌ Impossible d'importer utils.firebase")
        return 0

def migrate_predictions():
    """Migrer l'historique des prédictions vers Firestore"""
    history_path = os.path.join(os.path.dirname(__file__), 'prediction_history.csv')
    if not os.path.exists(history_path):
        print("⚠️  Aucun fichier prediction_history.csv trouvé")
        return 0
    
    try:
        from utils.firebase import save_prediction_firestore, firebase_ready
        
        if not firebase_ready:
            print("❌ Firebase non configuré")
            return 0
            
        df = pd.read_csv(history_path)
        migrated = 0
        
        for _, row in df.iterrows():
            try:
                # Extraire les données patient
                patient_data = {}
                for col in ['age', 'TSH', 'T3', 'TT4', 'FTI', 'T4U', 'sex']:
                    if col in row:
                        patient_data[col] = row[col]
                
                # Convertir la prédiction
                prediction = 1 if row.get('prediction') == 'Pathologique' else 0
                probability = float(str(row.get('probability', '0%')).replace('%', '')) / 100
                
                # Sauvegarder dans Firestore
                save_prediction_firestore(
                    patient_data=patient_data,
                    prediction=prediction,
                    probability=probability
                )
                migrated += 1
                
                if migrated % 10 == 0:
                    print(f"✅ {migrated} prédictions migrées...")
                    
            except Exception as e:
                print(f"❌ Erreur migration ligne {_}: {e}")
        
        return migrated
    except ImportError:
        print("❌ Impossible d'importer utils.firebase")
        return 0

def main():
    print("=== Migration vers Firebase ===")
    print("Ce script migre les données locales vers Firebase Firestore")
    print()
    
    # Vérifier Firebase
    try:
        from utils.firebase import is_firebase_enabled
        if not is_firebase_enabled():
            print("❌ Firebase n'est pas configuré")
            print("Suivez les instructions dans FIREBASE_SETUP_GUIDE.md")
            return
    except ImportError:
        print("❌ Impossible d'importer utils.firebase")
        return
    
    print("1. Migration des utilisateurs...")
    users_migrated = migrate_users()
    print(f"   {users_migrated} utilisateurs migrés")
    
    print("\n2. Migration des prédictions...")
    preds_migrated = migrate_predictions()
    print(f"   {preds_migrated} prédictions migrées")
    
    print("\n=== Migration terminée ===")
    print(f"Total: {users_migrated} utilisateurs, {preds_migrated} prédictions")
    print("\n⚠️  IMPORTANT:")
    print("- Les fichiers locaux (users.json, prediction_history.csv) n'ont PAS été supprimés")
    print("- Vous pouvez les supprimer manuellement après vérification")
    print("- Testez l'application pour vérifier que tout fonctionne")

if __name__ == "__main__":
    main()