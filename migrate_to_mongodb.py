"""
Script de migration des données JSON/CSV vers MongoDB
Exécuter : python migrate_to_mongodb.py
"""
import json
import os
import pandas as pd
from utils.database import (
    get_database, 
    get_users_collection, 
    get_patients_collection,
    get_predictions_collection,
    create_indexes,
    is_mongodb_available
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_PATH = os.path.join(BASE_DIR, 'users.json')
PATIENTS_PATH = os.path.join(BASE_DIR, 'patients.json')
PREDICTIONS_CSV_PATH = os.path.join(BASE_DIR, 'predictions.csv')

def migrate_users():
    """Migre les utilisateurs de users.json vers MongoDB."""
    print("\n📤 Migration des utilisateurs...")
    
    if not os.path.exists(USERS_PATH):
        print("⚠️ Fichier users.json introuvable")
        return 0
    
    with open(USERS_PATH, 'r', encoding='utf-8') as f:
        users_data = json.load(f)
    
    users_collection = get_users_collection()
    if users_collection is None:
        print("❌ Collection users non disponible")
        return 0
    
    # Supprimer les données existantes (optionnel)
    users_collection.delete_many({})
    
    migrated = 0
    for username, user_info in users_data.items():
        user_doc = {
            "username": username,
            "password": user_info.get("password"),
            "totp_secret": user_info.get("totp_secret"),
            "totp_verified": user_info.get("totp_verified", False),
            "role": user_info.get("role", "patient"),
            "created_at": user_info.get("created_at", None)
        }
        
        try:
            users_collection.insert_one(user_doc)
            migrated += 1
            print(f"  ✅ {username} ({user_info.get('role', 'patient')})")
        except Exception as e:
            print(f"  ❌ Erreur pour {username}: {e}")
    
    print(f"✅ {migrated}/{len(users_data)} utilisateurs migrés")
    return migrated

def migrate_patients():
    """Migre les patients de patients.json vers MongoDB."""
    print("\n📤 Migration des patients...")
    
    if not os.path.exists(PATIENTS_PATH):
        print("⚠️ Fichier patients.json introuvable")
        return 0
    
    with open(PATIENTS_PATH, 'r', encoding='utf-8') as f:
        patients_data = json.load(f)
    
    patients_collection = get_patients_collection()
    if patients_collection is None:
        print("❌ Collection patients non disponible")
        return 0
    
    # Supprimer les données existantes (optionnel)
    patients_collection.delete_many({})
    
    migrated = 0
    for patient_id, patient_info in patients_data.items():
        try:
            patients_collection.insert_one(patient_info)
            migrated += 1
            print(f"  ✅ {patient_id} - {patient_info.get('prenom', '')} {patient_info.get('nom', '')}")
        except Exception as e:
            print(f"  ❌ Erreur pour {patient_id}: {e}")
    
    print(f"✅ {migrated}/{len(patients_data)} patients migrés")
    return migrated

def migrate_predictions():
    """Migre les prédictions de predictions.csv vers MongoDB."""
    print("\n📤 Migration des prédictions...")
    
    if not os.path.exists(PREDICTIONS_CSV_PATH):
        print("⚠️ Fichier predictions.csv introuvable")
        return 0
    
    try:
        df = pd.read_csv(PREDICTIONS_CSV_PATH)
    except Exception as e:
        print(f"❌ Erreur lecture CSV: {e}")
        return 0
    
    predictions_collection = get_predictions_collection()
    if predictions_collection is None:
        print("❌ Collection predictions non disponible")
        return 0
    
    # Supprimer les données existantes (optionnel)
    predictions_collection.delete_many({})
    
    migrated = 0
    for _, row in df.iterrows():
        prediction_doc = row.to_dict()
        
        try:
            predictions_collection.insert_one(prediction_doc)
            migrated += 1
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
    
    print(f"✅ {migrated}/{len(df)} prédictions migrées")
    return migrated

def backup_files():
    """Crée des backups des fichiers JSON/CSV avant migration."""
    print("\n💾 Création des backups...")
    
    backup_dir = os.path.join(BASE_DIR, 'backup_before_mongodb')
    os.makedirs(backup_dir, exist_ok=True)
    
    files_to_backup = [
        USERS_PATH,
        PATIENTS_PATH,
        PREDICTIONS_CSV_PATH
    ]
    
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            backup_path = os.path.join(backup_dir, filename)
            
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    df = pd.read_csv(file_path)
                    df.to_csv(backup_path, index=False)
                
                print(f"  ✅ {filename} → backup/")
            except Exception as e:
                print(f"  ❌ Erreur backup {filename}: {e}")
    
    print(f"✅ Backups créés dans: {backup_dir}")

def verify_migration():
    """Vérifie que la migration s'est bien passée."""
    print("\n🔍 Vérification de la migration...")
    
    users_collection = get_users_collection()
    patients_collection = get_patients_collection()
    predictions_collection = get_predictions_collection()
    
    if users_collection is not None:
        users_count = users_collection.count_documents({})
        print(f"  👥 Utilisateurs dans MongoDB: {users_count}")
    
    if patients_collection is not None:
        patients_count = patients_collection.count_documents({})
        print(f"  🏥 Patients dans MongoDB: {patients_count}")
        
        # Compter par statut
        en_attente = patients_collection.count_documents({"status": "en_attente"})
        en_cours = patients_collection.count_documents({"status": "en_cours"})
        complete = patients_collection.count_documents({"status": "complete"})
        
        print(f"    ⏳ En attente: {en_attente}")
        print(f"    🔄 En cours: {en_cours}")
        print(f"    ✅ Complets: {complete}")
    
    if predictions_collection is not None:
        predictions_count = predictions_collection.count_documents({})
        print(f"  🔬 Prédictions dans MongoDB: {predictions_count}")

def main():
    """Fonction principale de migration."""
    print("=" * 60)
    print("🚀 MIGRATION VERS MONGODB")
    print("=" * 60)
    
    # Vérifier la connexion MongoDB
    if not is_mongodb_available():
        print("\n❌ MongoDB n'est pas disponible!")
        print("\n📋 Instructions:")
        print("  1. Installer MongoDB: https://www.mongodb.com/try/download/community")
        print("  2. Démarrer MongoDB: mongod")
        print("  3. Ou utiliser MongoDB Atlas (cloud)")
        print("  4. Configurer MONGODB_URI dans .env ou utils/database.py")
        return
    
    print("\n✅ Connexion MongoDB établie")
    
    # Demander confirmation
    print("\n⚠️  ATTENTION:")
    print("  - Cette opération va migrer toutes les données vers MongoDB")
    print("  - Les données MongoDB existantes seront écrasées")
    print("  - Un backup sera créé dans backup_before_mongodb/")
    
    response = input("\n❓ Continuer la migration? (oui/non): ").strip().lower()
    
    if response not in ['oui', 'o', 'yes', 'y']:
        print("\n❌ Migration annulée")
        return
    
    # Créer les backups
    backup_files()
    
    # Effectuer les migrations
    users_migrated = migrate_users()
    patients_migrated = migrate_patients()
    predictions_migrated = migrate_predictions()
    
    # Créer les index
    print("\n📊 Création des index...")
    create_indexes()
    
    # Vérifier la migration
    verify_migration()
    
    # Résumé
    print("\n" + "=" * 60)
    print("✅ MIGRATION TERMINÉE")
    print("=" * 60)
    print(f"  👥 Utilisateurs: {users_migrated}")
    print(f"  🏥 Patients: {patients_migrated}")
    print(f"  🔬 Prédictions: {predictions_migrated}")
    print("\n💡 Prochaines étapes:")
    print("  1. Vérifier que l'application fonctionne avec MongoDB")
    print("  2. Tester toutes les fonctionnalités")
    print("  3. Si tout fonctionne, vous pouvez supprimer les fichiers JSON/CSV")
    print("  4. Les backups sont dans: backup_before_mongodb/")
    print("=" * 60)

if __name__ == "__main__":
    main()
