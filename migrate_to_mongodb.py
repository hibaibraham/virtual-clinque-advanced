"""
Script de Migration des Données vers MongoDB
Migre les données de JSON/CSV vers MongoDB
"""
import json
import os
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError

# Configuration
MONGODB_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "clinique_virtuelle"

# Chemins des fichiers
USERS_JSON = "users.json"
PATIENTS_JSON = "patients.json"

def test_connection():
    """Teste la connexion à MongoDB."""
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("✅ Connexion à MongoDB réussie!")
        return client
    except ConnectionFailure as e:
        print(f"❌ Erreur de connexion à MongoDB: {e}")
        print("\n⚠️  Assurez-vous que MongoDB est démarré:")
        print("   1. Exécutez 'start_mongodb.bat' en tant qu'administrateur")
        print("   2. Ou démarrez le service MongoDB manuellement")
        return None

def migrate_users(db):
    """Migre les utilisateurs de JSON vers MongoDB."""
    print("\n📊 Migration des utilisateurs...")
    
    if not os.path.exists(USERS_JSON):
        print(f"⚠️  Fichier {USERS_JSON} introuvable. Aucun utilisateur à migrer.")
        return 0
    
    with open(USERS_JSON, 'r', encoding='utf-8') as f:
        users_data = json.load(f)
    
    if not users_data:
        print("ℹ️  Aucun utilisateur à migrer.")
        return 0
    
    collection = db.users
    migrated = 0
    skipped = 0
    
    for username, user_info in users_data.items():
        try:
            # Ajouter le username dans le document
            user_doc = {
                "username": username,
                **user_info,
                "migrated_at": datetime.now().isoformat()
            }
            
            # Insérer ou mettre à jour
            collection.update_one(
                {"username": username},
                {"$set": user_doc},
                upsert=True
            )
            migrated += 1
            print(f"  ✅ {username} migré")
            
        except Exception as e:
            print(f"  ❌ Erreur pour {username}: {e}")
            skipped += 1
    
    print(f"\n✅ Utilisateurs migrés: {migrated}")
    if skipped > 0:
        print(f"⚠️  Utilisateurs ignorés: {skipped}")
    
    return migrated

def migrate_patients(db):
    """Migre les patients de JSON vers MongoDB."""
    print("\n📊 Migration des patients...")
    
    if not os.path.exists(PATIENTS_JSON):
        print(f"⚠️  Fichier {PATIENTS_JSON} introuvable. Aucun patient à migrer.")
        return 0
    
    with open(PATIENTS_JSON, 'r', encoding='utf-8') as f:
        patients_data = json.load(f)
    
    if not patients_data:
        print("ℹ️  Aucun patient à migrer.")
        return 0
    
    collection = db.patients
    migrated = 0
    skipped = 0
    
    for patient_id, patient_info in patients_data.items():
        try:
            # Ajouter le patient_id dans le document si absent
            if "patient_id" not in patient_info:
                patient_info["patient_id"] = patient_id
            
            patient_info["migrated_at"] = datetime.now().isoformat()
            
            # Insérer ou mettre à jour
            collection.update_one(
                {"patient_id": patient_id},
                {"$set": patient_info},
                upsert=True
            )
            migrated += 1
            print(f"  ✅ {patient_id} - {patient_info.get('nom', '')} {patient_info.get('prenom', '')} migré")
            
        except Exception as e:
            print(f"  ❌ Erreur pour {patient_id}: {e}")
            skipped += 1
    
    print(f"\n✅ Patients migrés: {migrated}")
    if skipped > 0:
        print(f"⚠️  Patients ignorés: {skipped}")
    
    return migrated

def create_indexes(db):
    """Crée les index pour optimiser les recherches."""
    print("\n📊 Création des index...")
    
    try:
        # Index pour les patients
        db.patients.create_index("patient_id", unique=True)
        db.patients.create_index("nom")
        db.patients.create_index("prenom")
        db.patients.create_index("telephone")
        db.patients.create_index("status")
        db.patients.create_index("created_at")
        print("  ✅ Index patients créés")
        
        # Index pour les utilisateurs
        db.users.create_index("username", unique=True)
        db.users.create_index("role")
        print("  ✅ Index utilisateurs créés")
        
        # Index pour les rendez-vous
        db.appointments.create_index("patient_id")
        db.appointments.create_index("date")
        db.appointments.create_index("status")
        print("  ✅ Index rendez-vous créés")
        
        # Index pour les prédictions
        db.predictions.create_index("timestamp")
        db.predictions.create_index("username")
        print("  ✅ Index prédictions créés")
        
        print("\n✅ Tous les index ont été créés avec succès!")
        
    except Exception as e:
        print(f"⚠️  Erreur lors de la création des index: {e}")

def show_statistics(db):
    """Affiche les statistiques de la base de données."""
    print("\n" + "="*60)
    print("📊 STATISTIQUES DE LA BASE DE DONNÉES")
    print("="*60)
    
    # Utilisateurs
    users_count = db.users.count_documents({})
    print(f"\n👥 Utilisateurs: {users_count}")
    
    if users_count > 0:
        roles = db.users.aggregate([
            {"$group": {"_id": "$role", "count": {"$sum": 1}}}
        ])
        for role in roles:
            print(f"   - {role['_id']}: {role['count']}")
    
    # Patients
    patients_count = db.patients.count_documents({})
    print(f"\n🏥 Patients: {patients_count}")
    
    if patients_count > 0:
        statuses = db.patients.aggregate([
            {"$group": {"_id": "$status", "count": {"$sum": 1}}}
        ])
        for status in statuses:
            print(f"   - {status['_id']}: {status['count']}")
    
    # Rendez-vous
    appointments_count = db.appointments.count_documents({})
    print(f"\n📅 Rendez-vous: {appointments_count}")
    
    # Prédictions
    predictions_count = db.predictions.count_documents({})
    print(f"\n🔬 Prédictions: {predictions_count}")
    
    print("\n" + "="*60)

def backup_json_files():
    """Crée une sauvegarde des fichiers JSON avant migration."""
    print("\n💾 Création de sauvegardes...")
    
    backup_dir = "backup_before_mongodb"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    files_to_backup = [USERS_JSON, PATIENTS_JSON]
    backed_up = 0
    
    for file in files_to_backup:
        if os.path.exists(file):
            backup_path = os.path.join(backup_dir, file)
            with open(file, 'r', encoding='utf-8') as src:
                with open(backup_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            print(f"  ✅ {file} sauvegardé")
            backed_up += 1
    
    if backed_up > 0:
        print(f"\n✅ {backed_up} fichier(s) sauvegardé(s) dans '{backup_dir}/'")
    else:
        print("ℹ️  Aucun fichier à sauvegarder")

def main():
    """Fonction principale de migration."""
    print("="*60)
    print("  🔄 MIGRATION DES DONNÉES VERS MONGODB")
    print("  NovaClinic v4.1")
    print("="*60)
    
    # Tester la connexion
    client = test_connection()
    if not client:
        print("\n❌ Migration annulée. Veuillez démarrer MongoDB d'abord.")
        return
    
    # Sélectionner la base de données
    db = client[DATABASE_NAME]
    print(f"✅ Base de données: {DATABASE_NAME}")
    
    # Demander confirmation
    print("\n⚠️  Cette opération va migrer vos données vers MongoDB.")
    print("   Les fichiers JSON seront sauvegardés avant la migration.")
    response = input("\n❓ Voulez-vous continuer? (oui/non): ").lower().strip()
    
    if response not in ['oui', 'o', 'yes', 'y']:
        print("\n❌ Migration annulée par l'utilisateur.")
        return
    
    # Créer des sauvegardes
    backup_json_files()
    
    # Migrer les données
    users_migrated = migrate_users(db)
    patients_migrated = migrate_patients(db)
    
    # Créer les index
    create_indexes(db)
    
    # Afficher les statistiques
    show_statistics(db)
    
    # Résumé
    print("\n" + "="*60)
    print("✅ MIGRATION TERMINÉE AVEC SUCCÈS!")
    print("="*60)
    print(f"\n📊 Résumé:")
    print(f"   - Utilisateurs migrés: {users_migrated}")
    print(f"   - Patients migrés: {patients_migrated}")
    print(f"   - Base de données: {DATABASE_NAME}")
    print(f"   - URI: {MONGODB_URI}")
    
    print("\n💡 Prochaines étapes:")
    print("   1. Vérifiez les données dans MongoDB Compass")
    print("   2. Relancez l'application: streamlit run app.py")
    print("   3. L'application utilisera automatiquement MongoDB")
    
    print("\n⚠️  Note: Les fichiers JSON restent disponibles comme fallback")
    print("   si MongoDB n'est pas accessible.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Migration interrompue par l'utilisateur.")
    except Exception as e:
        print(f"\n\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("\nAppuyez sur Entrée pour quitter...")
