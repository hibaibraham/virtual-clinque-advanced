"""
Script de vérification de la configuration MongoDB
Exécuter : python check_mongodb.py
"""
import sys

def check_pymongo():
    """Vérifie que pymongo est installé."""
    try:
        import pymongo
        print(f"✅ pymongo installé (version {pymongo.__version__})")
        return True
    except ImportError:
        print("❌ pymongo n'est pas installé")
        print("   Installation: pip install pymongo")
        return False

def check_connection():
    """Vérifie la connexion à MongoDB."""
    try:
        from utils.database import is_mongodb_available, get_database
        
        if is_mongodb_available():
            print("✅ Connexion MongoDB établie")
            db = get_database()
            print(f"   Base de données: {db.name}")
            return True
        else:
            print("❌ MongoDB n'est pas disponible")
            print("   Vérifiez que MongoDB est démarré")
            print("   Ou configurez MONGODB_URI dans utils/database.py")
            return False
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")
        return False

def check_collections():
    """Vérifie les collections MongoDB."""
    try:
        from utils.database import get_database
        
        db = get_database()
        if db is None:
            return False
        
        collections = db.list_collection_names()
        print(f"\n📊 Collections disponibles: {len(collections)}")
        
        for coll_name in ['users', 'patients', 'appointments', 'predictions', 'consultations']:
            if coll_name in collections:
                count = db[coll_name].count_documents({})
                print(f"   ✅ {coll_name}: {count} documents")
            else:
                print(f"   ⚠️  {coll_name}: collection vide ou inexistante")
        
        return True
    except Exception as e:
        print(f"❌ Erreur lors de la vérification des collections: {e}")
        return False

def check_data_files():
    """Vérifie les fichiers de données JSON/CSV."""
    import os
    
    print("\n📁 Fichiers de données locaux:")
    
    files = {
        'users.json': 'Utilisateurs',
        'patients.json': 'Patients',
        'predictions.csv': 'Prédictions'
    }
    
    for filename, description in files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"   ✅ {filename} ({description}): {size} bytes")
        else:
            print(f"   ⚠️  {filename} ({description}): fichier introuvable")

def main():
    """Fonction principale."""
    print("=" * 60)
    print("🔍 VÉRIFICATION CONFIGURATION MONGODB")
    print("=" * 60)
    print()
    
    # Vérifier pymongo
    if not check_pymongo():
        print("\n❌ Installation requise: pip install pymongo")
        sys.exit(1)
    
    print()
    
    # Vérifier la connexion
    mongodb_ok = check_connection()
    
    print()
    
    # Vérifier les fichiers locaux
    check_data_files()
    
    if mongodb_ok:
        print()
        # Vérifier les collections
        check_collections()
        
        print("\n" + "=" * 60)
        print("✅ CONFIGURATION OK")
        print("=" * 60)
        print("\n💡 Prochaines étapes:")
        print("   1. Si MongoDB est vide, lancez: python migrate_to_mongodb.py")
        print("   2. Lancez l'application: streamlit run app.py")
        print("   3. Consultez MONGODB_SETUP.md pour plus d'informations")
    else:
        print("\n" + "=" * 60)
        print("⚠️  MONGODB NON DISPONIBLE")
        print("=" * 60)
        print("\n💡 L'application fonctionnera en mode JSON/CSV (fallback)")
        print("\n📋 Pour activer MongoDB:")
        print("   1. Installez MongoDB: https://www.mongodb.com/try/download/community")
        print("   2. Démarrez MongoDB: net start MongoDB (Windows)")
        print("   3. Ou utilisez MongoDB Atlas (cloud)")
        print("   4. Consultez MONGODB_SETUP.md pour le guide complet")
        print("\n   Lancez quand même l'application: streamlit run app.py")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
