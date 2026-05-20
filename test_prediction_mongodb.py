"""
Script de test pour vérifier la sauvegarde des prédictions dans MongoDB
"""
from utils.core import save_prediction
from utils.database import get_predictions_collection, is_mongodb_available

def test_prediction_save():
    """Test de sauvegarde d'une prédiction."""
    
    print("=" * 60)
    print("TEST SAUVEGARDE PREDICTION DANS MONGODB")
    print("=" * 60)
    
    # Vérifier MongoDB
    print("\n1️⃣ Vérification de MongoDB...")
    if is_mongodb_available():
        print("✅ MongoDB disponible")
        collection = get_predictions_collection()
        if collection is not None:
            print(f"✅ Collection 'predictions' accessible")
            
            # Compter les prédictions existantes
            count_before = collection.count_documents({})
            print(f"📊 Nombre de prédictions avant test: {count_before}")
        else:
            print("❌ Collection 'predictions' non accessible")
            return
    else:
        print("❌ MongoDB non disponible")
        return
    
    # Créer une prédiction de test
    print("\n2️⃣ Création d'une prédiction de test...")
    
    patient_data = {
        'age': 45,
        'sex': 'M',
        'TSH': 2.5,
        'T3': 1.2,
        'TT4': 100,
        'T4U': 0.9,
        'FTI': 110,
        'on_thyroxine': 'f',
        'query_on_thyroxine': 'f',
        'on_antithyroid_medication': 'f',
        'sick': 'f',
        'pregnant': 'f',
        'thyroid_surgery': 'f',
        'I131_treatment': 'f',
        'query_hypothyroid': 'f',
        'query_hyperthyroid': 'f',
        'lithium': 'f',
        'goitre': 'f',
        'tumor': 'f',
        'hypopituitary': 'f',
        'psych': 'f'
    }
    
    prediction = 0  # Normal
    probability = 0.85
    
    try:
        record = save_prediction(patient_data, prediction, probability)
        print(f"✅ Prédiction sauvegardée:")
        print(f"   Timestamp: {record['timestamp']}")
        print(f"   Prédiction: {record['prediction']}")
        print(f"   Probabilité: {record['probability']}")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde: {e}")
        return
    
    # Vérifier dans MongoDB
    print("\n3️⃣ Vérification dans MongoDB...")
    
    try:
        count_after = collection.count_documents({})
        print(f"📊 Nombre de prédictions après test: {count_after}")
        
        if count_after > count_before:
            print(f"✅ Nouvelle prédiction ajoutée! (+{count_after - count_before})")
            
            # Récupérer la dernière prédiction
            last_prediction = collection.find_one(sort=[('created_at', -1)])
            if last_prediction:
                print(f"\n📋 Dernière prédiction dans MongoDB:")
                print(f"   ID: {last_prediction.get('prediction_id', 'N/A')}")
                print(f"   Label: {last_prediction.get('prediction_label', 'N/A')}")
                print(f"   Probabilité: {last_prediction.get('probability_percent', 'N/A')}")
                print(f"   Date: {last_prediction.get('timestamp_iso', 'N/A')}")
        else:
            print("⚠️ Aucune nouvelle prédiction détectée dans MongoDB")
            print("   La prédiction a peut-être été sauvegardée uniquement en CSV")
    except Exception as e:
        print(f"❌ Erreur lors de la vérification: {e}")
    
    print("\n" + "=" * 60)
    print("✅ TEST TERMINÉ")
    print("=" * 60)
    print("\n💡 Pour voir les prédictions dans MongoDB Compass:")
    print("   1. Ouvrir MongoDB Compass")
    print("   2. Se connecter à: mongodb://localhost:27017")
    print("   3. Base de données: clinique_virtuelle")
    print("   4. Collection: predictions")
    print("=" * 60)


if __name__ == "__main__":
    test_prediction_save()
