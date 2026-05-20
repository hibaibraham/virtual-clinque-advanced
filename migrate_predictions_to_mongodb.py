"""
Script de migration des prédictions CSV vers MongoDB
"""
import os
import pandas as pd
from datetime import datetime
from utils.database import get_predictions_collection, is_mongodb_available

HISTORY_PATH = 'historique_predictions.csv'

def migrate_predictions():
    """Migre les prédictions du CSV vers MongoDB."""
    
    print("=" * 60)
    print("MIGRATION DES PREDICTIONS VERS MONGODB")
    print("=" * 60)
    
    # Vérifier MongoDB
    print("\n1️⃣ Vérification de MongoDB...")
    if not is_mongodb_available():
        print("❌ MongoDB non disponible. Migration impossible.")
        return
    
    collection = get_predictions_collection()
    if collection is None:
        print("❌ Collection 'predictions' non accessible.")
        return
    
    print("✅ MongoDB opérationnel")
    
    # Vérifier le fichier CSV
    print("\n2️⃣ Vérification du fichier CSV...")
    if not os.path.exists(HISTORY_PATH):
        print(f"⚠️ Fichier {HISTORY_PATH} non trouvé.")
        print("   Aucune prédiction à migrer.")
        return
    
    # Charger les prédictions CSV
    try:
        df = pd.read_csv(HISTORY_PATH)
        print(f"✅ {len(df)} prédiction(s) trouvée(s) dans le CSV")
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du CSV: {e}")
        return
    
    if len(df) == 0:
        print("⚠️ Le fichier CSV est vide.")
        return
    
    # Compter les prédictions existantes dans MongoDB
    count_before = collection.count_documents({})
    print(f"📊 Prédictions déjà dans MongoDB: {count_before}")
    
    # Migrer chaque prédiction
    print("\n3️⃣ Migration en cours...")
    migrated = 0
    errors = 0
    
    for idx, row in df.iterrows():
        try:
            # Extraire les données
            timestamp_str = row.get('timestamp', '')
            prediction_label = row.get('prediction', '')
            probability_str = row.get('probability', '0%')
            
            # Convertir la probabilité
            try:
                probability = float(probability_str.replace('%', '')) / 100
            except:
                probability = 0.0
            
            # Déterminer la valeur de prédiction
            prediction_value = 1 if prediction_label == 'Pathologique' else 0
            
            # Créer l'enregistrement MongoDB
            mongo_record = {
                'prediction_id': f"PRED{datetime.now().strftime('%Y%m%d%H%M%S')}{idx:04d}",
                'timestamp_iso': timestamp_str,
                'prediction_label': prediction_label,
                'prediction_value': prediction_value,
                'probability': probability,
                'probability_percent': probability_str,
                'patient_data': {
                    k: v for k, v in row.items() 
                    if k not in ['timestamp', 'prediction', 'probability']
                },
                'created_at': datetime.now().isoformat(),
                'migrated_from_csv': True
            }
            
            # Insérer dans MongoDB
            collection.insert_one(mongo_record)
            migrated += 1
            
            if (idx + 1) % 10 == 0:
                print(f"   Migré: {idx + 1}/{len(df)}")
                
        except Exception as e:
            print(f"   ⚠️ Erreur ligne {idx + 1}: {e}")
            errors += 1
    
    # Résumé
    print("\n4️⃣ Résumé de la migration:")
    count_after = collection.count_documents({})
    print(f"   ✅ Prédictions migrées: {migrated}")
    if errors > 0:
        print(f"   ⚠️ Erreurs: {errors}")
    print(f"   📊 Total dans MongoDB: {count_after}")
    print(f"   📈 Nouvelles entrées: +{count_after - count_before}")
    
    print("\n" + "=" * 60)
    print("✅ MIGRATION TERMINÉE")
    print("=" * 60)
    print("\n💡 Vérifiez dans MongoDB Compass:")
    print("   Base: clinique_virtuelle")
    print("   Collection: predictions")
    print("=" * 60)


if __name__ == "__main__":
    migrate_predictions()
