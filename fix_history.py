"""
Script pour ajouter model_type aux anciennes prédictions
"""
import pandas as pd
import os

HISTORY_PATH = 'prediction_history.csv'

if os.path.exists(HISTORY_PATH):
    print("📂 Lecture du fichier historique...")
    df = pd.read_csv(HISTORY_PATH)
    
    print(f"✅ {len(df)} prédictions trouvées")
    
    # Ajouter model_type si manquant
    if 'model_type' not in df.columns:
        print("➕ Ajout de la colonne model_type...")
        df['model_type'] = 'thyroid'  # Par défaut, anciennes prédictions = thyroïde
        
    # Ajouter patient_name si manquant
    if 'patient_name' not in df.columns:
        print("➕ Ajout de la colonne patient_name...")
        df['patient_name'] = 'Anonyme'
    
    # Sauvegarder
    df.to_csv(HISTORY_PATH, index=False)
    print(f"💾 Fichier mis à jour avec {len(df)} prédictions")
    print(f"📊 Colonnes: {list(df.columns)}")
else:
    print("❌ Fichier prediction_history.csv introuvable")
