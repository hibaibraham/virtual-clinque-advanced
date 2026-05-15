"""
Modèle de diagnostic thyroïdien (Random Forest)
"""
import os
import joblib
import json
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any

class ThyroidModel:
    """Modèle Random Forest pour le diagnostic thyroïdien"""
    
    def __init__(self, model_dir: str = "saved_models"):
        self.model_dir = model_dir
        self.model = None
        self.preprocessor = None
        self.config = None
        self.loaded = False
        
    def load(self):
        """Charger le modèle depuis le disque"""
        try:
            model_path = os.path.join(self.model_dir, 'model.joblib')
            preprocessor_path = os.path.join(self.model_dir, 'preprocessor.joblib')
            config_path = os.path.join(self.model_dir, 'feature_config.json')
            
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
            
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            self.loaded = True
            return True
        except Exception as e:
            print(f"Erreur chargement modèle thyroïde: {e}")
            return False
    
    def predict(self, patient_data: Dict[str, Any]) -> Tuple[int, float, Dict[str, Any]]:
        """Faire une prédiction"""
        if not self.loaded:
            raise ValueError("Modèle non chargé")
        
        # Feature engineering (identique à l'ancien code)
        patient_data = self._compute_engineered_features(patient_data)
        
        # Préparer les features
        all_features = self.config['all_features']
        X_input = pd.DataFrame([[patient_data.get(f, 0) for f in all_features]], 
                              columns=all_features)
        
        # Prétraitement
        X_proc = self.preprocessor.transform(X_input)
        
        # Prédiction
        prediction = self.model.predict(X_proc)[0]
        proba = self.model.predict_proba(X_proc)[0]
        prob_patho = proba[1]
        
        return prediction, prob_patho, patient_data
    
    def _compute_engineered_features(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Calculer les features supplémentaires"""
        row['TSH_abnormal'] = int((row.get('TSH', 0) < 0.4) or (row.get('TSH', 0) > 4.0))
        row['TT4_abnormal'] = int((row.get('TT4', 0) < 70) or (row.get('TT4', 0) > 180))
        row['T3_abnormal'] = int((row.get('T3', 0) < 1.2) or (row.get('T3', 0) > 3.1))
        row['FTI_abnormal'] = int((row.get('FTI', 0) < 70) or (row.get('FTI', 0) > 180))
        row['hormone_score'] = row['TSH_abnormal'] + row['TT4_abnormal'] + row['T3_abnormal'] + row['FTI_abnormal']
        row['T4U_TT4_ratio'] = row.get('T4U', 0) / (row.get('TT4', 0) + 1e-6)
        return row
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtenir les informations du modèle"""
        if not self.loaded:
            return {}
        
        return {
            'name': 'Random Forest Thyroid',
            'accuracy': self.config.get('test_accuracy', 0),
            'f1_score': self.config.get('test_f1', 0),
            'features': len(self.config.get('all_features', [])),
            'best_params': self.config.get('best_params', {}),
            'feature_importances': self.config.get('feature_importances', {})
        }
    
    def get_normal_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Obtenir les plages normales des marqueurs"""
        return {
            'TSH': (0.4, 4.0),
            'T3': (1.2, 3.1),
            'TT4': (70, 180),
            'FTI': (70, 180),
            'T4U': (0.7, 1.3)
        }