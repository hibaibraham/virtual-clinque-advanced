"""
Gestionnaire des modèles ML/DL
"""
import streamlit as st
from typing import Dict, Any, Optional
from .thyroid_model import ThyroidModel
from .ptdm_model import PtdmModel

# Import conditionnel du modèle brain cancer (nécessite TensorFlow)
try:
    from .brain_cancer_model import BrainCancerModel
    BRAIN_CANCER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Brain Cancer model non disponible (TensorFlow requis): {e}")
    BRAIN_CANCER_AVAILABLE = False
    BrainCancerModel = None

class ModelManager:
    """Gère le chargement et l'utilisation des modèles"""
    
    def __init__(self):
        self.thyroid_model = ThyroidModel()
        self.brain_cancer_model = BrainCancerModel() if BRAIN_CANCER_AVAILABLE else None
        self.ptdm_model = PtdmModel()
        self.models_loaded = False
        
    def load_all_models(self):
        """Charger tous les modèles"""
        with st.spinner("Chargement des modèles..."):
            thyroid_loaded = self.thyroid_model.load()
            brain_loaded = self.brain_cancer_model.load() if BRAIN_CANCER_AVAILABLE else False
            ptdm_loaded = self.ptdm_model.load()
            
            # Considérer comme chargé si thyroid et ptdm sont OK
            self.models_loaded = thyroid_loaded and ptdm_loaded
            
            if self.models_loaded:
                st.success("✅ Modèles chargés avec succès!")
                if not BRAIN_CANCER_AVAILABLE:
                    st.info("ℹ️ Modèle Brain Cancer non disponible (TensorFlow non installé)")
            else:
                st.warning("⚠️ Certains modèles n'ont pas pu être chargés")
                
        return self.models_loaded
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Obtenir la liste des modèles disponibles"""
        models = {
            'thyroid': {
                'name': 'Diagnostic Thyroïdien',
                'description': 'Analyse des marqueurs hormonaux thyroïdiens',
                'type': 'Machine Learning',
                'algorithm': 'Random Forest',
                'input_type': 'formulaire',
                'icon': '🦋',
                'loaded': self.thyroid_model.loaded
            },
            'ptdm': {
                'name': 'Prédiction Risque PTDM',
                'description': 'Prédiction du diabète post-transplantation',
                'type': 'Machine Learning',
                'algorithm': 'Random Forest / SVM',
                'input_type': 'formulaire',
                'icon': '🩸',
                'loaded': self.ptdm_model.loaded
            }
        }
        
        # Ajouter brain cancer seulement si disponible
        if BRAIN_CANCER_AVAILABLE and self.brain_cancer_model:
            models['brain_cancer'] = {
                'name': 'Détection Cancer Cérébral',
                'description': 'Analyse d\'images MRI du cerveau',
                'type': 'Deep Learning',
                'algorithm': 'CNN (ResNet)',
                'input_type': 'image',
                'icon': '🧠',
                'loaded': self.brain_cancer_model.loaded
            }
        
        return models
    
    def get_model(self, model_type: str):
        """Obtenir un modèle spécifique"""
        if model_type == 'thyroid':
            return self.thyroid_model
        elif model_type == 'brain_cancer':
            if not BRAIN_CANCER_AVAILABLE:
                raise ValueError("Modèle Brain Cancer non disponible (TensorFlow requis)")
            return self.brain_cancer_model
        elif model_type == 'ptdm':
            return self.ptdm_model
        else:
            raise ValueError(f"Modèle inconnu: {model_type}")
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Obtenir les statistiques des modèles"""
        stats = {
            'total_models': 2 if not BRAIN_CANCER_AVAILABLE else 3,
            'loaded_models': 0,
            'models': {}
        }
        
        # Modèle thyroïde
        if self.thyroid_model.loaded:
            stats['loaded_models'] += 1
            thyroid_info = self.thyroid_model.get_model_info()
            stats['models']['thyroid'] = {
                'name': 'Thyroid Model',
                'accuracy': f"{thyroid_info.get('accuracy', 0):.1%}",
                'features': thyroid_info.get('features', 0),
                'status': '✅ Chargé'
            }
        else:
            stats['models']['thyroid'] = {
                'name': 'Thyroid Model',
                'status': '❌ Non chargé'
            }
        
        # Modèle brain cancer (seulement si disponible)
        if BRAIN_CANCER_AVAILABLE and self.brain_cancer_model:
            if self.brain_cancer_model.loaded:
                stats['loaded_models'] += 1
                brain_info = self.brain_cancer_model.get_model_info()
                stats['models']['brain_cancer'] = {
                    'name': brain_info.get('name', 'Brain Cancer Model'),
                    'type': brain_info.get('type', 'CNN'),
                    'classes': brain_info.get('classes', 0),
                    'status': '✅ Chargé'
                }
            else:
                stats['models']['brain_cancer'] = {
                    'name': 'Brain Cancer Model',
                    'status': '❌ Non chargé'
                }
        else:
            stats['models']['brain_cancer'] = {
                'name': 'Brain Cancer Model',
                'status': '⚠️ Non disponible (TensorFlow requis)'
            }
            
        # Modèle PTDM
        if self.ptdm_model.loaded:
            stats['loaded_models'] += 1
            ptdm_info = self.ptdm_model.get_model_info()
            stats['models']['ptdm'] = {
                'name': ptdm_info.get('name', 'PTDM Model'),
                'accuracy': f"{ptdm_info.get('accuracy', 0):.1%}",
                'features': ptdm_info.get('features', 0),
                'status': '✅ Chargé'
            }
        else:
            stats['models']['ptdm'] = {
                'name': 'PTDM Model',
                'status': '❌ Non chargé'
            }
        
        return stats
    
    @st.cache_resource
    def get_cached_manager():
        """Obtenir une instance mise en cache du ModelManager"""
        return ModelManager()