"""
Modèle Deep Learning pour le diagnostic des maladies oculaires
"""
import os
import numpy as np
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
from typing import Dict, Tuple, Any, Optional
from PIL import Image
import io

class EyeDiseaseModel:
    """Modèle CNN pour la classification d'images oculaires"""
    
    def __init__(self, model_dir: str = "saved_models/eye_disease"):
        self.model_dir = model_dir
        self.model = None
        self.class_names = ['Bulging_Eyes', 'Cataracts', 'Crossed_Eyes', 'Glaucoma', 'Uveitis']
        self.class_names_fr = {
            'Bulging_Eyes': 'Yeux Exorbités',
            'Cataracts': 'Cataracte',
            'Crossed_Eyes': 'Strabisme',
            'Glaucoma': 'Glaucome',
            'Uveitis': 'Uvéite'
        }
        self.loaded = False
        
    def load(self):
        """Charger le modèle depuis le disque"""
        try:
            if not TF_AVAILABLE:
                raise ImportError("Tensorflow n'est pas installé.")
                
            model_path = os.path.join(self.model_dir, 'eye_disease_model.h5')
            
            # Vérifier si le modèle existe
            if os.path.exists(model_path):
                # Charger le modèle Keras
                self.model = keras.models.load_model(model_path)
                
                # Vérifier que le modèle est chargé
                self.model.predict(np.zeros((1, 224, 224, 3)))  # Test prediction
                
                self.loaded = True
                return True
            else:
                # Créer un modèle factice pour le développement
                self._create_dummy_model()
                return True
        except Exception as e:
            print(f"Erreur chargement modèle eye disease: {e}")
            # Créer un modèle factice pour le développement
            self._create_dummy_model()
            return True  # Retourner True pour le développement
    
    def _create_dummy_model(self):
        """Créer un modèle factice pour le développement"""
        print("⚠️  Création d'un modèle factice pour le développement (Eye Disease)")
        if TF_AVAILABLE:
            self.model = keras.Sequential([
                keras.layers.Input(shape=(224, 224, 3)),
                keras.layers.Flatten(),
                keras.layers.Dense(len(self.class_names), activation='softmax')
            ])
            self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        else:
            self.model = "dummy_model_no_tf"
        self.loaded = True
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Prétraiter une image oculaire"""
        # Charger l'image depuis les bytes
        image = Image.open(io.BytesIO(image_data))
        
        # Convertir en RGB si nécessaire
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Redimensionner à 224x224 (taille standard pour les modèles CNN)
        image = image.resize((224, 224))
        
        # Convertir en numpy array et normaliser
        img_array = np.array(image) / 255.0
        
        # Ajouter une dimension de batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_data: bytes) -> Tuple[str, float, Dict[str, float]]:
        """Faire une prédiction sur une image oculaire"""
        if not self.loaded:
            raise ValueError("Modèle non chargé")
        
        # Prétraiter l'image
        processed_image = self.preprocess_image(image_data)
        
        # Faire la prédiction
        if TF_AVAILABLE and hasattr(self.model, 'predict'):
            predictions = self.model.predict(processed_image, verbose=0)[0]
        else:
            # Fallback aléatoire si TF non disponible
            predictions = np.random.dirichlet(np.ones(len(self.class_names)), size=1)[0]
        
        # Obtenir la classe prédite
        predicted_class_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx])
        
        # Créer un dictionnaire de confiances pour toutes les classes
        confidences = {
            self.class_names_fr[cls]: float(pred) 
            for cls, pred in zip(self.class_names, predictions)
        }
        
        return predicted_class, confidence, confidences
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtenir les informations du modèle"""
        if not self.loaded:
            return {}
        
        # Pour un modèle factice, retourner des informations par défaut
        if not TF_AVAILABLE or isinstance(self.model, str) or len(self.model.layers) < 3:  # Modèle factice
            return {
                'name': 'CNN Eye Disease (Développement)',
                'type': 'Convolutional Neural Network',
                'input_shape': (224, 224, 3),
                'classes': len(self.class_names),
                'class_names': self.class_names_fr,
                'status': 'development'
            }
        
        # Pour un vrai modèle
        return {
            'name': 'CNN Eye Disease',
            'type': 'Convolutional Neural Network',
            'input_shape': self.model.input_shape[1:],
            'output_shape': self.model.output_shape[1:],
            'layers': len(self.model.layers),
            'classes': len(self.class_names),
            'class_names': self.class_names_fr,
            'status': 'production'
        }
    
    def get_class_description(self, class_name: str) -> Dict[str, str]:
        """Obtenir la description d'une classe"""
        descriptions = {
            'Bulging_Eyes': {
                'name': 'Yeux Exorbités (Exophtalmie)',
                'description': 'Protrusion anormale des yeux hors de leurs orbites',
                'prevalence': 'Souvent associée à l\'hyperthyroïdie',
                'treatment': 'Traitement de la cause sous-jacente, chirurgie si nécessaire'
            },
            'Cataracts': {
                'name': 'Cataracte',
                'description': 'Opacification du cristallin de l\'œil',
                'prevalence': 'Très fréquente après 60 ans',
                'treatment': 'Chirurgie de remplacement du cristallin'
            },
            'Crossed_Eyes': {
                'name': 'Strabisme',
                'description': 'Défaut d\'alignement des yeux',
                'prevalence': '2-4% de la population',
                'treatment': 'Orthoptie, lunettes, chirurgie'
            },
            'Glaucoma': {
                'name': 'Glaucome',
                'description': 'Maladie du nerf optique souvent liée à la pression intraoculaire',
                'prevalence': '2% de la population après 40 ans',
                'treatment': 'Collyres, laser, chirurgie'
            },
            'Uveitis': {
                'name': 'Uvéite',
                'description': 'Inflammation de l\'uvée (iris, corps ciliaire, choroïde)',
                'prevalence': 'Relativement rare',
                'treatment': 'Anti-inflammatoires, immunosuppresseurs'
            }
        }
        
        return descriptions.get(class_name, {
            'name': 'Inconnu',
            'description': 'Classe non reconnue',
            'prevalence': 'N/A',
            'treatment': 'N/A'
        })
    
    def validate_image(self, image_data: bytes) -> Tuple[bool, str]:
        """Valider qu'une image est appropriée pour l'analyse"""
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Vérifier la taille minimale
            if image.size[0] < 100 or image.size[1] < 100:
                return False, "Image trop petite (min 100x100 pixels)"
            
            # Vérifier le format
            if image.format not in ['JPEG', 'PNG', 'BMP', 'TIFF']:
                return False, "Format d'image non supporté (JPEG, PNG, BMP, TIFF seulement)"
            
            return True, "Image valide"
        except Exception as e:
            return False, f"Erreur lecture image: {str(e)}"