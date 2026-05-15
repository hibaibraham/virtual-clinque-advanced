"""
Modèle Deep Learning pour le diagnostic du cancer du cerveau
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Tuple, Any, Optional
import cv2
from PIL import Image
import io

class BrainCancerModel:
    """Modèle CNN pour la classification d'images MRI de cerveau"""
    
    def __init__(self, model_dir: str = "saved_models/brain_cancer"):
        self.model_dir = model_dir
        self.model = None
        self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.class_names_fr = {
            'glioma': 'Gliome',
            'meningioma': 'Méningiome', 
            'notumor': 'Pas de tumeur',
            'pituitary': 'Tumeur hypophysaire'
        }
        self.loaded = False
        
    def load(self):
        """Charger le modèle depuis le disque"""
        try:
            model_path = os.path.join(self.model_dir, 'brain_cancer_model.h5')
            
            # Charger le modèle Keras
            self.model = keras.models.load_model(model_path)
            
            # Vérifier que le modèle est chargé
            self.model.predict(np.zeros((1, 224, 224, 3)))  # Test prediction
            
            self.loaded = True
            return True
        except Exception as e:
            print(f"Erreur chargement modèle brain cancer: {e}")
            # Créer un modèle factice pour le développement
            self._create_dummy_model()
            return True  # Retourner True pour le développement
    
    def _create_dummy_model(self):
        """Créer un modèle factice pour le développement"""
        print("⚠️  Création d'un modèle factice pour le développement")
        self.model = keras.Sequential([
            keras.layers.Input(shape=(224, 224, 3)),
            keras.layers.Flatten(),
            keras.layers.Dense(4, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.loaded = True
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Prétraiter une image MRI"""
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
        """Faire une prédiction sur une image MRI"""
        if not self.loaded:
            raise ValueError("Modèle non chargé")
        
        # Prétraiter l'image
        processed_image = self.preprocess_image(image_data)
        
        # Faire la prédiction
        predictions = self.model.predict(processed_image, verbose=0)[0]
        
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
        if len(self.model.layers) < 3:  # Modèle factice
            return {
                'name': 'CNN Brain Cancer (Développement)',
                'type': 'Convolutional Neural Network',
                'input_shape': (224, 224, 3),
                'classes': 4,
                'class_names': self.class_names_fr,
                'status': 'development'
            }
        
        # Pour un vrai modèle
        return {
            'name': 'CNN Brain Cancer',
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
            'glioma': {
                'name': 'Gliome',
                'description': 'Tumeur cérébrale qui prend naissance dans les cellules gliales',
                'prevalence': '40-50% des tumeurs cérébrales primaires',
                'treatment': 'Chirurgie, radiothérapie, chimiothérapie'
            },
            'meningioma': {
                'name': 'Méningiome',
                'description': 'Tumeur se développant à partir des méninges',
                'prevalence': '30-35% des tumeurs cérébrales primaires',
                'treatment': 'Chirurgie, radiothérapie'
            },
            'notumor': {
                'name': 'Pas de tumeur',
                'description': 'Aucune tumeur cérébrale détectée',
                'prevalence': 'N/A',
                'treatment': 'Aucun traitement nécessaire'
            },
            'pituitary': {
                'name': 'Tumeur hypophysaire',
                'description': 'Tumeur de la glande pituitaire',
                'prevalence': '10-15% des tumeurs cérébrales primaires',
                'treatment': 'Chirurgie, médicaments, radiothérapie'
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