"""
Module d'Analyse Dentaire - Classification de Radiographies Dentaires
Modèle: ResNet18 avec Transfer Learning
Classes: Cavity, Fillings, Impacted Tooth, Implant, Normal
"""
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from datetime import datetime
import pandas as pd

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                         'tooth.model', 'data', 'tooth_model.pth')
CLASS_NAMES = ['Cavity', 'Fillings', 'Impacted Tooth', 'Implant', 'Normal']
NUM_CLASSES = 5
IMG_SIZE = 224
DEVELOPMENT_MODE = not os.path.exists(MODEL_PATH)

# Transformations pour les images
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Descriptions des classes
CLASS_DESCRIPTIONS = {
    'Cavity': {
        'icon': '🦷',
        'name': 'Carie Dentaire',
        'description': 'Lésion carieuse détectée. Nécessite un traitement dentaire.',
        'severity': 'Modéré à Élevé',
        'color': '#ef4444'
    },
    'Fillings': {
        'icon': '🔧',
        'name': 'Plombage',
        'description': 'Obturation dentaire présente. Dent précédemment traitée.',
        'severity': 'Traité',
        'color': '#3b82f6'
    },
    'Impacted Tooth': {
        'icon': '⚠️',
        'name': 'Dent Incluse',
        'description': 'Dent incluse ou semi-incluse. Peut nécessiter une extraction.',
        'severity': 'Élevé',
        'color': '#f59e0b'
    },
    'Implant': {
        'icon': '🦾',
        'name': 'Implant Dentaire',
        'description': 'Implant dentaire détecté. Prothèse en place.',
        'severity': 'Traité',
        'color': '#10b981'
    },
    'Normal': {
        'icon': '✅',
        'name': 'Dent Saine',
        'description': 'Aucune anomalie détectée. Dent en bonne santé.',
        'severity': 'Normal',
        'color': '#22c55e'
    }
}

@st.cache_resource
def load_model():
    """Charge le modèle ResNet18 pré-entraîné."""
    try:
        # Créer l'architecture ResNet18
        model = models.resnet18(pretrained=False)
        
        # Modifier la couche finale pour 5 classes
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, NUM_CLASSES)
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Charger les poids si le modèle existe
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            st.success("✅ Modèle ResNet18 chargé avec succès")
        else:
            st.warning(f"⚠️ Mode développement: Modèle non entraîné")
            st.info(f"📁 Chemin attendu: {MODEL_PATH}")
            # Initialiser avec des poids aléatoires pour le développement
        
        model.to(device)
        model.eval()
        return model, device
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        return None, None

def predict_image(image, model, device):
    """Effectue une prédiction sur une image."""
    try:
        # Prétraiter l'image
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Prédiction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Résultats
        pred_class = CLASS_NAMES[predicted.item()]
        pred_confidence = confidence.item() * 100
        
        # Toutes les probabilités
        all_probs = {CLASS_NAMES[i]: probabilities[0][i].item() * 100 
                    for i in range(NUM_CLASSES)}
        
        return pred_class, pred_confidence, all_probs
    
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction: {e}")
        return None, None, None

def save_tooth_prediction(patient_id, patient_name, prediction, confidence, image_name):
    """Sauvegarde la prédiction dans un fichier CSV."""
    try:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'tooth_predictions.csv')
        
        # Créer l'enregistrement
        record = {
            'timestamp': datetime.now().isoformat(),
            'patient_id': patient_id,
            'patient_name': patient_name,
            'prediction': prediction,
            'confidence': f"{confidence:.2f}%",
            'image_name': image_name,
            'medecin': st.session_state.get('auth_username', 'unknown')
        }
        
        # Charger ou créer le DataFrame
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        else:
            df = pd.DataFrame([record])
        
        # Sauvegarder
        df.to_csv(csv_path, index=False)
        return True
    
    except Exception as e:
        st.error(f"⚠️ Erreur lors de la sauvegarde: {e}")
        return False

def render():
    """Interface principale d'analyse dentaire."""
    
    # Charger le modèle
    model, device = load_model()
    
    if model is None:
        st.error("❌ Impossible de charger le modèle.")
        return
    
    # Informations sur le modèle
    if not DEVELOPMENT_MODE:
        st.success("✅ Modèle ResNet18 chargé avec succès")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🧠 Architecture", "ResNet18")
    with col2:
        st.metric("🎯 Classes", f"{NUM_CLASSES}")
    with col3:
        device_name = "GPU" if device.type == "cuda" else "CPU"
        st.metric("⚡ Device", device_name)
    
    st.markdown("---")
    
    # Section d'upload
    st.subheader("📤 Télécharger une Radiographie Dentaire")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Sélectionnez une image (JPG, PNG, JPEG)",
            type=['jpg', 'jpeg', 'png'],
            help="Téléchargez une radiographie dentaire pour analyse"
        )
    
    with col2:
        st.info("""
        **Classes détectées:**
        - 🦷 Carie
        - 🔧 Plombage
        - ⚠️ Dent Incluse
        - 🦾 Implant
        - ✅ Dent Saine
        """)
    
    if uploaded_file is not None:
        # Afficher l'image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Radiographie Téléchargée", use_container_width=True)
        
        with col2:
            # Informations patient (optionnel)
            st.markdown("### 👤 Informations Patient (Optionnel)")
            patient_id = st.text_input("ID Patient", placeholder="PAT20240101...")
            patient_name = st.text_input("Nom du Patient", placeholder="Nom Prénom")
            
            # Bouton d'analyse
            if st.button("🔍 Analyser la Radiographie", use_container_width=True, type="primary"):
                with st.spinner("🔄 Analyse en cours..."):
                    # Prédiction
                    pred_class, pred_confidence, all_probs = predict_image(image, model, device)
                    
                    if pred_class is not None:
                        # Sauvegarder si patient renseigné
                        if patient_id and patient_name:
                            save_tooth_prediction(patient_id, patient_name, pred_class, 
                                                pred_confidence, uploaded_file.name)
                        
                        # Stocker dans session state
                        st.session_state.tooth_prediction = {
                            'class': pred_class,
                            'confidence': pred_confidence,
                            'all_probs': all_probs
                        }
                        st.rerun()
        
        # Afficher les résultats si disponibles
        if 'tooth_prediction' in st.session_state:
            st.markdown("---")
            st.markdown("## 📊 Résultats de l'Analyse")
            
            pred = st.session_state.tooth_prediction
            pred_class = pred['class']
            pred_confidence = pred['confidence']
            all_probs = pred['all_probs']
            
            # Informations sur la classe prédite
            class_info = CLASS_DESCRIPTIONS[pred_class]
            
            # Carte de résultat principal
            st.markdown(f"""
            <div style='background:linear-gradient(135deg, {class_info['color']}15, {class_info['color']}05);
                        border:2px solid {class_info['color']};border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;'>
                <div style='text-align:center;'>
                    <div style='font-size:3rem;margin-bottom:0.5rem;'>{class_info['icon']}</div>
                    <h2 style='color:{class_info['color']};margin:0.5rem 0;'>{class_info['name']}</h2>
                    <p style='color:#94a3b8;font-size:1rem;margin:0.5rem 0;'>{class_info['description']}</p>
                    <div style='margin-top:1rem;'>
                        <span style='font-size:2rem;font-weight:700;color:{class_info['color']};'>
                            {pred_confidence:.1f}%
                        </span>
                        <p style='color:#64748b;font-size:0.85rem;margin:0.3rem 0;'>Confiance du Modèle</p>
                    </div>
                    <div style='margin-top:1rem;padding:0.8rem;background:rgba(0,0,0,0.2);border-radius:8px;'>
                        <span style='color:#94a3b8;font-size:0.85rem;'>Sévérité: </span>
                        <span style='color:{class_info['color']};font-weight:600;'>{class_info['severity']}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Distribution des probabilités
            st.markdown("### 📈 Distribution des Probabilités")
            
            # Trier par probabilité décroissante
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            
            for class_name, prob in sorted_probs:
                class_info = CLASS_DESCRIPTIONS[class_name]
                
                # Barre de progression avec couleur
                st.markdown(f"""
                <div style='margin-bottom:0.8rem;'>
                    <div style='display:flex;justify-content:space-between;margin-bottom:0.3rem;'>
                        <span style='color:#f1f5f9;font-size:0.9rem;'>
                            {class_info['icon']} {class_info['name']}
                        </span>
                        <span style='color:{class_info['color']};font-weight:600;font-size:0.9rem;'>
                            {prob:.1f}%
                        </span>
                    </div>
                    <div style='background:#1e293b;border-radius:8px;height:8px;overflow:hidden;'>
                        <div style='background:{class_info['color']};height:100%;width:{prob}%;
                                    transition:width 0.5s ease;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommandations
            st.markdown("---")
            st.markdown("### 💡 Recommandations Cliniques")
            
            if pred_class == 'Cavity':
                st.warning("""
                **Actions Recommandées:**
                - 🔍 Examen clinique approfondi
                - 🦷 Traitement de la carie (obturation)
                - 📅 Suivi régulier
                - 🪥 Renforcement de l'hygiène bucco-dentaire
                """)
            
            elif pred_class == 'Fillings':
                st.info("""
                **Actions Recommandées:**
                - ✅ Vérifier l'intégrité du plombage
                - 🔍 Surveiller les signes de carie secondaire
                - 📅 Contrôle périodique
                """)
            
            elif pred_class == 'Impacted Tooth':
                st.error("""
                **Actions Recommandées:**
                - ⚠️ Évaluation chirurgicale
                - 📸 Radiographies complémentaires (panoramique)
                - 🏥 Consultation spécialisée (chirurgie orale)
                - 💊 Gestion de la douleur si nécessaire
                """)
            
            elif pred_class == 'Implant':
                st.success("""
                **Actions Recommandées:**
                - ✅ Vérifier l'ostéointégration
                - 🔍 Surveiller les tissus péri-implantaires
                - 📅 Suivi régulier
                - 🪥 Hygiène péri-implantaire
                """)
            
            elif pred_class == 'Normal':
                st.success("""
                **Actions Recommandées:**
                - ✅ Aucune intervention nécessaire
                - 🪥 Maintenir une bonne hygiène bucco-dentaire
                - 📅 Contrôle annuel de routine
                - 🦷 Prévention continue
                """)
            
            # Bouton pour nouvelle analyse
            if st.button("🔄 Nouvelle Analyse", use_container_width=True):
                del st.session_state.tooth_prediction
                st.rerun()
    
    else:
        # Instructions
        st.info("""
        ### 📋 Instructions d'Utilisation
        
        1. **Téléchargez** une radiographie dentaire (format JPG, PNG ou JPEG)
        2. **Renseignez** les informations du patient (optionnel)
        3. **Cliquez** sur "Analyser la Radiographie"
        4. **Consultez** les résultats et recommandations
        
        ⚠️ **Note Importante:** Ce système est un outil d'aide au diagnostic. 
        Les résultats doivent toujours être validés par un professionnel de santé qualifié.
        """)
        
        # Exemples de classes
        st.markdown("---")
        st.markdown("### 🎯 Classes Détectées par le Modèle")
        
        cols = st.columns(5)
        for i, (class_name, info) in enumerate(CLASS_DESCRIPTIONS.items()):
            with cols[i]:
                st.markdown(f"""
                <div style='text-align:center;padding:1rem;background:rgba(0,212,255,0.05);
                            border:1px solid rgba(0,212,255,0.15);border-radius:10px;'>
                    <div style='font-size:2rem;margin-bottom:0.5rem;'>{info['icon']}</div>
                    <div style='color:#f1f5f9;font-size:0.85rem;font-weight:600;margin-bottom:0.3rem;'>
                        {info['name']}
                    </div>
                    <div style='color:#64748b;font-size:0.7rem;'>
                        {info['severity']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Informations sur le modèle
        st.markdown("---")
        st.markdown("### 🤖 À Propos du Modèle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Caractéristiques Techniques:**
            - 🧠 Architecture: ResNet18
            - 📚 Transfer Learning: ImageNet
            - 🎯 Accuracy: ~93%
            - ⚡ Temps d'inférence: <1s
            """)
        
        with col2:
            st.markdown("""
            **Dataset d'Entraînement:**
            - 📊 Source: Dental Radiography Segmentation
            - 🖼️ Images: Radiographies dentaires
            - 🏷️ Classes: 5 catégories
            - ✅ Validation: Test set indépendant
            """)

if __name__ == "__main__":
    render()
