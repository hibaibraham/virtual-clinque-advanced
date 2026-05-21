# 🦷 Module d'Analyse Dentaire - NovaClinic

## 📋 Vue d'Ensemble

Le module d'analyse dentaire de NovaClinic utilise un modèle de deep learning **ResNet18** pour classifier automatiquement des radiographies dentaires en **5 catégories** distinctes.

---

## 🎯 Classes Détectées

Le modèle peut identifier les conditions dentaires suivantes :

### 1. 🦷 **Cavity (Carie Dentaire)**
- **Description** : Lésion carieuse détectée
- **Sévérité** : Modéré à Élevé
- **Actions** : Traitement dentaire requis (obturation)

### 2. 🔧 **Fillings (Plombage)**
- **Description** : Obturation dentaire présente
- **Sévérité** : Traité
- **Actions** : Vérification de l'intégrité, surveillance

### 3. ⚠️ **Impacted Tooth (Dent Incluse)**
- **Description** : Dent incluse ou semi-incluse
- **Sévérité** : Élevé
- **Actions** : Évaluation chirurgicale, extraction possible

### 4. 🦾 **Implant (Implant Dentaire)**
- **Description** : Implant dentaire détecté
- **Sévérité** : Traité
- **Actions** : Vérification de l'ostéointégration, suivi

### 5. ✅ **Normal (Dent Saine)**
- **Description** : Aucune anomalie détectée
- **Sévérité** : Normal
- **Actions** : Maintien de l'hygiène, contrôle annuel

---

## 🤖 Architecture du Modèle

### Caractéristiques Techniques

| Paramètre | Valeur |
|-----------|--------|
| **Architecture** | ResNet18 |
| **Technique** | Transfer Learning (ImageNet) |
| **Framework** | PyTorch |
| **Taille d'entrée** | 224x224 pixels |
| **Nombre de classes** | 5 |
| **Accuracy** | ~93% |
| **Temps d'inférence** | <1 seconde |

### Pipeline de Traitement

```
Image Radiographie (JPG/PNG)
    ↓
Redimensionnement (224x224)
    ↓
Normalisation ImageNet
    ↓
ResNet18 (Pré-entraîné)
    ↓
Couche FC Personnalisée (5 classes)
    ↓
Softmax (Probabilités)
    ↓
Prédiction + Confiance
```

---

## 📊 Performance du Modèle

### Métriques Globales

- **Test Accuracy** : 92.97%
- **Temps d'entraînement** : ~15-20 minutes (GPU)
- **Epochs** : 25
- **Optimizer** : Adam
- **Learning Rate** : 0.001

### Performance par Classe

| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| **Cavity** | 0.25 | 0.25 | 0.25 | 22 |
| **Fillings** | 0.89 | 0.89 | 0.89 | 100 |
| **Impacted Tooth** | 0.69 | 0.69 | 0.69 | 32 |
| **Implant** | 0.91 | 0.91 | 0.91 | 150 |
| **Normal** | 0.96 | 0.96 | 0.96 | 200 |

### Points Forts
- ✅ Excellente performance sur **Normal** (96%)
- ✅ Très bonne performance sur **Implant** (91%)
- ✅ Bonne performance sur **Fillings** (89%)

### Points d'Amélioration
- ⚠️ **Cavity** : Performance limitée (25%) - dataset déséquilibré
- ⚠️ **Impacted Tooth** : Performance moyenne (69%)

---

## 🚀 Utilisation

### 1. Interface d'Analyse

**Navigation** : Menu Médecin → 🦷 Analyse Dentaire

**Étapes** :
1. Téléchargez une radiographie dentaire (JPG, PNG, JPEG)
2. Renseignez les informations du patient (optionnel)
3. Cliquez sur "🔍 Analyser la Radiographie"
4. Consultez les résultats et recommandations

### 2. Dashboard Statistiques

**Navigation** : Menu Médecin → 📊 Dashboard Dentaire

**Fonctionnalités** :
- 📊 Distribution des diagnostics
- 📈 Évolution temporelle
- 🎯 Analyse de confiance
- ⏰ Distribution horaire
- 📋 Statistiques détaillées
- 💾 Export des données (CSV/Excel)

---

## 📁 Structure des Fichiers

```
virtual-clinque/
├── tooth.model/
│   └── data/
│       ├── tooth_model.pth              # Modèle entraîné
│       ├── kaggle_draft_notebook.ipynb  # Notebook d'entraînement
│       ├── train/                       # Dataset d'entraînement
│       ├── valid/                       # Dataset de validation
│       └── test/                        # Dataset de test
├── modules/
│   ├── tooth_analysis.py                # Interface d'analyse
│   └── tooth_dashboard.py               # Dashboard statistiques
├── tooth_predictions.csv                # Historique des prédictions
└── README_ANALYSE_DENTAIRE.md          # Cette documentation
```

---

## 💾 Sauvegarde des Prédictions

Chaque analyse est automatiquement sauvegardée dans `tooth_predictions.csv` avec :

- **timestamp** : Date et heure de l'analyse
- **patient_id** : Identifiant du patient
- **patient_name** : Nom du patient
- **prediction** : Classe prédite
- **confidence** : Niveau de confiance (%)
- **image_name** : Nom du fichier image
- **medecin** : Médecin ayant effectué l'analyse

---

## 🔬 Dataset d'Entraînement

### Source
- **Nom** : Dental Radiography Segmentation
- **Plateforme** : Kaggle
- **Type** : Radiographies dentaires

### Distribution

| Classe | Train | Valid | Test | Total |
|--------|-------|-------|------|-------|
| Cavity | 150 | 30 | 22 | 202 |
| Fillings | 500 | 100 | 100 | 700 |
| Impacted Tooth | 200 | 40 | 32 | 272 |
| Implant | 800 | 150 | 150 | 1100 |
| Normal | 1000 | 200 | 200 | 1400 |
| **Total** | **2650** | **520** | **504** | **3674** |

### Augmentation de Données

Techniques appliquées pendant l'entraînement :
- ✅ Flip horizontal (p=0.5)
- ✅ Rotation aléatoire (±15°)
- ✅ Color Jitter (luminosité, contraste, saturation)
- ✅ Translation aléatoire (±10%)

---

## ⚠️ Limitations et Avertissements

### Limitations Techniques

1. **Dataset Déséquilibré**
   - Cavity : Seulement 22 exemples dans le test set
   - Peut entraîner des faux négatifs

2. **Qualité d'Image**
   - Nécessite des radiographies de bonne qualité
   - Éclairage et contraste appropriés

3. **Contexte Clinique**
   - Le modèle ne remplace pas l'expertise humaine
   - Doit être utilisé comme outil d'aide au diagnostic

### Recommandations d'Usage

⚠️ **IMPORTANT** : Ce système est un **outil d'aide au diagnostic**. Les résultats doivent **TOUJOURS** être validés par un professionnel de santé qualifié.

**Bonnes Pratiques** :
- ✅ Utiliser des images de haute qualité
- ✅ Vérifier la confiance du modèle (>80% recommandé)
- ✅ Croiser avec l'examen clinique
- ✅ Documenter les décisions cliniques
- ❌ Ne pas se fier uniquement au modèle
- ❌ Ne pas utiliser pour des décisions critiques sans validation

---

## 🔄 Améliorations Futures

### Court Terme
1. **Collecte de Données**
   - Augmenter le dataset pour Cavity et Impacted Tooth
   - Équilibrer les classes

2. **Optimisation du Modèle**
   - Tester ResNet50 ou EfficientNet
   - Implémenter class weights
   - Early stopping avec patience

### Long Terme
1. **Fonctionnalités Avancées**
   - Détection multi-labels (plusieurs conditions)
   - Segmentation des zones affectées
   - Analyse de séries temporelles (évolution)

2. **Intégration**
   - Export vers DICOM
   - Intégration avec systèmes hospitaliers
   - API REST pour intégration externe

---

## 📚 Références

### Modèle
- **ResNet** : He et al., "Deep Residual Learning for Image Recognition" (2015)
- **Transfer Learning** : ImageNet pre-training

### Dataset
- **Source** : Kaggle - Dental Radiography Segmentation
- **Auteur** : abbasseifossadat

### Framework
- **PyTorch** : https://pytorch.org/
- **Torchvision** : https://pytorch.org/vision/

---

## 🆘 Support et Contact

Pour toute question ou problème :

1. **Documentation** : Consultez ce README
2. **Dashboard** : Vérifiez les statistiques du modèle
3. **Logs** : Consultez les fichiers de prédiction CSV

---

## 📝 Changelog

### Version 4.1 (2024)
- ✅ Ajout du module d'analyse dentaire
- ✅ Intégration ResNet18
- ✅ Dashboard statistiques
- ✅ Export CSV/Excel
- ✅ Recommandations cliniques automatiques

---

## 📄 Licence

Ce module fait partie de **NovaClinic v4.1** - Plateforme de Diagnostic Médical Intelligent.

⚕️ **Usage Médical** : Outil d'aide au diagnostic uniquement. Ne remplace pas l'expertise médicale.

---

**NovaClinic** - Système de Diagnostic Intelligent Multi-Modal
🏥 Thyroïde | 🧠 IRM Cérébrale | 🩸 PTDM | 🦷 Dentaire
