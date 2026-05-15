# 🏥 Clinique Virtuelle Intelligente — Diagnostic Thyroïdien

Application de diagnostic thyroïdien basée sur le Machine Learning, avec une interface Streamlit moderne et professionnelle.

## 🚀 Installation

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Entraîner et sauvegarder le modèle

```bash
python train_and_save_model.py
```

Ce script reproduit le pipeline ML du notebook `thyroid_ml_pipeline.ipynb` :
- Nettoyage des données
- Feature engineering
- Entraînement du Random Forest optimisé
- Sauvegarde du modèle, preprocessor et configuration dans `saved_models/`

### 3. Lancer l'application

```bash
streamlit run app.py
```

## 📂 Structure du Projet

```
virtual clinque/
├── thyroid.csv                 # Dataset original (9172 échantillons)
├── thyroid_ml_pipeline.ipynb   # Notebook ML original
├── train_and_save_model.py     # Script d'entraînement et sauvegarde
├── app.py                      # Application Streamlit
├── requirements.txt            # Dépendances Python
├── README.md                   # Ce fichier
└── saved_models/               # Modèles sauvegardés (généré)
    ├── model.joblib             # Random Forest optimisé
    ├── preprocessor.joblib      # ColumnTransformer
    └── feature_config.json      # Métadonnées features + métriques
```

## 🧬 Le Modèle

- **Algorithme** : Random Forest (optimisé par RandomizedSearchCV)
- **F1 Score (CV)** : ~95.8%
- **Accuracy (Test)** : ~94.3%
- **Équilibrage** : SMOTE (26% → 50% pathologiques)

## 📱 Pages de l'Application

| Page | Description |
|------|-------------|
| 🩺 Prédiction | Formulaire patient + diagnostic en temps réel |
| 📊 Tableau de Bord | Statistiques, graphiques, métriques du modèle |
| 📜 Historique | Historique des prédictions avec filtrage et export |
| ℹ️ À Propos | Informations sur le modèle et plages normales |

