# 🏥 MedAI — Clinique Virtuelle Intelligente v3.1

Plateforme de diagnostic médical par Intelligence Artificielle, combinant deux modules cliniques indépendants dans une interface Streamlit moderne et sécurisée.

---

## 🧩 Modules Disponibles

| Module | Modèle | Accuracy | Description |
|--------|--------|----------|-------------|
| 🩺 Diagnostic Thyroïdien | Random Forest | ~94.3% | Analyse des marqueurs biologiques thyroïdiens |
| 🧠 Tumeur Cérébrale (IRM) | EfficientNet-B0 | **99.85%** | Classification d'images IRM en 4 classes |

---

## 🚀 Installation

### 1. Cloner le projet et installer les dépendances

```bash
pip install -r requirements.txt
```

> Pour le module IRM, PyTorch est requis :
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. Entraîner le modèle Thyroïde

```bash
python train_and_save_model.py
```

### 3. Entraîner le modèle Tumeur Cérébrale

```bash
cd brain_tumer_deep
python train.py --data_dir brain-tumor-mri-dataset --save_dir output
```

> ⏱️ Durée estimée sur CPU : ~5–6 heures (30 epochs). Sur GPU : ~15–20 min.

### 4. Lancer l'application

```bash
streamlit run app.py
```

---

## 📂 Structure du Projet

```
virtual-clinque-advanced/
│
├── app.py                          # Application Streamlit principale
├── requirements.txt                # Dépendances Python
├── train_and_save_model.py         # Entraînement modèle thyroïde
├── thyroid.csv                     # Dataset thyroïde (9172 échantillons)
├── thyroid_ml_pipeline.ipynb       # Notebook ML thyroïde
├── prediction_history.csv          # Historique des prédictions
├── users.json                      # Utilisateurs (auth locale)
├── firebase_config.json            # Configuration Firebase
│
├── modules/                        # Pages de l'application
│   ├── prediction.py               # 🩺 Diagnostic thyroïdien
│   ├── brain_tumor.py              # 🧠 Analyse IRM cérébrale  ← NOUVEAU
│   ├── dashboard.py                # 📊 Tableau de bord
│   ├── historique.py               # 📜 Historique
│   └── apropos.py                  # ℹ️ À propos
│
├── utils/                          # Utilitaires partagés
│   ├── core.py                     # CSS, helpers, chargement modèle
│   ├── auth.py                     # Authentification 2FA
│   ├── firebase.py                 # Intégration Firebase
│   └── __init__.py
│
├── saved_models/                   # Modèle thyroïde (généré)
│   ├── model.joblib                # Random Forest optimisé
│   ├── preprocessor.joblib         # ColumnTransformer
│   └── feature_config.json         # Métadonnées + métriques
│
└── brain_tumer_deep/               # Module IRM cérébrale
    ├── train.py                    # Script d'entraînement EfficientNet-B0
    ├── predict.py                  # Script d'inférence standalone
    ├── brain_tumor_classifier.ipynb
    ├── requirements.txt
    ├── brain-tumor-mri-dataset/    # Dataset MRI (Training + Testing)
    └── output/                     # Modèle entraîné (généré)
        ├── best_model.pth          # ✅ Meilleur modèle sauvegardé
        ├── history.json            # Historique d'entraînement
        ├── training_curves.png
        ├── confusion_matrix.png
        └── roc_curves.png
```

---

## 🧬 Modèle 1 — Diagnostic Thyroïdien

- **Algorithme** : Random Forest (optimisé par RandomizedSearchCV)
- **Dataset** : UCI Thyroid Disease (9172 patients)
- **F1 Score (CV)** : ~95.8%
- **Accuracy (Test)** : ~94.3%
- **Équilibrage** : SMOTE (26% → 50% pathologiques)
- **Pipeline** : Nettoyage → Feature Engineering → StandardScaler → SMOTE → RandomForest

---

## 🧠 Modèle 2 — Tumeur Cérébrale (IRM)

- **Architecture** : EfficientNet-B0 (Transfer Learning, pré-entraîné ImageNet)
- **Dataset** : Brain Tumor MRI Dataset — Kaggle (5712 images d'entraînement)
- **Test Accuracy** : **99.85%**
- **Val Accuracy** : **99.69%**
- **Classes** : glioma · meningioma · notumor · pituitary
- **Entraînement** : 2 phases (tête gelée → fine-tuning complet)

### Résultats détaillés

| Classe | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| glioma | 1.00 | 1.00 | 1.00 |
| meningioma | 0.99 | 1.00 | 1.00 |
| notumor | 1.00 | 1.00 | 1.00 |
| pituitary | 1.00 | 0.99 | 1.00 |

### Architecture du classifier

```
EfficientNet-B0 (pré-entraîné ImageNet)
    └── Classifier Head :
          Dropout(0.4)
          Linear(1280 → 256)
          ReLU
          Dropout(0.2)
          Linear(256 → 4)
```

---

## 📱 Pages de l'Application

| Page | Description |
|------|-------------|
| 🩺 Prédiction | Formulaire patient + diagnostic thyroïdien en temps réel |
| 🧠 Tumeur Cérébrale | Upload IRM + classification par deep learning |
| 📊 Tableau de Bord | Statistiques, graphiques, métriques du modèle |
| 📜 Historique | Historique des prédictions avec filtrage et export |
| ℹ️ À Propos | Architecture, pipeline ML et valeurs de référence |

---

## 🔐 Authentification

L'application intègre un système d'authentification **2FA (TOTP)** via `pyotp`.  
Les comptes sont gérés dans `users.json` ou via Firebase selon la configuration.

---

## 🛠️ Stack Technique

| Couche | Technologies |
|--------|-------------|
| Frontend | Streamlit + CSS custom |
| ML Thyroïde | scikit-learn, imbalanced-learn (SMOTE) |
| Deep Learning IRM | PyTorch, torchvision (EfficientNet-B0) |
| Data | pandas, numpy |
| Visualisation | Plotly Express & Graph Objects |
| Auth | pyotp (2FA TOTP), Firebase |
| Export | Rapport texte téléchargeable |

---

## ⚠️ Avertissement Médical

> Ce système est un **outil d'aide à la décision** et ne remplace en aucun cas un diagnostic médical professionnel. Tout résultat doit être interprété par un professionnel de santé qualifié (médecin, radiologue, neurologue).
