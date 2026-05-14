# 🧠 Brain Tumor Classifier — EfficientNet-B0 (PyTorch)

Classification de tumeurs cérébrales en 4 classes à partir d'images IRM, intégré dans la plateforme **MedAI Clinique Virtuelle v3.1**.

| Classe | Description |
|--------|-------------|
| `glioma` | Tumeur des cellules gliales — souvent agressive |
| `meningioma` | Tumeur des méninges — généralement bénigne |
| `pituitary` | Tumeur de l'hypophyse — affecte la régulation hormonale |
| `notumor` | Aucune tumeur détectée |

---

## 🏆 Résultats Obtenus

| Métrique | Valeur |
|----------|--------|
| Test Accuracy | **99.85%** |
| Val Accuracy (meilleure) | **99.69%** |
| Epochs | 30 (10 Phase 1 + 20 Phase 2) |
| Dispositif | CPU |

### Rapport de classification (jeu de test — 656 images)

| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| glioma | 1.00 | 1.00 | 1.00 | 147 |
| meningioma | 0.99 | 1.00 | 1.00 | 149 |
| notumor | 1.00 | 1.00 | 1.00 | 210 |
| pituitary | 1.00 | 0.99 | 1.00 | 150 |
| **accuracy** | | | **1.00** | **656** |

---

## 📁 Structure du Projet

```
brain_tumer_deep/
├── train.py                        # Script d'entraînement complet (2 phases)
├── predict.py                      # Script d'inférence standalone
├── brain_tumor_classifier.ipynb    # Notebook Jupyter interactif
├── requirements.txt                # Dépendances Python
├── README.md                       # Ce fichier
│
├── brain-tumor-mri-dataset/        # Dataset MRI
│   ├── Training/
│   │   ├── glioma/       (1321 images)
│   │   ├── meningioma/   (1339 images)
│   │   ├── notumor/      (1595 images)
│   │   └── pituitary/    (1457 images)
│   └── Testing/
│       ├── glioma/       (300 images)
│       ├── meningioma/   (306 images)
│       ├── notumor/      (405 images)
│       └── pituitary/    (300 images)
│
└── output/                         # Généré après entraînement
    ├── best_model.pth              # ✅ Meilleur modèle (val acc: 99.69%)
    ├── history.json                # Historique loss/accuracy par epoch
    ├── training_curves.png         # Courbes d'entraînement
    ├── confusion_matrix.png        # Matrice de confusion
    └── roc_curves.png              # Courbes ROC + AUC par classe
```

---

## 📦 Dataset

> **Kaggle Brain Tumor MRI Dataset**  
> https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Structure attendue :
```
brain-tumor-mri-dataset/
  Training/
    glioma/         ← images .jpg / .png
    meningioma/
    notumor/
    pituitary/
  Testing/
    glioma/
    meningioma/
    notumor/
    pituitary/
```

---

## 🚀 Installation

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 🎓 Entraînement

```bash
python train.py --data_dir brain-tumor-mri-dataset --save_dir output
```

Le modèle s'entraîne en **2 phases** :

| Phase | Epochs | LR | Description |
|-------|--------|----|-------------|
| Phase 1 | 10 | 1e-3 | Base EfficientNet gelée — seule la tête est entraînée |
| Phase 2 | 20 | 1e-4 | Fine-tuning complet du modèle |

> ⏱️ Durée estimée : ~5–6h sur CPU · ~15–20 min sur GPU CUDA

---

## 🔍 Prédiction Standalone

**Image unique :**
```bash
python predict.py --model output/best_model.pth --image path/to/mri.jpg
```

**Dossier complet (batch) :**
```bash
python predict.py --model output/best_model.pth --folder path/to/images/
```

**Avec sauvegarde de la visualisation :**
```bash
python predict.py --model output/best_model.pth --image mri.jpg --save result.png
```

---

## 🏗️ Architecture du Modèle

```
EfficientNet-B0 (pré-entraîné ImageNet — 5.3M paramètres)
    └── Classifier Head (remplacé) :
          Dropout(p=0.4)
          Linear(1280 → 256)
          ReLU
          Dropout(p=0.2)
          Linear(256 → 4)
```

### Hyperparamètres

| Paramètre | Valeur |
|-----------|--------|
| Image size | 224 × 224 |
| Batch size | 32 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Phase 1 LR | 1e-3 |
| Phase 2 LR | 1e-4 |
| Dropout | 0.4 / 0.2 |
| Seed | 42 |

### Augmentations (entraînement)

- RandomCrop (224×224 depuis 244×244)
- RandomHorizontalFlip / RandomVerticalFlip
- RandomRotation (±15°)
- ColorJitter (brightness, contrast, saturation)
- Normalisation ImageNet ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225])

---

## 📊 Visualisations Générées

| Fichier | Contenu |
|---------|---------|
| `training_curves.png` | Loss et accuracy train/val par epoch (30 epochs) |
| `confusion_matrix.png` | Matrice de confusion (counts + pourcentages) |
| `roc_curves.png` | Courbes ROC + AUC par classe |
| `batch_predictions.png` | Grille de prédictions (mode batch, max 16 images) |

---

## 🔗 Intégration dans MedAI

Ce module est intégré dans l'application principale via `modules/brain_tumor.py` :
- Chargement du modèle via `@st.cache_resource` (une seule fois)
- Upload d'image IRM depuis l'interface Streamlit
- Affichage du résultat, des probabilités et d'un rapport téléchargeable

Pour lancer l'application complète :
```bash
# Depuis la racine du projet
streamlit run app.py
```

---

## 📓 Notebook Jupyter

Pour une expérience interactive avec visualisations step-by-step :

```bash
jupyter notebook brain_tumor_classifier.ipynb
```

---

## ⚠️ Avertissement Médical

> Ce système est un **outil d'aide à la décision** uniquement. Tout résultat doit être confirmé par un radiologue ou neurologue qualifié. Il ne remplace pas un diagnostic médical professionnel.
