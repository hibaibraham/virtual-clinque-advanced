# 🧠 Brain Tumor Classifier — PyTorch

Classification de tumeurs cérébrales en 4 classes à partir d'images MRI.

| Classe | Description |
|--------|-------------|
| `glioma` | Tumeur des cellules gliales (aggressive) |
| `meningioma` | Tumeur des méninges (souvent bénigne) |
| `pituitary` | Tumeur de l'hypophyse |
| `notumor` | Pas de tumeur détectée |

---

## 📁 Structure du projet

```
brain_tumor_classifier/
  train.py                     ← Script d'entraînement complet
  predict.py                   ← Script de prédiction (inférence)
  brain_tumor_classifier.ipynb ← Notebook Jupyter interactif
  requirements.txt             ← Dépendances Python
  README.md                    ← Ce fichier

  output/                      ← Créé automatiquement
    best_model.pth             ← Meilleur modèle sauvegardé
    history.json               ← Historique d'entraînement
    training_curves.png
    confusion_matrix.png
    roc_curves.png
```

---

## 📦 Structure du Dataset attendue

```
dataset/
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

> Compatible avec le dataset **Kaggle Brain Tumor MRI Dataset**  
> https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

---

## 🎓 Entraînement

```bash
python train.py --data_dir ./dataset --save_dir ./output
```

Le modèle s'entraîne en **2 phases** :
- **Phase 1 (10 epochs)** : La base EfficientNet-B0 est gelée, seule la tête est entraînée
- **Phase 2 (20 epochs)** : Fine-tuning de tout le modèle avec un LR plus faible

Résultats typiques sur le dataset Kaggle : **~97-99% accuracy**

---

## 🔍 Prédiction

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

## 📊 Visualisations générées

| Fichier | Contenu |
|---------|---------|
| `training_curves.png` | Loss et accuracy train/val par epoch |
| `confusion_matrix.png` | Matrice de confusion (counts + %) |
| `roc_curves.png` | Courbes ROC + AUC par classe |
| `batch_predictions.png` | Grille de prédictions (mode batch) |

---

## 🏗️ Architecture

```
EfficientNet-B0 (pré-entraîné ImageNet)
    └── Classifier Head :
          Dropout(0.4)
          Linear(1280 → 256)
          ReLU
          Dropout(0.2)
          Linear(256 → 4)
```

**Hyperparamètres :**
- Image size : 224×224
- Batch size : 32
- Optimizer  : AdamW (weight_decay=1e-4)
- Scheduler  : CosineAnnealingLR
- Loss       : CrossEntropy (label_smoothing=0.1)
- Phase 1 LR : 1e-3
- Phase 2 LR : 1e-4

---

## 📓 Notebook Jupyter

Pour une expérience interactive avec visualisations step-by-step :

```bash
jupyter notebook brain_tumor_classifier.ipynb
```
