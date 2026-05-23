# 🏥 NovaClinic - Plateforme de Diagnostic Médical Intelligent v4.1

## 📋 Vue d'Ensemble

**NovaClinic** est une plateforme complète de diagnostic médical par Intelligence Artificielle, combinant plusieurs modules de Deep Learning pour l'aide au diagnostic médical.

---

## 🎯 Modules Deep Learning

| Module | Modèle | Accuracy | Paramètres | Description |
|--------|--------|----------|------------|-------------|
| 🧠 **IRM Cérébrale** | EfficientNet-B0 | **99.85%** | 5.3M | Classification tumeurs (4 classes) |
| 🦷 **Analyse Dentaire** | ResNet18 | **93%** | 11.7M | Classification radiographies (5 classes) |
| � **PTDM** | SVM / Random Forest | **85%** | N/A | Prédiction diabète post-transplantation |
| �🩺 **Thyroïde** | Random Forest | 94.3% | N/A | Analyse marqueurs biologiques |

---

## 🚀 Installation

```bash
# Cloner
git clone https://github.com/hibaibraham/virtual-clinque-advanced.git
cd virtual-clinque-advanced

# Environnement
python -m venv .venv
.venv\Scripts\activate

# Dépendances
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Lancer
streamlit run app.py
```

---

## 🧠 MODULE IRM CÉRÉBRALE (EfficientNet-B0)

### Caractéristiques
- **Architecture**: EfficientNet-B0 (Transfer Learning)
- **Dataset**: Brain Tumor MRI (7,023 images équilibrées)
- **Classes**: Gliome, Méningiome, Pas de tumeur, Hypophysaire
- **Test Accuracy**: **99.85%**
- **Temps**: ~50ms par image

### Architecture
```
EfficientNet-B0 (ImageNet) → GAP → Dropout(0.4) → Linear(1280→256) 
→ ReLU → Dropout(0.2) → Linear(256→4) → Softmax
```

### Pourquoi EfficientNet-B0?
✅ Précision exceptionnelle (99.85%)  
✅ Léger (5.3M params, 20 MB)  
✅ Rapide (50ms)  
✅ Compound scaling optimal  

### Performances
| Classe | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| Gliome | 1.00 | 1.00 | 1.00 |
| Méningiome | 0.99 | 1.00 | 1.00 |
| Pas tumeur | 1.00 | 1.00 | 1.00 |
| Hypophysaire | 1.00 | 0.99 | 1.00 |

---

## 🦷 MODULE DENTAIRE (ResNet18)

### Caractéristiques
- **Architecture**: ResNet18 (Transfer Learning)
- **Dataset**: Dental Radiography (3,674 images déséquilibrées)
- **Classes**: Cavity, Fillings, Impacted, Implant, Normal
- **Test Accuracy**: **93%**
- **Temps**: <1s par image

### Architecture
```
ResNet18 (ImageNet) → 4 Residual Blocks (64→128→256→512 filtres)
→ GAP → Dropout(0.5) → Linear(512→5) → Softmax
```

### Pourquoi ResNet18?
✅ Robuste sur datasets déséquilibrés  
✅ Residual connections (évite vanishing gradient)  
✅ Taille optimale (11.7M params)  
✅ Stable et éprouvé  

### Performances
| Classe | Precision | Recall | F1 | Support |
|--------|-----------|--------|-----|---------|
| Cavity | 0.25 | 0.25 | 0.25 | 22 ⚠️ |
| Fillings | 0.89 | 0.89 | 0.89 | 100 ✅ |
| Impacted | 0.69 | 0.69 | 0.69 | 32 ⚠️ |
| Implant | 0.91 | 0.91 | 0.91 | 150 ✅ |
| Normal | 0.96 | 0.96 | 0.96 | 200 ✅ |

**Note**: Cavity faible (22 images) → Besoin de plus de données

### Analyse des Courbes
```
✅ PAS d'overfitting (Gap Train-Val: 1.4%)
✅ PAS d'underfitting (93% accuracy)
✅ Convergence stable (epoch 20-25)
⚠️ Déséquilibre classes (Cavity: 25%)
```

---

## 🩸 MODULE PTDM (SVM / Random Forest)

### Caractéristiques
- **Architecture**: Ensemble de modèles ML (SVM, Random Forest, Logistic Regression)
- **Dataset**: Données de transplantation rénale
- **Objectif**: Prédiction du diabète post-transplantation (PTDM)
- **Test Accuracy**: **85%**
- **AUC**: **88%**

### Modèles Utilisés
```
Ensemble de 3 modèles:
├─ Logistic Regression (régression logistique)
├─ Random Forest (forêt aléatoire)
└─ SVM (Support Vector Machine)

Prédiction finale: Vote majoritaire ou moyenne des probabilités
```

### Features Importantes
| Feature | Importance | Description |
|---------|------------|-------------|
| HbA1c_pre_TR_R | 35% | Hémoglobine glyquée pré-transplantation |
| glycémie_pre_TR_R | 25% | Glycémie à jeun pré-transplantation |
| age_receveur_TR | 15% | Âge du receveur |
| obésité_pre_TR_receveur | 10% | Obésité pré-transplantation |
| durée_dialyse_année | 8% | Durée de dialyse en années |
| HTA_pre_TR_receveur | 4% | Hypertension artérielle |
| age_donneur | 2% | Âge du donneur |
| sexe_receveur_M | 1% | Sexe du receveur (Masculin) |

### Pourquoi un Ensemble de Modèles?
✅ Robustesse accrue (vote majoritaire)  
✅ Meilleure généralisation  
✅ Capture différents patterns  
✅ Réduit le risque d'overfitting  

### Règles de Prédiction
```python
# Calcul du score de risque
risk_score = 0.1  # Base

if HbA1c > 6.5:
    risk_score += 0.4  # Risque élevé
elif HbA1c > 5.7:
    risk_score += 0.2  # Risque modéré

if glycémie > 1.26 g/L:
    risk_score += 0.3  # Hyperglycémie
elif glycémie > 1.1 g/L:
    risk_score += 0.15  # Glycémie élevée

if obésité == 1:
    risk_score += 0.1  # Facteur de risque

if age > 50:
    risk_score += 0.1  # Âge avancé

# Prédiction
PTDM = 1 if risk_score > 0.5 else 0
```

### Plages Normales
| Marqueur | Plage Normale | Unité |
|----------|---------------|-------|
| Glycémie | 0.7 - 1.1 | g/L |
| HbA1c | 4.0 - 5.7 | % |

---

## 🔍 COMPARAISON MODÈLES

| Modèle | Accuracy | Params | Temps | Verdict |
|--------|----------|--------|-------|---------|
| **EfficientNet-B0** | **99.85%** | **5.3M** | **50ms** | 🥇 Optimal |
| **ResNet18** | **93%** | **11.7M** | **<1s** | 🥇 Optimal |
| ResNet50 | 94% | 25.6M | 120ms | ❌ Trop lourd |
| VGG-16 | 91% | 138M | 200ms | ❌ Obsolète |
| MobileNet-V2 | 88% | 3.5M | 40ms | ❌ Moins précis |

---

## 🎓 LEÇONS APPRISES

1. **Plus Grand ≠ Meilleur**: EfficientNet-B0 (5.3M) bat VGG-16 (138M)
2. **Transfer Learning Essentiel**: +15-20% accuracy avec ImageNet
3. **Dataset Équilibré Important**: IRM (équilibré) → 99.85%, Dentaire (déséquilibré) → 93%
4. **Compound Scaling Fonctionne**: Optimisation multi-dimensionnelle

---

## 🛠️ STACK TECHNIQUE

- **Frontend**: Streamlit
- **Deep Learning**: PyTorch, torchvision
- **ML Classique**: scikit-learn
- **Data**: pandas, numpy
- **Viz**: Plotly
- **Auth**: pyotp (2FA)
- **DB**: MongoDB (fallback JSON/CSV)

---

## 📚 RÉFÉRENCES

- **EfficientNet** (2019): Tan & Le, Google Research
- **ResNet** (2015): He et al., Microsoft Research
- **Datasets**: Kaggle (Brain Tumor MRI, Dental Radiography)

---

## ⚠️ AVERTISSEMENT

> **IMPORTANT**: Outil d'aide à la décision uniquement. Ne remplace pas un diagnostic médical professionnel.

---

## 📄 LICENCE

**NovaClinic v4.1** - Système de Diagnostic Intelligent Multi-Modal  
🏥 Thyroïde | 🧠 IRM | 🩸 PTDM | 🦷 Dentaire | 👁️ Oculaire

*Document créé le 23 Mai 2026*
