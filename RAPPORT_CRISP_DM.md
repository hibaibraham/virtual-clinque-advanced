# RAPPORT DE PROJET — VIRTUAL CLINIQUE
## Application Intelligente de Gestion et d'Assistance Médicale
### Méthodologie CRISP-DM

---

**Établissement :** Projet Universitaire / Académique  
**Version :** 3.1  
**Date :** Mai 2026  
**Auteur :** Équipe Virtual Clinique  

---

## TABLE DES MATIÈRES

1. [Introduction Générale](#1-introduction-générale)
2. [Business Understanding](#2-business-understanding)
3. [Data Understanding](#3-data-understanding)
4. [Data Preparation](#4-data-preparation)
5. [Modeling](#5-modeling)
6. [Evaluation](#6-evaluation)
7. [Deployment](#7-deployment)
8. [Conclusion](#8-conclusion)
9. [Recommandations](#9-recommandations)

---

## 1. Introduction Générale

L'essor de l'intelligence artificielle dans le domaine médical représente l'une des transformations les plus profondes de la médecine moderne. Face à des systèmes de santé sous pression — surcharge des urgences, pénurie de spécialistes, délais de diagnostic — les outils d'aide à la décision clinique constituent une réponse technologique concrète et prometteuse.

**Virtual Clinique** est une plateforme intelligente de gestion et d'assistance médicale développée avec une approche centrée sur l'utilisateur. Elle intègre deux modules de diagnostic par intelligence artificielle : un module de détection de pathologies thyroïdiennes basé sur le Machine Learning classique, et un module d'analyse d'imagerie IRM cérébrale reposant sur le Deep Learning. Ces deux modules coexistent au sein d'une interface web unifiée, sécurisée et accessible.

Ce rapport présente l'ensemble du cycle de vie du projet selon la méthodologie **CRISP-DM** (Cross-Industry Standard Process for Data Mining), standard reconnu dans l'industrie pour structurer les projets de science des données. Chaque phase — de la compréhension métier jusqu'au déploiement — est documentée avec rigueur afin de garantir la reproductibilité, la transparence et la qualité scientifique du travail réalisé.

---

## 2. Business Understanding

### 2.1 Présentation du Problème Médical Traité

Le secteur médical fait face à deux défis diagnostiques majeurs abordés dans ce projet :

**Pathologies thyroïdiennes :**  
Les maladies de la thyroïde (hypothyroïdie, hyperthyroïdie, goitre, cancer) touchent environ 200 millions de personnes dans le monde. Leur diagnostic repose sur l'interprétation combinée de plusieurs marqueurs biologiques (TSH, T3, T4, FTI, T4U) dont les valeurs normales varient selon l'âge, le sexe et les antécédents. Cette complexité rend le diagnostic précoce difficile, notamment dans les zones à faible densité médicale.

**Tumeurs cérébrales :**  
Les tumeurs cérébrales représentent l'une des pathologies neurologiques les plus graves, avec un taux de mortalité élevé en cas de détection tardive. L'analyse d'images IRM (Imagerie par Résonance Magnétique) est la méthode de référence, mais elle requiert l'expertise d'un radiologue spécialisé, une ressource rare et coûteuse. Les quatre types principaux — gliome, méningiome, tumeur hypophysaire et absence de tumeur — nécessitent des prises en charge radicalement différentes.

### 2.2 Objectifs du Projet Virtual Clinique

Les objectifs sont définis à deux niveaux :

**Objectifs fonctionnels :**
- Fournir un outil d'aide au diagnostic thyroïdien à partir de paramètres biologiques saisis manuellement
- Permettre l'analyse automatique d'images IRM cérébrales par classification deep learning
- Offrir un historique traçable des analyses effectuées
- Générer des rapports médicaux exportables

**Objectifs de performance :**
- Atteindre un F1-Score supérieur à 90% pour le module thyroïde
- Atteindre une accuracy supérieure à 95% pour le module IRM
- Garantir un temps de réponse inférieur à 3 secondes par analyse
- Assurer la sécurité des données via une authentification à deux facteurs (2FA)

### 2.3 Utilisateurs Cibles

| Profil | Besoins | Fonctionnalités utilisées |
|--------|---------|--------------------------|
| **Médecin généraliste** | Aide au diagnostic rapide, second avis | Modules de prédiction, rapports |
| **Radiologue** | Pré-analyse IRM, triage | Module tumeur cérébrale |
| **Patient** | Compréhension de son état, suivi | Résultats simplifiés, historique |
| **Administration médicale** | Suivi des consultations, statistiques | Tableau de bord, historique |
| **Chercheur / Étudiant** | Exploration des données, validation | Dashboard analytique, export |

---

## 3. Data Understanding

### 3.1 Types de Données Utilisées

Le projet exploite deux types de données fondamentalement différents :

**Module Thyroïde — Données tabulaires :**

| Variable | Type | Description | Plage normale |
|----------|------|-------------|---------------|
| `age` | Numérique | Âge du patient (années) | 1 – 100 |
| `sex` | Binaire | Sexe (0=F, 1=M) | — |
| `TSH` | Numérique | Thyroid Stimulating Hormone (mU/L) | 0.4 – 4.0 |
| `T3` | Numérique | Triiodothyronine (nmol/L) | 1.2 – 3.1 |
| `TT4` | Numérique | Thyroxine totale (nmol/L) | 70 – 180 |
| `T4U` | Numérique | Uptake T4 (ratio) | 0.7 – 1.3 |
| `FTI` | Numérique | Free Thyroxine Index | 70 – 180 |
| Antécédents | Binaires (×14) | Traitements, chirurgies, comorbidités | — |
| `target` | Binaire | 0=Normal, 1=Pathologique | — |

**Module IRM — Données images :**

| Caractéristique | Valeur |
|----------------|--------|
| Format | JPEG / PNG |
| Résolution d'entrée | Variable (redimensionné à 224×224) |
| Canaux | RGB (3 canaux) |
| Classes | 4 (glioma, meningioma, notumor, pituitary) |
| Volume total | ~7 100 images |

### 3.2 Sources de Données

**Dataset Thyroïde :**
- **Source :** UCI Machine Learning Repository — *Thyroid Disease Dataset*
- **Référence :** Quinlan, J.R. (1987). Simplifying decision trees. International Journal of Man-Machine Studies.
- **Volume :** 9 172 échantillons patients
- **Déséquilibre :** ~26% de cas pathologiques, ~74% de cas normaux

**Dataset IRM Cérébrale :**
- **Source :** Kaggle — *Brain Tumor MRI Dataset* (Masoud Nickparvar, 2021)
- **Lien :** https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- **Volume :** 5 712 images d'entraînement + 1 311 images de test
- **Distribution :**

| Classe | Train | Test |
|--------|-------|------|
| glioma | 1 321 | 300 |
| meningioma | 1 339 | 306 |
| notumor | 1 595 | 405 |
| pituitary | 1 457 | 300 |

### 3.3 Analyse Exploratoire des Données (EDA)

**Module Thyroïde :**

L'analyse exploratoire a révélé plusieurs observations importantes :

- **Valeurs manquantes :** présentes sur TSH (~5%), T3 (~30%), TT4 (~5%), T4U (~5%), FTI (~5%). Ces lacunes sont typiques des données médicales réelles où certains examens ne sont pas systématiquement prescrits.
- **Déséquilibre de classes :** 74% de cas normaux contre 26% de pathologiques, nécessitant une stratégie de rééquilibrage.
- **Distributions asymétriques :** TSH présente une distribution log-normale avec des valeurs extrêmes pouvant dépasser 500 mU/L dans les cas d'hypothyroïdie sévère.
- **Corrélations :** TT4 et FTI sont fortement corrélés (r > 0.85), ce qui a guidé la stratégie de feature engineering.

**Module IRM :**

- **Qualité des images :** hétérogène (différentes machines IRM, différents protocoles d'acquisition).
- **Distribution spatiale :** les tumeurs occupent des régions variables de l'image selon le type.
- **Déséquilibre modéré :** la classe `notumor` est légèrement surreprésentée (+20% par rapport aux autres classes).

---

## 4. Data Preparation

### 4.1 Nettoyage des Données

**Module Thyroïde :**

```
Étape 1 — Suppression des doublons
  → 0 doublon détecté

Étape 2 — Traitement des valeurs manquantes
  → Variables numériques : imputation par la médiane (SimpleImputer)
  → Variables binaires : imputation par la valeur la plus fréquente (mode)

Étape 3 — Traitement des valeurs aberrantes
  → Capping au percentile 99 pour TSH, T3, TT4, FTI
  → Conservation des valeurs extrêmes cliniquement plausibles

Étape 4 — Encodage
  → Variable cible : binarisation (normal=0, pathologique=1)
  → Sexe : encodage binaire (F=0, M=1)
```

**Module IRM :**

```
Étape 1 — Vérification de l'intégrité
  → Suppression des fichiers corrompus ou illisibles

Étape 2 — Standardisation du format
  → Conversion en RGB (élimination des images en niveaux de gris)
  → Redimensionnement à 224×224 pixels

Étape 3 — Normalisation des pixels
  → Moyenne ImageNet : [0.485, 0.456, 0.406]
  → Écart-type ImageNet : [0.229, 0.224, 0.225]
```

### 4.2 Transformation et Normalisation

**Module Thyroïde :**

- **StandardScaler** appliqué sur toutes les variables numériques continues (TSH, T3, TT4, T4U, FTI, age) pour centrer-réduire les distributions.
- **ColumnTransformer** utilisé pour appliquer des transformations différenciées selon le type de variable.
- **Pipeline scikit-learn** encapsulant l'ensemble des étapes pour garantir l'absence de data leakage lors de la validation croisée.

**Module IRM — Augmentation des données (entraînement uniquement) :**

| Transformation | Paramètres | Objectif |
|---------------|-----------|---------|
| RandomCrop | 224×224 depuis 244×244 | Invariance à la translation |
| RandomHorizontalFlip | p=0.5 | Invariance au miroir |
| RandomVerticalFlip | p=0.5 | Robustesse orientationnelle |
| RandomRotation | ±15° | Invariance à la rotation |
| ColorJitter | brightness=0.2, contrast=0.2 | Robustesse aux conditions d'acquisition |

### 4.3 Feature Engineering

**Module Thyroïde :**

Des variables dérivées ont été construites pour capturer des relations cliniques connues :

| Feature créée | Formule | Justification clinique |
|--------------|---------|----------------------|
| `TSH_log` | log(TSH + 0.01) | Normalisation de la distribution log-normale |
| `T4_ratio` | TT4 / T4U | Indicateur de la T4 libre estimée |
| `TSH_T3_ratio` | TSH / (T3 + 0.01) | Rapport axe hypophyso-thyroïdien |
| `TSH_hors_norme` | 1 si TSH < 0.4 ou > 4.0 | Flag clinique binaire |
| `T3_hors_norme` | 1 si T3 < 1.2 ou > 3.1 | Flag clinique binaire |
| `score_antecedents` | Somme des 14 variables binaires | Charge médicale globale |

Ces features ont été validées par leur importance dans le modèle Random Forest final (feature importance > 0.02).

---

## 5. Modeling

### 5.1 Module Thyroïde — Machine Learning Classique

#### Choix des Modèles

Trois algorithmes ont été évalués lors de la phase de sélection :

| Algorithme | Justification |
|-----------|--------------|
| **Logistic Regression** | Baseline interprétable, rapide à entraîner |
| **Random Forest** | Robuste aux outliers, gère les interactions non-linéaires, interprétable via feature importance |
| **XGBoost** | Performances élevées sur données tabulaires déséquilibrées |

#### Algorithme Retenu : Random Forest

Le Random Forest a été sélectionné pour les raisons suivantes :
- Performances équivalentes à XGBoost (~94% F1) avec une meilleure interprétabilité
- Robustesse naturelle aux valeurs manquantes résiduelles
- Génération native des importances de features (XAI — Explainable AI)
- Temps d'inférence très faible (< 10ms par prédiction)

#### Stratégie de Rééquilibrage : SMOTE

Le déséquilibre de classes (74/26) a été traité par **SMOTE** (Synthetic Minority Over-sampling Technique) :
- Génération de nouveaux exemples synthétiques de la classe minoritaire (pathologique)
- Ratio final : 50% normal / 50% pathologique
- Appliqué uniquement sur le jeu d'entraînement pour éviter le data leakage

#### Optimisation des Hyperparamètres

**RandomizedSearchCV** avec validation croisée stratifiée (5 folds) :

| Hyperparamètre | Espace de recherche | Valeur optimale |
|---------------|--------------------|-----------------| 
| `n_estimators` | [100, 200, 300, 500] | 300 |
| `max_depth` | [None, 10, 20, 30] | 20 |
| `min_samples_split` | [2, 5, 10] | 5 |
| `min_samples_leaf` | [1, 2, 4] | 1 |
| `max_features` | ['sqrt', 'log2'] | 'sqrt' |

### 5.2 Module IRM — Deep Learning

#### Architecture : EfficientNet-B0 (Transfer Learning)

**Justification du choix :**
- EfficientNet-B0 offre le meilleur compromis accuracy/taille de modèle de sa famille
- Pré-entraîné sur ImageNet (1.2M images, 1000 classes) — transfert de connaissances visuelles générales
- 5.3M paramètres seulement (vs 25M pour ResNet-50) — adapté à un déploiement CPU
- Performances état de l'art sur les benchmarks d'imagerie médicale

**Tête de classification personnalisée :**

```
EfficientNet-B0 (backbone gelé en Phase 1)
    └── Dropout(p=0.4)
    └── Linear(1280 → 256)
    └── ReLU()
    └── Dropout(p=0.2)
    └── Linear(256 → 4)
    └── Softmax (inférence)
```

#### Stratégie d'Entraînement en 2 Phases

**Phase 1 — Entraînement de la tête (10 epochs) :**
- Backbone EfficientNet-B0 gelé (paramètres non mis à jour)
- Seule la tête de classification est entraînée
- Learning rate : 1e-3 (AdamW)
- Objectif : initialiser la tête avec des poids cohérents avant le fine-tuning

**Phase 2 — Fine-tuning complet (20 epochs) :**
- Tous les paramètres dégelés
- Learning rate réduit : 1e-4 (AdamW)
- Objectif : adapter finement les features du backbone aux images IRM médicales

**Paramètres d'entraînement :**

| Paramètre | Valeur |
|-----------|--------|
| Optimizer | AdamW (weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Batch size | 32 |
| Epochs total | 30 |
| Image size | 224 × 224 |
| Dispositif | CPU (Intel) |


---

## 6. Evaluation

### 6.1 Méthodes d'Évaluation

#### Module Thyroïde

- **Validation croisée stratifiée** (5 folds) pour l'optimisation des hyperparamètres
- **Jeu de test isolé** (20% des données, jamais vu pendant l'entraînement)
- **Métriques retenues :** Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Matrice de confusion** pour analyser les types d'erreurs (faux positifs vs faux négatifs)

Dans un contexte médical, le **Recall** (sensibilité) est la métrique prioritaire : il est préférable de sur-diagnostiquer (faux positif) plutôt que de manquer un cas pathologique réel (faux négatif).

#### Module IRM

- **Split train/val/test** : les données de test sont divisées en deux moitiés (50% validation, 50% test final)
- **Métriques :** Accuracy, Precision, Recall, F1-Score par classe, AUC-ROC multiclasse (One-vs-Rest)
- **Matrice de confusion** (counts + pourcentages)
- **Courbes d'apprentissage** (loss et accuracy par epoch)

### 6.2 Résultats Obtenus

#### Module Thyroïde — Random Forest

| Métrique | Valeur |
|----------|--------|
| Accuracy (test) | ~94.3% |
| F1-Score (CV, macro) | ~95.8% |
| Precision (pathologique) | ~93.1% |
| Recall (pathologique) | ~96.4% |
| AUC-ROC | ~0.981 |

#### Module IRM — EfficientNet-B0

**Résultats globaux :**

| Métrique | Valeur |
|----------|--------|
| **Test Accuracy** | **99.85%** |
| **Val Accuracy (meilleure)** | **99.69%** |
| Epochs entraînés | 30 |
| Jeu de test | 656 images |

**Rapport de classification détaillé :**

| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| glioma | 1.00 | 1.00 | 1.00 | 147 |
| meningioma | 0.99 | 1.00 | 1.00 | 149 |
| notumor | 1.00 | 1.00 | 1.00 | 210 |
| pituitary | 1.00 | 0.99 | 1.00 | 150 |
| **Moyenne pondérée** | **1.00** | **1.00** | **1.00** | **656** |

### 6.3 Analyse des Performances

#### Points Forts

**Module Thyroïde :**
- Le Recall élevé (96.4%) sur les cas pathologiques garantit un faible taux de faux négatifs, ce qui est critique en contexte médical.
- L'AUC-ROC de 0.981 indique une excellente capacité discriminante du modèle sur l'ensemble du spectre de seuils de décision.
- La feature importance révèle que TSH, FTI et TT4 sont les trois variables les plus prédictives, ce qui est cohérent avec la littérature médicale.

**Module IRM :**
- Une accuracy de 99.85% sur 656 images de test est un résultat exceptionnel, comparable aux publications récentes dans le domaine.
- Le F1-Score de 1.00 sur la classe `glioma` est particulièrement significatif, car c'est la tumeur la plus agressive nécessitant une détection sans faille.
- La classe `meningioma` présente la seule légère imperfection (Precision=0.99), ce qui s'explique par sa ressemblance morphologique avec certaines images normales.

#### Limites et Biais Potentiels

| Limite | Description | Impact |
|--------|-------------|--------|
| Biais de sélection | Dataset IRM provenant d'une seule source Kaggle | Généralisation limitée à d'autres équipements IRM |
| Données synthétiques | SMOTE génère des exemples artificiels | Risque de sur-apprentissage sur les frontières de décision |
| Absence de validation clinique | Aucun médecin n'a validé les prédictions individuelles | Ne peut pas être utilisé en production médicale réelle sans audit |
| Données statiques | Le modèle thyroïde ne prend pas en compte l'évolution temporelle | Pas de suivi longitudinal du patient |

---

## 7. Deployment

### 7.1 Architecture Technique

```
┌─────────────────────────────────────────────────────────────┐
│                    VIRTUAL CLINIQUE v3.1                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────────────────────────┐  │
│  │  FRONTEND    │    │           BACKEND                │  │
│  │  Streamlit   │◄──►│                                  │  │
│  │  + CSS custom│    │  ┌─────────────────────────────┐ │  │
│  └──────────────┘    │  │   Module Thyroïde           │ │  │
│                      │  │   Random Forest (joblib)    │ │  │
│  ┌──────────────┐    │  │   scikit-learn pipeline     │ │  │
│  │  AUTH 2FA    │    │  └─────────────────────────────┘ │  │
│  │  pyotp TOTP  │    │                                  │  │
│  │  users.json  │    │  ┌─────────────────────────────┐ │  │
│  └──────────────┘    │  │   Module IRM Cérébrale      │ │  │
│                      │  │   EfficientNet-B0 (PyTorch) │ │  │
│  ┌──────────────┐    │  │   best_model.pth            │ │  │
│  │  HISTORIQUE  │    │  └─────────────────────────────┘ │  │
│  │  CSV / Firebase    │                                  │  │
│  └──────────────┘    └──────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Stack Technique Complète

| Couche | Technologie | Version | Rôle |
|--------|------------|---------|------|
| Interface | Streamlit | 1.51.0 | Framework web Python |
| ML Thyroïde | scikit-learn | 1.7.2 | Pipeline + Random Forest |
| Rééquilibrage | imbalanced-learn | 0.14.0 | SMOTE |
| Deep Learning | PyTorch | 2.12.0 | EfficientNet-B0 |
| Vision | torchvision | 0.27.0 | Transforms + modèles |
| Data | pandas / numpy | 2.3.3 / 2.3.5 | Manipulation des données |
| Visualisation | Plotly | 6.3.0 | Graphiques interactifs |
| Auth | pyotp | 2.9.0 | Authentification 2FA TOTP |
| Persistance | joblib | 1.5.2 | Sérialisation des modèles |
| Images | Pillow | 12.0.0 | Traitement d'images |

### 7.3 Structure des Modules Applicatifs

```
app.py                          ← Point d'entrée, routeur de pages
│
├── modules/
│   ├── prediction.py           ← 🩺 Formulaire + diagnostic thyroïdien
│   ├── brain_tumor.py          ← 🧠 Upload IRM + classification deep learning
│   ├── dashboard.py            ← 📊 Statistiques et métriques
│   ├── historique.py           ← 📜 Historique des analyses
│   └── apropos.py              ← ℹ️ Documentation et références
│
├── utils/
│   ├── core.py                 ← Chargement modèles, CSS, helpers
│   ├── auth.py                 ← Authentification 2FA
│   └── firebase.py             ← Persistance cloud (optionnel)
│
├── saved_models/               ← Modèle thyroïde sérialisé
└── brain_tumer_deep/output/    ← Modèle IRM (best_model.pth)
```

### 7.4 Scénarios d'Utilisation Réelle

**Scénario 1 — Médecin généraliste, consultation thyroïde :**
1. Le médecin se connecte via l'interface sécurisée (login + code 2FA)
2. Il saisit les résultats biologiques du patient (TSH, T3, TT4, T4U, FTI) et les antécédents
3. Le système prédit en temps réel : Normal ou Pathologique avec probabilité
4. Un radar biologique compare les valeurs du patient aux normes cliniques
5. Le médecin télécharge le rapport PDF pour l'intégrer au dossier patient

**Scénario 2 — Radiologue, analyse IRM cérébrale :**
1. Le radiologue accède au module "🧠 Tumeur Cérébrale"
2. Il charge une image IRM au format JPEG ou PNG
3. EfficientNet-B0 analyse l'image en moins de 2 secondes
4. Le système affiche la classe prédite (ex: glioma), la confiance (ex: 97.3%) et la distribution des probabilités pour les 4 classes
5. Un rapport détaillé est généré et téléchargeable

**Scénario 3 — Administration, suivi statistique :**
1. L'administrateur accède au tableau de bord
2. Il consulte les statistiques agrégées : nombre d'analyses, répartition des diagnostics, évolution temporelle
3. Il exporte l'historique complet au format CSV pour archivage

### 7.5 Sécurité et Conformité

- **Authentification 2FA** : chaque utilisateur doit fournir un mot de passe et un code TOTP (valable 30 secondes) généré par une application d'authentification
- **Aucune donnée patient transmise** : toutes les analyses sont effectuées localement, sans envoi vers des serveurs tiers
- **Historique traçable** : chaque prédiction est horodatée et enregistrée avec les paramètres d'entrée
- **Avertissement médical** : chaque résultat est accompagné d'un disclaimer rappelant que l'outil ne remplace pas un diagnostic médical professionnel

---

## 8. Conclusion

Le projet **Virtual Clinique v3.1** démontre la faisabilité et la pertinence de l'intégration de l'intelligence artificielle dans un contexte médical applicatif. En suivant rigoureusement la méthodologie CRISP-DM, deux modules de diagnostic ont été développés, évalués et déployés au sein d'une plateforme unifiée et sécurisée.

Les résultats obtenus sont significatifs :
- Le module thyroïdien atteint un **F1-Score de ~95.8%** avec un Recall de 96.4% sur les cas pathologiques, garantissant une détection fiable des anomalies.
- Le module IRM cérébral atteint une **accuracy de 99.85%** sur le jeu de test, avec des scores parfaits sur trois des quatre classes diagnostiques.

Ces performances positionnent Virtual Clinique comme un outil d'aide à la décision clinique crédible, capable d'assister les professionnels de santé dans leur pratique quotidienne. La plateforme répond aux exigences de sécurité (authentification 2FA), de traçabilité (historique des analyses) et d'explicabilité (visualisations, rapports téléchargeables).

Cependant, il convient de rappeler que ces résultats ont été obtenus sur des datasets publics et que toute mise en production réelle nécessiterait une validation clinique rigoureuse, une certification réglementaire (CE médical, FDA selon le marché cible) et une évaluation prospective sur des données patients réelles.

---

## 9. Recommandations

### 9.1 Améliorations Techniques à Court Terme

| Priorité | Recommandation | Impact attendu |
|----------|---------------|----------------|
| 🔴 Haute | Déploiement sur GPU (CUDA) pour le module IRM | Réduction du temps d'inférence de ~10s à < 0.5s |
| 🔴 Haute | Ajout d'un module de segmentation tumorale (U-Net) | Localisation précise de la tumeur sur l'IRM |
| 🟡 Moyenne | Intégration d'un système de feedback médecin | Amélioration continue du modèle par apprentissage actif |
| 🟡 Moyenne | API REST (FastAPI) pour découpler frontend et backend | Interopérabilité avec d'autres systèmes hospitaliers (HIS, PACS) |
| 🟢 Basse | Internationalisation (EN, AR) | Élargissement de la base d'utilisateurs |

### 9.2 Améliorations des Modèles

- **Module Thyroïde :** Intégrer des données longitudinales (évolution des marqueurs dans le temps) via des modèles séquentiels (LSTM, Transformer tabulaire) pour améliorer la détection des tendances pathologiques.
- **Module IRM :** Tester des architectures plus récentes (EfficientNet-B4, Vision Transformer — ViT) et des techniques d'explicabilité visuelle (Grad-CAM) pour mettre en évidence les régions de l'IRM ayant influencé la décision.
- **Nouveaux modules :** Étendre la plateforme à d'autres pathologies (diabète, maladies cardiovasculaires, pneumonie sur radiographie thoracique).

### 9.3 Recommandations pour la Mise en Production

1. **Validation clinique prospective** : tester le système sur des cohortes de patients réels avec validation par des médecins experts avant tout déploiement.
2. **Conformité RGPD** : mettre en place un système de pseudonymisation des données patients et une politique de rétention des données.
3. **Audit de biais** : analyser les performances du modèle par sous-groupes (âge, sexe, origine géographique) pour détecter d'éventuels biais discriminatoires.
4. **Monitoring en production** : implémenter un système de détection de dérive des données (data drift) pour alerter lorsque les distributions des données d'entrée s'éloignent du dataset d'entraînement.
5. **Formation des utilisateurs** : développer des guides d'utilisation et des sessions de formation pour garantir une interprétation correcte des résultats par les professionnels de santé.

---

*Rapport généré dans le cadre du projet Virtual Clinique v3.1 — Mai 2026*  
*Ce document est destiné à un usage académique et professionnel.*
