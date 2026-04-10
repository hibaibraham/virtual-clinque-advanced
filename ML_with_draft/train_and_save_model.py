"""
🏥 Script d'entraînement et de sauvegarde du modèle ML
Clinique Virtuelle Intelligente — Diagnostic Thyroïdien

Ce script reproduit exactement le pipeline du notebook thyroid_ml_pipeline.ipynb,
entraîne le meilleur modèle (Random Forest optimisé) et le sauvegarde avec joblib.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

# ── Configuration ──────────────────────────────────────────────────────────────
FILE_PATH = os.path.join(os.path.dirname(__file__), 'thyroid.csv')
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
os.makedirs(SAVE_DIR, exist_ok=True)

# ── 1. Chargement ──────────────────────────────────────────────────────────────
print("📂 Chargement des données...")
df_raw = pd.read_csv(FILE_PATH)
print(f"   Dimensions : {df_raw.shape[0]} lignes × {df_raw.shape[1]} colonnes")

# ── 2. Nettoyage ───────────────────────────────────────────────────────────────
df = df_raw.copy()

# 2.1 Remplacer '?' par NaN
df.replace('?', np.nan, inplace=True)

# 2.2 Nettoyage de la colonne 'class'
def extract_class_label(val):
    if pd.isna(val):
        return np.nan
    match = re.match(r'^([^\[]+)', str(val))
    return match.group(1).strip() if match else val

df['class'] = df['class'].apply(extract_class_label)

# 2.3 Création cible binaire
df['target'] = (df['class'] != '-').astype(int)
print(f"   Cible binaire : {df['target'].mean():.1%} pathologique")

# 2.4 Suppression colonnes inutiles
cols_to_drop = [
    'TBG', 'TBG_measured', 'TSH_measured', 'T3_measured',
    'TT4_measured', 'T4U_measured', 'FTI_measured',
    'referral_source', 'class',
]
df.drop(columns=cols_to_drop, inplace=True)

# 2.5 Conversion des types
bool_cols = [
    'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment',
    'query_hypothyroid', 'query_hyperthyroid', 'lithium',
    'goitre', 'tumor', 'hypopituitary', 'psych'
]
for col in bool_cols:
    df[col] = df[col].map({'t': 1, 'f': 0})

num_cols_raw = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
for col in num_cols_raw:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['sex'] = df['sex'].map({'M': 1, 'F': 0})

# ── 3. Feature Engineering ────────────────────────────────────────────────────
df['TSH_abnormal']  = ((df['TSH']  < 0.4) | (df['TSH']  > 4.0)).astype('Int8')
df['TT4_abnormal']  = ((df['TT4']  < 70)  | (df['TT4']  > 180)).astype('Int8')
df['T3_abnormal']   = ((df['T3']   < 1.2) | (df['T3']   > 3.1)).astype('Int8')
df['FTI_abnormal']  = ((df['FTI']  < 70)  | (df['FTI']  > 180)).astype('Int8')

df['hormone_score'] = (
    df['TSH_abnormal'].fillna(0).astype(int) +
    df['TT4_abnormal'].fillna(0).astype(int) +
    df['T3_abnormal'].fillna(0).astype(int) +
    df['FTI_abnormal'].fillna(0).astype(int)
)

df['T4U_TT4_ratio'] = df['T4U'] / (df['TT4'] + 1e-6)

print("   ✅ Feature engineering terminé")

# ── 4. Préparation ─────────────────────────────────────────────────────────────
TARGET_COL   = 'target'
AUX_TARGETS  = ['TSH_abnormal', 'TT4_abnormal']
EXCLUDE_COLS = [TARGET_COL] + AUX_TARGETS

X = df.drop(columns=EXCLUDE_COLS)
y = df[TARGET_COL]

num_features = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'hormone_score', 'T4U_TT4_ratio']
bin_features = [c for c in X.columns if c not in num_features]

print(f"   Features numériques ({len(num_features)}) : {num_features}")
print(f"   Features binaires   ({len(bin_features)}) : {bin_features}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessor
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

binary_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_features),
    ('bin', binary_transformer,  bin_features),
])

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep  = preprocessor.transform(X_test)

# SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_prep, y_train)
print(f"   Après SMOTE : {X_train_res.shape[0]} échantillons | {y_train_res.mean():.1%} positifs")

# ── 5. Entraînement du modèle optimisé ────────────────────────────────────────
print("\n🤖 Optimisation du Random Forest par RandomizedSearchCV...")
param_dist_rf = {
    'n_estimators'      : [100, 200, 300, 500],
    'max_depth'         : [None, 5, 10, 20, 30],
    'min_samples_split' : [2, 5, 10],
    'min_samples_leaf'  : [1, 2, 4],
    'max_features'      : ['sqrt', 'log2', 0.5],
    'class_weight'      : ['balanced', 'balanced_subsample']
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rs_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_dist_rf,
    n_iter=30,
    scoring='f1_weighted',
    cv=skf,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rs_rf.fit(X_train_res, y_train_res)

best_rf = rs_rf.best_estimator_
y_pred = best_rf.predict(X_test_prep)

test_f1  = f1_score(y_test, y_pred, average='weighted')
test_acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Meilleurs hyperparamètres : {rs_rf.best_params_}")
print(f"   CV F1 (best)  : {rs_rf.best_score_:.4f}")
print(f"   Test F1       : {test_f1:.4f}")
print(f"   Test Accuracy : {test_acc:.4f}")
print()
print(classification_report(y_test, y_pred, zero_division=0))

# ── 6. Sauvegarde ──────────────────────────────────────────────────────────────
print("\n💾 Sauvegarde du modèle et du preprocessor...")

joblib.dump(best_rf, os.path.join(SAVE_DIR, 'model.joblib'))
joblib.dump(preprocessor, os.path.join(SAVE_DIR, 'preprocessor.joblib'))

# Sauvegarder la config des features
feature_config = {
    'num_features': num_features,
    'bin_features': bin_features,
    'all_features': num_features + bin_features,
    'bool_cols': bool_cols,
    'target_col': TARGET_COL,
    'best_params': {k: str(v) if v is not None else None for k, v in rs_rf.best_params_.items()},
    'test_f1': float(test_f1),
    'test_accuracy': float(test_acc),
    'cv_f1': float(rs_rf.best_score_),
    'train_samples': int(X_train_res.shape[0]),
    'test_samples': int(X_test_prep.shape[0]),
    'feature_importances': {
        name: float(imp)
        for name, imp in zip(num_features + bin_features, best_rf.feature_importances_)
    }
}

with open(os.path.join(SAVE_DIR, 'feature_config.json'), 'w', encoding='utf-8') as f:
    json.dump(feature_config, f, indent=2, ensure_ascii=False)

print(f"   ✅ Modèle sauvegardé dans : {SAVE_DIR}/model.joblib")
print(f"   ✅ Preprocessor sauvegardé dans : {SAVE_DIR}/preprocessor.joblib")
print(f"   ✅ Config sauvegardée dans : {SAVE_DIR}/feature_config.json")
print("\n🎉 Terminé avec succès !")
