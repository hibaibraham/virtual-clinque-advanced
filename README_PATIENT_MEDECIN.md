# Système Patient/Médecin - MedAI Santé

## 🎯 Nouvelle Organisation
L'application MedAI a été réorganisée en deux interfaces distinctes :

### 1. **Interface Patient (👤)**
- **Accès** : Rôle "patient"
- **Fonctionnalités** :
  - Remplir ses informations personnelles
  - Consulter ses résultats médicaux
  - Voir son historique médical
- **Ne peut pas** : Accéder aux outils de diagnostic (thyroïde, brain cancer)

### 2. **Interface Médecin (👨‍⚕️)**
- **Accès** : Rôle "medecin"
- **Fonctionnalités** :
  - Consulter les dossiers patients
  - Compléter les informations médicales
  - Utiliser les outils de diagnostic :
    - 🦋 Diagnostic Thyroïdien (Random Forest)
    - 🧠 Cancer Cérébral (Deep Learning - en développement)
  - Tableau de bord et historique
- **Peut tout faire** : Accès complet au système

## 🔄 Flux de Travail

1. **Choix du profil** avant connexion (Patient ou Médecin)
2. **Authentification 2FA** (TOTP obligatoire)
3. **Interface adaptée** selon le rôle

### Pour les Patients :
```
Connexion → Mes Infos (remplir) → Attente médecin → Mes Résultats (consultation)
```

### Pour les Médecins :
```
Connexion → Consultation (voir patients) → Diagnostic (thyroïde/brain) → Enregistrement
```

## 🛠️ Installation et Test

### 1. Créer les utilisateurs de test
```bash
python init_roles.py
```

Créera 5 utilisateurs :
- **Patients** : patient1 / patient123, patient2 / patient456
- **Médecins** : docteur1 / medecin123, docteur2 / medecin456, admin / admin123

### 2. Lancer l'application
```bash
streamlit run app.py
```

### 3. Première connexion
1. Choisissez "Patient" ou "Médecin"
2. Connectez-vous avec un compte de test
3. Configurez le 2FA (Google Authenticator)
4. L'interface s'adapte automatiquement

## 📁 Structure des Fichiers

```
├── app.py                    # Application principale avec choix rôle
├── modules/
│   ├── patient.py           # NOUVEAU : Interface patient
│   ├── medecin.py          # Interface médecin (modifié)
│   ├── prediction.py       # Module thyroïde (inchangé)
│   ├── brain_cancer.py    # Module cancer cérébral (inchangé)
│   ├── dashboard.py       # Tableau de bord (inchangé)
│   ├── historique.py      # Historique (inchangé)
│   └── apropos.py         # À propos (inchangé)
├── utils/
│   ├── auth.py            # Authentification avec choix rôle (modifié)
│   ├── patients.py        # Gestion des patients (inchangé)
│   ├── core.py           # Utilitaires (inchangé)
│   └── __init__.py
├── patients.json         # Stockage des patients
├── patients.csv          # Backup CSV
├── users.json           # Utilisateurs avec rôles
└── init_roles.py        # Script d'initialisation
```

## 🔐 Sécurité

- **Séparation stricte** des interfaces
- **2FA obligatoire** (TOTP)
- **Hash bcrypt** pour les mots de passe
- **Patient** ne voit que ses propres données
- **Médecin** voit tous les dossiers

## 🚀 Pour Commencer

1. **Test Patient** :
   - Connectez-vous en tant que `patient1` / `patient123`
   - Remplissez vos informations
   - Attendez qu'un médecin complète votre dossier

2. **Test Médecin** :
   - Connectez-vous en tant que `docteur1` / `medecin123`
   - Consultez les patients en attente
   - Utilisez les outils thyroïde/brain cancer
   - Complétez les dossiers

## ⚠️ Notes Importantes

- Les patients **ne voient pas** les modules thyroïde/brain cancer
- Les médecins ont **accès complet** à tous les outils
- Les données sont stockées localement (JSON/CSV)
- Pensez aux sauvegardes régulières
- Le module brain cancer est **en développement** (placeholder)