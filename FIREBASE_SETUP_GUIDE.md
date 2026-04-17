# Guide d'Intégration Firebase

## État Actuel
L'intégration Firebase est **partiellement implémentée**. Le code est prêt à utiliser Firebase Firestore pour:
- Stockage des utilisateurs (remplace `users.json`)
- Historique des prédictions (remplace `prediction_history.csv`)
- Authentification Firebase (optionnel)

## Ce qui a été fait
1. ✅ Création de `utils/firebase.py` - wrapper Firestore + Auth
2. ✅ Mise à jour de `utils/auth.py` - utilise Firestore si disponible
3. ✅ Mise à jour de `utils/core.py` - `save_prediction()` utilise Firestore
4. ✅ Mise à jour de `modules/historique.py` - lit depuis Firestore
5. ✅ Création de `firebase_config.json` template

## Prochaines Étapes

### 1. Configurer Firebase
Pour activer Firebase, vous devez:

#### a) Créer un projet Firebase
1. Allez sur [Firebase Console](https://console.firebase.google.com/)
2. Cliquez "Add project" ou sélectionnez un projet existant
3. Notez votre **Project ID**

#### b) Activer Firestore
1. Dans votre projet Firebase, allez dans "Firestore Database"
2. Cliquez "Create database"
3. Choisissez "Start in test mode" (pour le développement)
4. Sélectionnez une région (ex: europe-west1)

#### c) Générer une clé de service
1. Allez dans "Project Settings" (roue dentée)
2. Onglet "Service Accounts"
3. Cliquez "Generate new private key"
4. Téléchargez le fichier JSON

### 2. Configurer le fichier `firebase_config.json`
1. Renommez le fichier téléchargé en `firebase_config.json`
2. Placez-le dans le dossier racine du projet
3. Vérifiez que le `project_id` correspond à votre projet Firebase

### 3. Tester l'intégration
1. Lancez le script de test:
   ```bash
   python test_firebase.py
   ```
2. Si tout fonctionne, vous verrez:
   ```
   Firebase enabled: True
   Config exists: True
   Project ID: votre-project-id
   ```

### 4. Lancer l'application
1. Lancez Streamlit:
   ```bash
   streamlit run app.py
   ```
2. Créez un nouveau compte (les données seront stockées dans Firestore)
3. Faites une prédiction (elle sera sauvegardée dans Firestore)
4. Vérifiez l'historique (données chargées depuis Firestore)

## Structure Firestore

### Collections créées automatiquement:
1. **`users`** - Stocke les comptes utilisateurs
   - Champs: `username`, `password_hash`, `totp_secret`, `totp_verified`, `created_at`, `last_login`

2. **`predictions`** - Historique des prédictions
   - Champs: `id`, `username`, `timestamp`, `prediction`, `probability`, `patient_data`, `created_at`

## Fonctionnalités Avancées (Optionnelles)

### 1. Firebase Authentication (au lieu de l'authentification custom)
- Activer "Authentication" dans Firebase Console
- Mettre à jour `utils/auth.py` pour utiliser Firebase Auth
- Avantages: gestion des sessions, réinitialisation de mot de passe, OAuth

### 2. Stockage de fichiers (images MRI)
- Activer "Storage" dans Firebase Console
- Ajouter le support dans `utils/firebase.py`
- Permettre le téléchargement/consultation d'images médicales

### 3. Analytics Firebase
- Activer "Analytics" dans Firebase Console
- Suivre l'utilisation de l'application
- Dashboard d'utilisation en temps réel

## Dépannage

### Erreur: "Firebase non configuré"
- Vérifiez que `firebase_config.json` existe
- Vérifiez que le `project_id` est correct
- Vérifiez les permissions Firestore (mode test pour le développement)

### Erreur: "Unable to load PEM file"
- Le fichier JSON est corrompu
- Régénérez la clé de service depuis Firebase Console

### Données non affichées dans l'historique
- Vérifiez que Firestore est activé
- Vérifiez les règles de sécurité Firestore (devraient être en mode test)
- Vérifiez que l'utilisateur est connecté

## Avantages de Firebase
- ✅ **Données en temps réel** - synchronisation automatique
- ✅ **Scalabilité** - supporte des millions d'utilisateurs
- ✅ **Sécurité** - règles Firestore configurables
- ✅ **Backup automatique** - pas de perte de données
- ✅ **Multi-utilisateurs** - idéal pour une application médicale collaborative

## Notes Importantes
1. En mode développement, utilisez les règles Firestore en "test mode"
2. Pour la production, configurez des règles de sécurité appropriées
3. Firebase a un plan gratuit généreux (parfait pour le développement)
4. Les données sont chiffrées en transit et au repos