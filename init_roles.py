"""
Script d'initialisation des rôles utilisateurs
Crée des comptes de test pour secrétaire et médecin
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.auth import create_user, user_exists
import bcrypt

def init_test_users():
    """Crée des utilisateurs de test avec différents rôles."""
    
    test_users = [
        {"username": "patient1", "password": "patient123", "role": "patient"},
        {"username": "patient2", "password": "patient456", "role": "patient"},
        {"username": "docteur1", "password": "medecin123", "role": "medecin"},
        {"username": "docteur2", "password": "medecin456", "role": "medecin"},
        {"username": "admin", "password": "admin123", "role": "medecin"},
    ]
    
    print("🔧 Initialisation des utilisateurs de test...")
    print("-" * 50)
    
    created_count = 0
    for user in test_users:
        username = user["username"]
        password = user["password"]
        role = user["role"]
        
        if user_exists(username):
            print(f"❌ L'utilisateur '{username}' existe déjà.")
        else:
            try:
                secret = create_user(username, password, role)
                print(f"✅ Créé: {username} (Rôle: {role})")
                print(f"   Mot de passe: {password}")
                print(f"   Secret TOTP: {secret}")
                created_count += 1
            except Exception as e:
                print(f"❌ Erreur pour {username}: {str(e)}")
    
    print("-" * 50)
    print(f"📊 {created_count} utilisateur(s) créé(s) avec succès.")
    print("\n🔐 Informations de connexion:")
    print("Patients:")
    print("  - patient1 / patient123")
    print("  - patient2 / patient456")
    print("\nMédecins:")
    print("  - docteur1 / medecin123")
    print("  - docteur2 / medecin456")
    print("  - admin / admin123")
    print("\n⚠️ Note: Vous devrez configurer le 2FA (TOTP) à la première connexion.")

if __name__ == "__main__":
    init_test_users()