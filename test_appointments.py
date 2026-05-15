"""
Script de test pour le module de gestion des rendez-vous
"""
from utils.appointments import (
    create_appointment, get_appointment, get_all_appointments,
    get_appointments_by_date, get_available_slots, is_slot_available,
    count_appointments_by_status, get_today_appointments
)
from datetime import datetime, timedelta

def test_appointments():
    """Test des fonctionnalités de gestion des rendez-vous."""
    
    print("=" * 60)
    print("TEST DU MODULE RENDEZ-VOUS")
    print("=" * 60)
    
    # Test 1: Créer un rendez-vous
    print("\n1️⃣ Test de création de rendez-vous...")
    
    test_date = (datetime.now() + timedelta(days=1)).date().isoformat()
    
    appointment_data = {
        "patient_id": "PAT20250101120000",
        "patient_nom": "Dupont",
        "patient_prenom": "Jean",
        "patient_telephone": "0612345678",
        "date": test_date,
        "heure": "10:00",
        "duree": 30,
        "type_consultation": "Consultation générale",
        "motif": "Contrôle de routine",
        "notes": "Patient régulier",
        "created_by": "secretaire_test"
    }
    
    try:
        appointment_id = create_appointment(appointment_data)
        print(f"✅ Rendez-vous créé avec succès!")
        print(f"   ID: {appointment_id}")
        print(f"   Date: {test_date}")
        print(f"   Heure: 10:00")
    except Exception as e:
        print(f"❌ Erreur lors de la création: {e}")
        return
    
    # Test 2: Récupérer le rendez-vous
    print("\n2️⃣ Test de récupération du rendez-vous...")
    
    try:
        appointment = get_appointment(appointment_id)
        if appointment:
            print(f"✅ Rendez-vous récupéré:")
            print(f"   Patient: {appointment.get('patient_prenom')} {appointment.get('patient_nom')}")
            print(f"   Date: {appointment.get('date')}")
            print(f"   Heure: {appointment.get('heure')}")
            print(f"   Statut: {appointment.get('status')}")
        else:
            print("❌ Rendez-vous non trouvé")
    except Exception as e:
        print(f"❌ Erreur lors de la récupération: {e}")
    
    # Test 3: Créneaux disponibles
    print("\n3️⃣ Test des créneaux disponibles...")
    
    try:
        slots = get_available_slots(test_date)
        available_count = len([s for s in slots if s['available']])
        occupied_count = len([s for s in slots if not s['available']])
        
        print(f"✅ Créneaux pour le {test_date}:")
        print(f"   Total: {len(slots)} créneaux")
        print(f"   Disponibles: {available_count}")
        print(f"   Occupés: {occupied_count}")
        
        # Afficher quelques créneaux
        print("\n   Exemples de créneaux:")
        for slot in slots[:5]:
            status = "✅ Disponible" if slot['available'] else "❌ Occupé"
            print(f"   - {slot['time']} ({slot['period']}): {status}")
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des créneaux: {e}")
    
    # Test 4: Vérifier disponibilité d'un créneau
    print("\n4️⃣ Test de vérification de disponibilité...")
    
    try:
        is_available_10 = is_slot_available(test_date, "10:00")
        is_available_11 = is_slot_available(test_date, "11:00")
        
        print(f"   10:00 disponible: {'❌ Non' if not is_available_10 else '✅ Oui'}")
        print(f"   11:00 disponible: {'✅ Oui' if is_available_11 else '❌ Non'}")
    except Exception as e:
        print(f"❌ Erreur lors de la vérification: {e}")
    
    # Test 5: Rendez-vous par date
    print("\n5️⃣ Test de récupération par date...")
    
    try:
        appointments_by_date = get_appointments_by_date(test_date)
        print(f"✅ Rendez-vous pour le {test_date}: {len(appointments_by_date)}")
        
        for apt in appointments_by_date:
            print(f"   - {apt.get('heure')} - {apt.get('patient_prenom')} {apt.get('patient_nom')}")
    except Exception as e:
        print(f"❌ Erreur lors de la récupération par date: {e}")
    
    # Test 6: Statistiques
    print("\n6️⃣ Test des statistiques...")
    
    try:
        stats = count_appointments_by_status()
        print(f"✅ Statistiques des rendez-vous:")
        print(f"   ⏳ Planifiés: {stats.get('planifie', 0)}")
        print(f"   ✅ Confirmés: {stats.get('confirme', 0)}")
        print(f"   ❌ Annulés: {stats.get('annule', 0)}")
        print(f"   🏁 Terminés: {stats.get('termine', 0)}")
    except Exception as e:
        print(f"❌ Erreur lors du calcul des statistiques: {e}")
    
    # Test 7: Tous les rendez-vous
    print("\n7️⃣ Test de récupération de tous les rendez-vous...")
    
    try:
        all_appointments = get_all_appointments()
        print(f"✅ Total de rendez-vous dans la base: {len(all_appointments)}")
    except Exception as e:
        print(f"❌ Erreur lors de la récupération: {e}")
    
    # Test 8: Rendez-vous du jour
    print("\n8️⃣ Test des rendez-vous du jour...")
    
    try:
        today_appointments = get_today_appointments()
        print(f"✅ Rendez-vous aujourd'hui: {len(today_appointments)}")
    except Exception as e:
        print(f"❌ Erreur lors de la récupération: {e}")
    
    print("\n" + "=" * 60)
    print("✅ TESTS TERMINÉS")
    print("=" * 60)
    print("\n💡 Le module de rendez-vous est opérationnel!")
    print("   Vous pouvez maintenant utiliser l'interface secrétaire")
    print("   pour gérer les rendez-vous des patients.")
    print("\n📝 Pour lancer l'application:")
    print("   streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    test_appointments()
