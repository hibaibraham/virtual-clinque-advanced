"""
Gestion des rendez-vous - Stockage et récupération
Supporte MongoDB avec fallback vers JSON
"""
import os
import json
from datetime import datetime, timedelta
from utils.database import get_appointments_collection, is_mongodb_available

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPOINTMENTS_PATH = os.path.join(BASE_DIR, 'appointments.json')

# ── Fonctions JSON (fallback) ────────────────────────────────────────────────

def _load_appointments() -> dict:
    """Charge les rendez-vous depuis le fichier JSON."""
    if not os.path.exists(APPOINTMENTS_PATH):
        return {}
    with open(APPOINTMENTS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def _save_appointments(appointments: dict):
    """Sauvegarde les rendez-vous dans le fichier JSON."""
    with open(APPOINTMENTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(appointments, f, indent=2, ensure_ascii=False)

# ── Fonctions principales ────────────────────────────────────────────────────

def create_appointment(appointment_data: dict) -> str:
    """
    Crée un nouveau rendez-vous.
    Retourne l'ID du rendez-vous.
    """
    # Générer un ID unique
    appointment_id = f"RDV{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Ajouter des métadonnées
    appointment_record = {
        "appointment_id": appointment_id,
        "created_at": datetime.now().isoformat(),
        "status": "planifie",  # planifie, confirme, annule, termine
        **appointment_data
    }
    
    # Essayer MongoDB d'abord
    if is_mongodb_available():
        collection = get_appointments_collection()
        if collection is not None:
            try:
                collection.insert_one(appointment_record.copy())
                return appointment_id
            except Exception as e:
                print(f"⚠️ Erreur MongoDB, fallback vers JSON: {e}")
    
    # Fallback vers JSON
    appointments = _load_appointments()
    appointments[appointment_id] = appointment_record
    _save_appointments(appointments)
    
    return appointment_id


def get_appointment(appointment_id: str) -> dict:
    """Récupère les informations d'un rendez-vous par son ID."""
    # Essayer MongoDB d'abord
    if is_mongodb_available():
        collection = get_appointments_collection()
        if collection is not None:
            try:
                appointment = collection.find_one({"appointment_id": appointment_id})
                if appointment:
                    appointment.pop('_id', None)
                    return appointment
            except Exception as e:
                print(f"⚠️ Erreur MongoDB, fallback vers JSON: {e}")
    
    # Fallback vers JSON
    appointments = _load_appointments()
    return appointments.get(appointment_id, {})


def update_appointment(appointment_id: str, updates: dict):
    """Met à jour les informations d'un rendez-vous."""
    updates["updated_at"] = datetime.now().isoformat()
    
    # Essayer MongoDB d'abord
    if is_mongodb_available():
        collection = get_appointments_collection()
        if collection is not None:
            try:
                collection.update_one(
                    {"appointment_id": appointment_id},
                    {"$set": updates}
                )
                return
            except Exception as e:
                print(f"⚠️ Erreur MongoDB, fallback vers JSON: {e}")
    
    # Fallback vers JSON
    appointments = _load_appointments()
    if appointment_id in appointments:
        appointments[appointment_id].update(updates)
        _save_appointments(appointments)


def delete_appointment(appointment_id: str):
    """Supprime un rendez-vous."""
    # Essayer MongoDB d'abord
    if is_mongodb_available():
        collection = get_appointments_collection()
        if collection is not None:
            try:
                collection.delete_one({"appointment_id": appointment_id})
                return
            except Exception as e:
                print(f"⚠️ Erreur MongoDB, fallback vers JSON: {e}")
    
    # Fallback vers JSON
    appointments = _load_appointments()
    if appointment_id in appointments:
        del appointments[appointment_id]
        _save_appointments(appointments)


def get_all_appointments() -> list:
    """Récupère tous les rendez-vous."""
    # Essayer MongoDB d'abord
    if is_mongodb_available():
        collection = get_appointments_collection()
        if collection is not None:
            try:
                appointments = list(collection.find({}))
                for appointment in appointments:
                    appointment.pop('_id', None)
                return appointments
            except Exception as e:
                print(f"⚠️ Erreur MongoDB, fallback vers JSON: {e}")
    
    # Fallback vers JSON
    appointments = _load_appointments()
    return list(appointments.values())


def get_appointments_by_date(date: str) -> list:
    """Récupère les rendez-vous pour une date donnée (format: YYYY-MM-DD)."""
    # Essayer MongoDB d'abord
    if is_mongodb_available():
        collection = get_appointments_collection()
        if collection is not None:
            try:
                appointments = list(collection.find({"date": date}))
                for appointment in appointments:
                    appointment.pop('_id', None)
                return appointments
            except Exception as e:
                print(f"⚠️ Erreur MongoDB, fallback vers JSON: {e}")
    
    # Fallback vers JSON
    appointments = _load_appointments()
    return [apt for apt in appointments.values() if apt.get("date") == date]


def get_appointments_by_patient(patient_id: str) -> list:
    """Récupère tous les rendez-vous d'un patient."""
    # Essayer MongoDB d'abord
    if is_mongodb_available():
        collection = get_appointments_collection()
        if collection is not None:
            try:
                appointments = list(collection.find({"patient_id": patient_id}))
                for appointment in appointments:
                    appointment.pop('_id', None)
                return appointments
            except Exception as e:
                print(f"⚠️ Erreur MongoDB, fallback vers JSON: {e}")
    
    # Fallback vers JSON
    appointments = _load_appointments()
    return [apt for apt in appointments.values() if apt.get("patient_id") == patient_id]


def get_appointments_by_status(status: str) -> list:
    """Récupère les rendez-vous par statut."""
    # Essayer MongoDB d'abord
    if is_mongodb_available():
        collection = get_appointments_collection()
        if collection is not None:
            try:
                appointments = list(collection.find({"status": status}))
                for appointment in appointments:
                    appointment.pop('_id', None)
                return appointments
            except Exception as e:
                print(f"⚠️ Erreur MongoDB, fallback vers JSON: {e}")
    
    # Fallback vers JSON
    appointments = _load_appointments()
    return [apt for apt in appointments.values() if apt.get("status") == status]


def get_available_slots(date: str, duration_minutes: int = 30) -> list:
    """
    Retourne les créneaux disponibles pour une date donnée.
    Horaires: 9h-12h et 14h-18h
    """
    # Récupérer les rendez-vous existants pour cette date
    existing_appointments = get_appointments_by_date(date)
    
    # Créer les créneaux de la journée
    slots = []
    
    # Matin: 9h-12h
    current_time = datetime.strptime("09:00", "%H:%M")
    end_morning = datetime.strptime("12:00", "%H:%M")
    
    while current_time < end_morning:
        time_str = current_time.strftime("%H:%M")
        # Vérifier si le créneau est libre
        is_available = not any(
            apt.get("heure") == time_str and apt.get("status") != "annule"
            for apt in existing_appointments
        )
        slots.append({
            "time": time_str,
            "available": is_available,
            "period": "matin"
        })
        current_time += timedelta(minutes=duration_minutes)
    
    # Après-midi: 14h-18h
    current_time = datetime.strptime("14:00", "%H:%M")
    end_afternoon = datetime.strptime("18:00", "%H:%M")
    
    while current_time < end_afternoon:
        time_str = current_time.strftime("%H:%M")
        is_available = not any(
            apt.get("heure") == time_str and apt.get("status") != "annule"
            for apt in existing_appointments
        )
        slots.append({
            "time": time_str,
            "available": is_available,
            "period": "apres-midi"
        })
        current_time += timedelta(minutes=duration_minutes)
    
    return slots


def is_slot_available(date: str, time: str) -> bool:
    """Vérifie si un créneau est disponible."""
    appointments = get_appointments_by_date(date)
    return not any(
        apt.get("heure") == time and apt.get("status") != "annule"
        for apt in appointments
    )


def get_upcoming_appointments(days: int = 7) -> list:
    """Récupère les rendez-vous à venir dans les X prochains jours."""
    today = datetime.now().date()
    future_date = today + timedelta(days=days)
    
    all_appointments = get_all_appointments()
    
    upcoming = []
    for apt in all_appointments:
        try:
            apt_date = datetime.fromisoformat(apt.get("date", "")).date()
            if today <= apt_date <= future_date and apt.get("status") != "annule":
                upcoming.append(apt)
        except:
            continue
    
    # Trier par date et heure
    upcoming.sort(key=lambda x: (x.get("date", ""), x.get("heure", "")))
    
    return upcoming


def get_today_appointments() -> list:
    """Récupère les rendez-vous du jour."""
    today = datetime.now().date().isoformat()
    return get_appointments_by_date(today)


def count_appointments_by_status() -> dict:
    """Compte les rendez-vous par statut."""
    all_appointments = get_all_appointments()
    
    counts = {
        "planifie": 0,
        "confirme": 0,
        "annule": 0,
        "termine": 0
    }
    
    for apt in all_appointments:
        status = apt.get("status", "planifie")
        if status in counts:
            counts[status] += 1
    
    return counts
