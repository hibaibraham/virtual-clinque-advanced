"""
Interface Secrétaire - Enregistrement et gestion des patients
"""
import streamlit as st
from utils.patients import create_patient, get_all_patients, search_patients, get_patients_by_status
from utils.core import section_label
import pandas as pd
from datetime import datetime

def render():
    """Affiche l'interface secrétaire."""
    
    username = st.session_state.get("auth_username", "")
    
    st.markdown("""
    <div style='text-align:center;padding:1.5rem;background:rgba(236,72,153,0.05);
                border-radius:12px;border:1px solid rgba(236,72,153,0.2);margin-bottom:2rem;'>
        <div style='font-size:2.5rem;margin-bottom:0.5rem;'>📋</div>
        <h1 style='color:#f1f5f9;margin-bottom:0.3rem;'>Interface Secrétaire</h1>
        <p style='color:#94a3b8;font-size:1rem;'>
        Bienvenue {username} - Gestion administrative des patients
        </p>
    </div>
    """.format(username=username), unsafe_allow_html=True)
    
    # Onglets
    tab1, tab2, tab3, tab4 = st.tabs(["➕ Nouveau Patient", "👥 Liste des Patients", "🔍 Rechercher", "📊 Statistiques"])
    
    # ── Tab 1: Nouveau Patient ─────────────────────────────────────────────
    with tab1:
        section_label("➕ Enregistrement d'un Nouveau Patient")
        
        st.info("📝 Remplissez les informations du patient. Le médecin pourra ensuite compléter le dossier médical.")
        
        with st.form("new_patient_form", clear_on_submit=True):
            st.subheader("Informations Personnelles")
            
            col1, col2 = st.columns(2)
            
            with col1:
                nom = st.text_input("Nom*", placeholder="Nom de famille")
                prenom = st.text_input("Prénom*", placeholder="Prénom")
                date_naissance = st.date_input("Date de Naissance*")
                age = st.number_input("Âge*", min_value=0, max_value=120, value=30)
                sexe = st.selectbox("Sexe*", ["Homme", "Femme", "Autre"])
            
            with col2:
                telephone = st.text_input("Téléphone*", placeholder="06 12 34 56 78")
                email = st.text_input("Email", placeholder="patient@email.com")
                adresse = st.text_area("Adresse", placeholder="123 Rue de la Santé, 75000 Paris")
                profession = st.text_input("Profession", placeholder="Profession du patient")
            
            st.subheader("Informations Médicales Basiques")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                poids = st.number_input("Poids (kg)", min_value=0.0, max_value=300.0, value=70.0, step=0.1)
                taille = st.number_input("Taille (cm)", min_value=0.0, max_value=250.0, value=170.0, step=0.1)
            
            with col2:
                groupe_sanguin = st.selectbox("Groupe Sanguin", ["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
                allergies = st.text_input("Allergies connues", placeholder="Pénicilline, latex...")
            
            with col3:
                antecedents_familiaux = st.text_area("Antécédents Familiaux", 
                                                     placeholder="Diabète, hypertension...",
                                                     height=100)
            
            st.subheader("Motif de Consultation")
            motif = st.text_area("Raison de la visite*", 
                                placeholder="Décrivez brièvement la raison de la consultation...",
                                height=100)
            
            st.markdown("---")
            st.caption("* Champs obligatoires")
            
            submitted = st.form_submit_button("✅ Enregistrer le Patient", use_container_width=True)
            
            if submitted:
                # Validation
                if not nom or not prenom or not age or not sexe or not telephone or not motif:
                    st.error("❌ Veuillez remplir tous les champs obligatoires (*).")
                else:
                    # Préparation des données
                    patient_data = {
                        "nom": nom,
                        "prenom": prenom,
                        "age": age,
                        "sexe": sexe,
                        "telephone": telephone,
                        "email": email if email else "",
                        "adresse": adresse if adresse else "",
                        "profession": profession if profession else "",
                        "date_naissance": date_naissance.isoformat() if date_naissance else "",
                        "poids": poids,
                        "taille": taille,
                        "groupe_sanguin": groupe_sanguin,
                        "antecedents": {
                            "familiaux": antecedents_familiaux if antecedents_familiaux else "",
                            "allergies": allergies if allergies else ""
                        },
                        "motif_consultation": motif,
                        "created_by": username
                    }
                    
                    # Création du patient
                    try:
                        patient_id = create_patient(patient_data)
                        st.success(f"✅ Patient enregistré avec succès !")
                        st.info(f"**ID Patient:** `{patient_id}`")
                        st.info(f"**Nom complet:** {prenom} {nom}")
                        
                        st.markdown("""
                        <div style='padding:1rem;background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.2);
                                    border-radius:8px;margin-top:1rem;'>
                            <h4 style='color:#10b981;margin-bottom:0.5rem;'>📋 Prochaines étapes:</h4>
                            <ol style='color:#94a3b8;margin:0;padding-left:1.2rem;'>
                                <li>Le dossier patient a été créé</li>
                                <li>Le médecin peut maintenant le voir dans "Patients en Attente"</li>
                                <li>Le médecin complétera les informations médicales</li>
                            </ol>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'enregistrement: {str(e)}")
    
    # ── Tab 2: Liste des Patients ─────────────────────────────────────────────
    with tab2:
        section_label("👥 Liste de Tous les Patients")
        
        all_patients = get_all_patients()
        
        if not all_patients:
            st.info("📭 Aucun patient enregistré pour le moment.")
        else:
            st.success(f"👥 {len(all_patients)} patient(s) enregistré(s)")
            
            # Statistiques rapides
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                en_attente = len([p for p in all_patients if p.get('status') == 'en_attente'])
                st.metric("⏳ En Attente", en_attente)
            with col2:
                en_cours = len([p for p in all_patients if p.get('status') == 'en_cours'])
                st.metric("🔄 En Cours", en_cours)
            with col3:
                complets = len([p for p in all_patients if p.get('status') == 'complete'])
                st.metric("✅ Complets", complets)
            with col4:
                today = datetime.now().date()
                today_patients = len([p for p in all_patients 
                                     if p.get('created_at', '').startswith(str(today))])
                st.metric("📅 Aujourd'hui", today_patients)
            
            # Filtres
            st.markdown("---")
            col1, col2 = st.columns([2, 1])
            with col1:
                filter_status = st.multiselect(
                    "Filtrer par statut",
                    options=["en_attente", "en_cours", "complete"],
                    default=["en_attente", "en_cours", "complete"],
                    format_func=lambda x: {
                        "en_attente": "⏳ En Attente",
                        "en_cours": "🔄 En Cours",
                        "complete": "✅ Complet"
                    }[x]
                )
            with col2:
                sort_by = st.selectbox(
                    "Trier par",
                    options=["date_desc", "date_asc", "nom"],
                    format_func=lambda x: {
                        "date_desc": "📅 Plus récent",
                        "date_asc": "📅 Plus ancien",
                        "nom": "🔤 Nom"
                    }[x]
                )
            
            # Filtrer les patients
            filtered_patients = [p for p in all_patients if p.get('status') in filter_status]
            
            # Trier
            if sort_by == "date_desc":
                filtered_patients.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            elif sort_by == "date_asc":
                filtered_patients.sort(key=lambda x: x.get('created_at', ''))
            else:  # nom
                filtered_patients.sort(key=lambda x: x.get('nom', ''))
            
            # Afficher le tableau
            st.markdown("---")
            if filtered_patients:
                patients_data = []
                for patient in filtered_patients:
                    patients_data.append({
                        "ID": patient.get('patient_id', ''),
                        "Nom": patient.get('nom', ''),
                        "Prénom": patient.get('prenom', ''),
                        "Âge": patient.get('age', ''),
                        "Téléphone": patient.get('telephone', ''),
                        "Statut": patient.get('status', '').replace('_', ' ').title(),
                        "Date": patient.get('created_at', '').split('T')[0] if patient.get('created_at') else '',
                        "Motif": patient.get('motif_consultation', '')[:40] + "..." 
                                if len(patient.get('motif_consultation', '')) > 40 
                                else patient.get('motif_consultation', '')
                    })
                
                df = pd.DataFrame(patients_data)
                
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "ID": st.column_config.TextColumn("ID", width="small"),
                        "Nom": st.column_config.TextColumn("Nom", width="small"),
                        "Prénom": st.column_config.TextColumn("Prénom", width="small"),
                        "Âge": st.column_config.NumberColumn("Âge", width="small"),
                        "Téléphone": st.column_config.TextColumn("Téléphone", width="medium"),
                        "Statut": st.column_config.TextColumn("Statut", width="small"),
                        "Date": st.column_config.TextColumn("Date", width="small"),
                        "Motif": st.column_config.TextColumn("Motif", width="large")
                    }
                )
            else:
                st.info("Aucun patient ne correspond aux filtres sélectionnés.")
    
    # ── Tab 3: Rechercher ─────────────────────────────────────────────────
    with tab3:
        section_label("🔍 Recherche de Patients")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("Rechercher un patient", 
                                       placeholder="Nom, prénom, téléphone, ID...",
                                       key="search_sec")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_btn = st.button("🔎 Rechercher", use_container_width=True, key="btn_search_sec")
        
        if search_btn and search_term:
            results = search_patients(search_term)
            if results:
                st.success(f"📊 {len(results)} patient(s) trouvé(s)")
                
                # Afficher les résultats
                for patient in results:
                    status_color = {
                        "en_attente": "rgba(251,191,36,0.1)",
                        "en_cours": "rgba(59,130,246,0.1)",
                        "complete": "rgba(34,197,94,0.1)"
                    }.get(patient.get('status', ''), "rgba(100,100,100,0.1)")
                    
                    status_icon = {
                        "en_attente": "⏳",
                        "en_cours": "🔄",
                        "complete": "✅"
                    }.get(patient.get('status', ''), "❓")
                    
                    with st.expander(f"{status_icon} {patient.get('prenom', '')} {patient.get('nom', '')} - {patient.get('patient_id', '')}", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Informations Personnelles:**")
                            st.text(f"ID: {patient.get('patient_id', '')}")
                            st.text(f"Nom: {patient.get('nom', '')}")
                            st.text(f"Prénom: {patient.get('prenom', '')}")
                            st.text(f"Âge: {patient.get('age', '')}")
                            st.text(f"Sexe: {patient.get('sexe', '')}")
                            st.text(f"Téléphone: {patient.get('telephone', '')}")
                            if patient.get('email'):
                                st.text(f"Email: {patient.get('email', '')}")
                        
                        with col2:
                            st.markdown("**Informations Médicales:**")
                            st.text(f"Statut: {patient.get('status', '').replace('_', ' ').title()}")
                            st.text(f"Motif: {patient.get('motif_consultation', '')}")
                            st.text(f"Créé le: {patient.get('created_at', '').split('T')[0] if patient.get('created_at') else ''}")
                            if patient.get('poids'):
                                st.text(f"Poids: {patient.get('poids', '')} kg")
                            if patient.get('taille'):
                                st.text(f"Taille: {patient.get('taille', '')} cm")
            else:
                st.info("Aucun patient trouvé avec ces critères.")
        elif search_btn:
            st.warning("Veuillez entrer un terme de recherche.")
    
    # ── Tab 4: Statistiques ─────────────────────────────────────────────────
    with tab4:
        section_label("📊 Statistiques et Rapports")
        
        all_patients = get_all_patients()
        
        if not all_patients:
            st.info("📭 Aucune donnée disponible pour les statistiques.")
        else:
            # Statistiques générales
            st.subheader("📈 Vue d'Ensemble")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("👥 Total Patients", len(all_patients))
            
            with col2:
                en_attente = len([p for p in all_patients if p.get('status') == 'en_attente'])
                st.metric("⏳ En Attente", en_attente)
            
            with col3:
                complets = len([p for p in all_patients if p.get('status') == 'complete'])
                st.metric("✅ Complets", complets)
            
            with col4:
                today = datetime.now().date()
                today_patients = len([p for p in all_patients 
                                     if p.get('created_at', '').startswith(str(today))])
                st.metric("📅 Aujourd'hui", today_patients)
            
            st.markdown("---")
            
            # Statistiques par statut
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Répartition par Statut")
                status_counts = {
                    "En Attente": len([p for p in all_patients if p.get('status') == 'en_attente']),
                    "En Cours": len([p for p in all_patients if p.get('status') == 'en_cours']),
                    "Complet": len([p for p in all_patients if p.get('status') == 'complete'])
                }
                
                for status, count in status_counts.items():
                    percentage = (count / len(all_patients) * 100) if len(all_patients) > 0 else 0
                    st.write(f"**{status}:** {count} ({percentage:.1f}%)")
            
            with col2:
                st.subheader("👥 Répartition par Sexe")
                sexe_counts = {}
                for patient in all_patients:
                    sexe = patient.get('sexe', 'Non spécifié')
                    sexe_counts[sexe] = sexe_counts.get(sexe, 0) + 1
                
                for sexe, count in sexe_counts.items():
                    percentage = (count / len(all_patients) * 100) if len(all_patients) > 0 else 0
                    st.write(f"**{sexe}:** {count} ({percentage:.1f}%)")
            
            st.markdown("---")
            
            # Patients récents
            st.subheader("🕐 Derniers Patients Enregistrés")
            recent_patients = sorted(all_patients, 
                                   key=lambda x: x.get('created_at', ''), 
                                   reverse=True)[:5]
            
            for patient in recent_patients:
                status_icon = {
                    "en_attente": "⏳",
                    "en_cours": "🔄",
                    "complete": "✅"
                }.get(patient.get('status', ''), "❓")
                
                date_str = patient.get('created_at', '').split('T')[0] if patient.get('created_at') else ''
                st.write(f"{status_icon} **{patient.get('prenom', '')} {patient.get('nom', '')}** - {date_str}")
