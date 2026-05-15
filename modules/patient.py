"""
Interface Patient - Saisie des informations personnelles et consultation des résultats
"""
import streamlit as st
from utils.patients import create_patient, get_patient, search_patients, update_patient
from utils.core import section_label, DARK_LAYOUT
import pandas as pd

def render():
    """Affiche l'interface patient."""
    
    username = st.session_state.get("auth_username", "")
    
    st.markdown("""
    <div style='text-align:center;padding:1.5rem;background:rgba(34,197,94,0.05);
                border-radius:12px;border:1px solid rgba(34,197,94,0.2);margin-bottom:2rem;'>
        <div style='font-size:2.5rem;margin-bottom:0.5rem;'>👤</div>
        <h1 style='color:#f1f5f9;margin-bottom:0.3rem;'>Interface Patient</h1>
        <p style='color:#94a3b8;font-size:1rem;'>
        Bienvenue {username} - Gérez vos informations de santé
        </p>
    </div>
    """.format(username=username), unsafe_allow_html=True)
    
    # Onglets
    tab1, tab2, tab3 = st.tabs(["📝 Mes Informations", "🔍 Mes Résultats", "📋 Mon Historique"])
    
    # ── Tab 1: Mes Informations ─────────────────────────────────────────────
    with tab1:
        section_label("📝 Mes Informations Personnelles")
        
        # Vérifier si le patient a déjà un dossier
        patient_data = search_patients(username)
        existing_patient = patient_data[0] if patient_data else None
        
        if existing_patient:
            st.success("✅ Vous avez déjà un dossier patient.")
            
            with st.expander("📄 Voir mes informations actuelles", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Informations Personnelles:**")
                    st.text(f"Nom: {existing_patient.get('nom', 'N/A')}")
                    st.text(f"Prénom: {existing_patient.get('prenom', 'N/A')}")
                    st.text(f"Âge: {existing_patient.get('age', 'N/A')}")
                    st.text(f"Sexe: {existing_patient.get('sexe', 'N/A')}")
                    st.text(f"Téléphone: {existing_patient.get('telephone', 'N/A')}")
                    
                with col2:
                    st.markdown("**Informations de Santé:**")
                    if existing_patient.get('poids'):
                        st.text(f"Poids: {existing_patient.get('poids', '')} kg")
                    if existing_patient.get('taille'):
                        st.text(f"Taille: {existing_patient.get('taille', '')} cm")
                    
                    antecedents = existing_patient.get('antecedents', {})
                    if antecedents.get('allergies'):
                        st.text(f"Allergies: {antecedents.get('allergies', '')}")
                    
                    st.text(f"Statut: {existing_patient.get('status', '').replace('_', ' ').title()}")
            
            st.info("ℹ️ Pour modifier vos informations, contactez l'administration.")
        
        else:
            st.info("📝 Vous n'avez pas encore de dossier patient. Veuillez remplir le formulaire ci-dessous.")
            
            with st.form("patient_info_form", clear_on_submit=True):
                st.subheader("Informations Personnelles")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    nom = st.text_input("Nom*", placeholder="Votre nom")
                    prenom = st.text_input("Prénom*", placeholder="Votre prénom")
                    age = st.number_input("Âge*", min_value=0, max_value=120, value=30)
                    sexe = st.selectbox("Sexe*", ["Homme", "Femme", "Autre"])
                    date_naissance = st.date_input("Date de Naissance")
                
                with col2:
                    telephone = st.text_input("Téléphone*", placeholder="06 12 34 56 78")
                    email = st.text_input("Email*", placeholder="votre@email.com")
                    adresse = st.text_area("Adresse", placeholder="123 Rue de la Santé, 75000 Paris")
                    profession = st.text_input("Profession", placeholder="Votre profession")
                
                st.subheader("Informations de Santé Basiques")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    poids = st.number_input("Poids (kg)", min_value=0.0, max_value=300.0, value=70.0)
                    taille = st.number_input("Taille (cm)", min_value=0.0, max_value=250.0, value=170.0)
                    groupe_sanguin = st.selectbox("Groupe Sanguin", ["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
                
                with col2:
                    st.markdown("**Antécédents**")
                    antecedents_familiaux = st.text_area("Antécédents Familiaux", 
                                                        placeholder="Diabète, hypertension, cancer dans la famille...")
                    antecedents_personnels = st.text_area("Antécédents Personnels", 
                                                         placeholder="Opérations, maladies chroniques...")
                
                st.subheader("Informations Complémentaires")
                
                allergies = st.text_area("Allergies connues", placeholder="Pénicilline, latex, aliments...")
                medicaments = st.text_area("Médicaments en cours", 
                                          placeholder="Liste des médicaments que vous prenez régulièrement")
                habitudes = st.selectbox("Tabagisme", ["Non-fumeur", "Ancien fumeur", "Fumeur occasionnel", "Fumeur régulier"])
                alcool = st.selectbox("Consommation d'alcool", ["Jamais", "Occasionnelle", "Régulière", "Excessive"])
                activite_physique = st.selectbox("Activité physique", ["Sédentaire", "Légère", "Modérée", "Intense"])
                
                motif = st.text_area("Raison de la consultation (optionnel)", 
                                    placeholder="Décrivez brièvement la raison de votre venue...")
                
                st.markdown("---")
                st.caption("* Champs obligatoires")
                
                submitted = st.form_submit_button("✅ Enregistrer mes Informations", use_container_width=True)
                
                if submitted:
                    # Validation
                    if not nom or not prenom or not age or not sexe or not telephone or not email:
                        st.error("Veuillez remplir tous les champs obligatoires (*).")
                    else:
                        # Préparation des données
                        patient_data = {
                            "nom": nom,
                            "prenom": prenom,
                            "age": age,
                            "sexe": sexe,
                            "telephone": telephone,
                            "email": email,
                            "adresse": adresse if adresse else "",
                            "profession": profession if profession else "",
                            "date_naissance": date_naissance.isoformat() if date_naissance else "",
                            "poids": poids,
                            "taille": taille,
                            "groupe_sanguin": groupe_sanguin,
                            "antecedents": {
                                "familiaux": antecedents_familiaux,
                                "personnels": antecedents_personnels,
                                "allergies": allergies,
                                "medicaments": medicaments
                            },
                            "habitudes_vie": {
                                "tabagisme": habitudes,
                                "alcool": alcool,
                                "activite_physique": activite_physique
                            },
                            "motif_consultation": motif if motif else "Création de dossier",
                            "username": username,  # Lier au compte utilisateur
                            "created_by": "patient"
                        }
                        
                        # Création du patient
                        try:
                            patient_id = create_patient(patient_data)
                            st.success(f"✅ Vos informations ont été enregistrées avec succès !")
                            st.info(f"**Votre ID Patient:** `{patient_id}`")
                            st.info(f"**Nom complet:** {prenom} {nom}")
                            
                            st.markdown("""
                            <div style='padding:1rem;background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.2);
                                        border-radius:8px;margin-top:1rem;'>
                                <h4 style='color:#10b981;margin-bottom:0.5rem;'>📋 Prochaines étapes:</h4>
                                <ol style='color:#94a3b8;margin:0;padding-left:1.2rem;'>
                                    <li>Votre dossier a été créé avec succès</li>
                                    <li>Un médecin examinera vos informations</li>
                                    <li>Vous recevrez les résultats de vos examens ici</li>
                                </ol>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Erreur lors de l'enregistrement: {str(e)}")
    
    # ── Tab 2: Mes Résultats ─────────────────────────────────────────────
    with tab2:
        section_label("🔍 Mes Résultats Médicaux")
        
        # Rechercher le dossier du patient
        patient_data = search_patients(username)
        
        if not patient_data:
            st.info("📭 Vous n'avez pas encore de dossier patient. Veuillez d'abord remplir vos informations.")
        else:
            patient = patient_data[0]
            patient_id = patient.get("patient_id", "")
            
            st.success(f"📁 Dossier trouvé : {patient.get('prenom', '')} {patient.get('nom', '')}")
            
            # Vérifier si le médecin a complété les informations
            if patient.get("status") == "complete" and patient.get("medical_data"):
                medical_data = patient.get("medical_data", {})
                
                st.balloons()
                st.success("🎉 Votre dossier médical est complet !")
                
                # Afficher les résultats
                with st.expander("🏥 Examen Clinique", expanded=True):
                    examen = medical_data.get("examen_clinique", {})
                    if examen:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if examen.get("tension_arterielle"):
                                st.metric("Tension Artérielle", examen.get("tension_arterielle"))
                            if examen.get("frequence_cardiaque"):
                                st.metric("Fréquence Cardiaque", f"{examen.get('frequence_cardiaque')} bpm")
                        
                        with col2:
                            if examen.get("temperature"):
                                st.metric("Température", f"{examen.get('temperature')}°C")
                            if examen.get("saturation_o2"):
                                st.metric("Saturation O₂", f"{examen.get('saturation_o2')}%")
                
                with st.expander("🔬 Résultats de Laboratoire", expanded=False):
                    labo = medical_data.get("laboratoire", {})
                    
                    if labo.get("hematologie"):
                        st.subheader("Hématologie")
                        hemato = labo.get("hematologie", {})
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if hemato.get("hemoglobine"):
                                st.metric("Hémoglobine", f"{hemato.get('hemoglobine')} g/dL")
                        with col2:
                            if hemato.get("leucocytes"):
                                st.metric("Leucocytes", f"{hemato.get('leucocytes')} 10³/µL")
                        with col3:
                            if hemato.get("plaquettes"):
                                st.metric("Plaquettes", f"{hemato.get('plaquettes')} 10³/µL")
                    
                    if labo.get("biochimie"):
                        st.subheader("Biochimie")
                        bio = labo.get("biochimie", {})
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if bio.get("glycemie"):
                                st.metric("Glycémie", f"{bio.get('glycemie')} mg/dL")
                        with col2:
                            if bio.get("creatinine"):
                                st.metric("Créatinine", f"{bio.get('creatinine')} mg/dL")
                        with col3:
                            if bio.get("cholesterol"):
                                st.metric("Cholestérol", f"{bio.get('cholesterol')} mg/dL")
                
                with st.expander("📋 Diagnostic et Traitement", expanded=False):
                    if medical_data.get("diagnostic"):
                        st.subheader("Diagnostic")
                        st.info(medical_data.get("diagnostic"))
                    
                    if medical_data.get("traitement"):
                        st.subheader("Traitement Prescrit")
                        st.warning(medical_data.get("traitement"))
                    
                    if medical_data.get("recommandations"):
                        st.subheader("Recommandations")
                        st.success(medical_data.get("recommandations"))
                
                # Date de la consultation
                if patient.get("updated_at"):
                    st.caption(f"📅 Dernière mise à jour: {patient.get('updated_at', '').replace('T', ' ')}")
                
            elif patient.get("status") == "en_cours":
                st.warning("⏳ Votre dossier est en cours d'examen par un médecin.")
                st.info("Les résultats seront disponibles ici une fois l'examen terminé.")
            else:
                st.info("📋 Votre dossier a été créé et est en attente d'examen par un médecin.")
    
    # ── Tab 3: Mon Historique ─────────────────────────────────────────────
    with tab3:
        section_label("📋 Mon Historique Médical")
        
        patient_data = search_patients(username)
        
        if not patient_data:
            st.info("📭 Aucun historique disponible. Veuillez d'abord créer votre dossier.")
        else:
            patient = patient_data[0]
            
            # Informations de base
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Âge", patient.get("age", "N/A"))
            with col2:
                if patient.get("poids") and patient.get("taille"):
                    taille_m = patient.get("taille", 0) / 100
                    imc = patient.get("poids", 0) / (taille_m ** 2) if taille_m > 0 else 0
                    st.metric("IMC", f"{imc:.1f}")
            with col3:
                st.metric("Statut", patient.get("status", "").replace("_", " ").title())
            
            # Timeline
            st.markdown("---")
            st.subheader("📅 Chronologie")
            
            timeline_items = []
            
            if patient.get("created_at"):
                timeline_items.append({
                    "date": patient.get("created_at").split("T")[0],
                    "event": "Création du dossier",
                    "description": "Dossier patient créé"
                })
            
            if patient.get("medical_data") and patient.get("medical_data", {}).get("medecin"):
                timeline_items.append({
                    "date": patient.get("updated_at", patient.get("created_at", "")).split("T")[0],
                    "event": "Consultation médicale",
                    "description": f"Examen par Dr. {patient.get('medical_data', {}).get('medecin', '')}"
                })
            
            if timeline_items:
                for item in timeline_items:
                    with st.container():
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.markdown(f"**{item['date']}**")
                        with col2:
                            st.markdown(f"**{item['event']}**")
                            st.caption(item['description'])
                        st.markdown("---")
            
            # Documents (placeholder pour futur)
            st.subheader("📎 Documents")
            st.info("Fonctionnalité à venir : téléchargement des comptes-rendus et ordonnances.")