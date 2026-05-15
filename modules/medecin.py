"""
Interface Médecin - Complétion des informations médicales professionnelles
"""
import streamlit as st
from utils.patients import get_patients_by_status, get_patient, update_patient, search_patients, get_all_patients
from utils.core import section_label, DARK_LAYOUT, page_header
import pandas as pd

def render():
    """Affiche l'interface médecin."""
    
    st.markdown("""
    <div style='text-align:center;padding:1.5rem;background:rgba(139,92,246,0.05);
                border-radius:12px;border:1px solid rgba(139,92,246,0.2);margin-bottom:2rem;'>
        <div style='font-size:2.5rem;margin-bottom:0.5rem;'>👨‍⚕️</div>
        <h1 style='color:#f1f5f9;margin-bottom:0.3rem;'>Interface Médecin</h1>
        <p style='color:#94a3b8;font-size:1rem;'>
        Complétion des informations médicales professionnelles
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Onglets
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📋 Patients en Attente", "🔍 Rechercher Patient", "📊 Dossiers Complets", "👥 Tous les Patients", "🦋 Thyroïde", "🧠 Cancer Cérébral", "🩸 PTDM"])
    
    # ── Tab 1: Patients en Attente ─────────────────────────────────────────────
    with tab1:
        section_label("📋 Patients en Attente de Complétion Médicale")
        
        pending_patients = get_patients_by_status("en_attente")
        
        if not pending_patients:
            st.success("🎉 Tous les dossiers patients sont complets !")
            st.info("Aucun patient n'attend de complétion médicale.")
        else:
            st.info(f"👥 {len(pending_patients)} patient(s) en attente de complétion")
            
            # Sélection du patient
            patient_options = {f"{p.get('prenom', '')} {p.get('nom', '')} (ID: {p.get('patient_id', '')})": p.get('patient_id', '') 
                              for p in pending_patients}
            
            selected_patient_label = st.selectbox(
                "Sélectionner un patient à compléter",
                list(patient_options.keys()),
                key="select_patient"
            )
            
            if selected_patient_label:
                patient_id = patient_options[selected_patient_label]
                patient = get_patient(patient_id)
                
                if patient:
                    # Afficher les informations existantes
                    with st.expander("📄 Informations Patient (Saisies par la Secrétaire)", expanded=True):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Informations Personnelles:**")
                            st.text(f"Nom: {patient.get('nom', 'N/A')}")
                            st.text(f"Prénom: {patient.get('prenom', 'N/A')}")
                            st.text(f"Âge: {patient.get('age', 'N/A')}")
                            st.text(f"Sexe: {patient.get('sexe', 'N/A')}")
                            st.text(f"Téléphone: {patient.get('telephone', 'N/A')}")
                            
                        with col2:
                            st.markdown("**Informations Basiques:**")
                            st.text(f"Motif: {patient.get('motif_consultation', 'N/A')}")
                            if patient.get('poids'):
                                st.text(f"Poids: {patient.get('poids', '')} kg")
                            if patient.get('taille'):
                                st.text(f"Taille: {patient.get('taille', '')} cm")
                            
                            antecedents = patient.get('antecedents', {})
                            if antecedents.get('allergies'):
                                st.text(f"Allergies: {antecedents.get('allergies', '')}")
                    
                    # Formulaire de complétion médicale
                    st.markdown("---")
                    section_label("🏥 Complétion des Informations Médicales")
                    
                    with st.form("medical_form"):
                        st.subheader("Examen Clinique")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            tension_arterielle = st.text_input("Tension Artérielle", placeholder="120/80")
                            frequence_cardiaque = st.number_input("Fréquence Cardiaque (bpm)", min_value=0, max_value=200, value=72)
                            temperature = st.number_input("Température (°C)", min_value=30.0, max_value=45.0, value=36.6, step=0.1)
                            saturation_o2 = st.number_input("Saturation O₂ (%)", min_value=0, max_value=100, value=98)
                        
                        with col2:
                            poids = st.number_input("Poids confirmé (kg)", min_value=0.0, max_value=300.0, 
                                                   value=float(patient.get('poids', 70.0)))
                            taille = st.number_input("Taille confirmée (cm)", min_value=0.0, max_value=250.0, 
                                                    value=float(patient.get('taille', 170.0)))
                            imc = st.number_input("IMC", min_value=0.0, max_value=100.0, 
                                                 value=round(poids / ((taille/100) ** 2), 1) if taille > 0 else 0.0,
                                                 disabled=True)
                        
                        st.subheader("Résultats de Laboratoire")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Hématologie**")
                            hemoglobine = st.number_input("Hémoglobine (g/dL)", min_value=0.0, max_value=30.0, value=14.0, step=0.1)
                            leucocytes = st.number_input("Leucocytes (10³/µL)", min_value=0.0, max_value=100.0, value=7.5, step=0.1)
                            plaquettes = st.number_input("Plaquettes (10³/µL)", min_value=0.0, max_value=1000.0, value=250.0, step=1.0)
                        
                        with col2:
                            st.markdown("**Biochimie**")
                            glycemie = st.number_input("Glycémie (mg/dL)", min_value=0.0, max_value=500.0, value=95.0, step=0.1)
                            creatinine = st.number_input("Créatinine (mg/dL)", min_value=0.0, max_value=20.0, value=0.9, step=0.01)
                            cholesterol = st.number_input("Cholestérol (mg/dL)", min_value=0.0, max_value=500.0, value=180.0, step=1.0)
                        
                        with col3:
                            st.markdown("**Marqueurs Spécifiques**")
                            crp = st.number_input("CRP (mg/L)", min_value=0.0, max_value=200.0, value=2.0, step=0.1)
                            vs = st.number_input("VS (mm/h)", min_value=0, max_value=200, value=15)
                            autres_analyses = st.text_area("Autres analyses", placeholder="Résultats supplémentaires...")
                        
                        st.subheader("Diagnostic et Traitement")
                        
                        diagnostic = st.text_area("Diagnostic*", 
                                                 placeholder="Décrire le diagnostic établi...",
                                                 height=100)
                        traitement = st.text_area("Traitement Prescrit", 
                                                 placeholder="Médicaments, posologie, durée...",
                                                 height=100)
                        recommandations = st.text_area("Recommandations", 
                                                      placeholder="Conseils, suivi, précautions...",
                                                      height=100)
                        
                        notes_medecin = st.text_area("Notes du Médecin", 
                                                    placeholder="Observations complémentaires...",
                                                    height=80)
                        
                        st.markdown("---")
                        st.caption("* Champ obligatoire")
                        
                        submitted = st.form_submit_button("💾 Enregistrer les Informations Médicales", 
                                                         use_container_width=True)
                        
                        if submitted:
                            if not diagnostic:
                                st.error("Veuillez saisir un diagnostic.")
                            else:
                                # Préparation des données médicales
                                medical_data = {
                                    "examen_clinique": {
                                        "tension_arterielle": tension_arterielle,
                                        "frequence_cardiaque": frequence_cardiaque,
                                        "temperature": temperature,
                                        "saturation_o2": saturation_o2,
                                        "poids_confirme": poids,
                                        "taille_confirmee": taille,
                                        "imc": imc
                                    },
                                    "laboratoire": {
                                        "hematologie": {
                                            "hemoglobine": hemoglobine,
                                            "leucocytes": leucocytes,
                                            "plaquettes": plaquettes
                                        },
                                        "biochimie": {
                                            "glycemie": glycemie,
                                            "creatinine": creatinine,
                                            "cholesterol": cholesterol
                                        },
                                        "autres": {
                                            "crp": crp,
                                            "vs": vs,
                                            "autres_analyses": autres_analyses
                                        }
                                    },
                                    "diagnostic": diagnostic,
                                    "traitement": traitement,
                                    "recommandations": recommandations,
                                    "notes_medecin": notes_medecin,
                                    "medecin": st.session_state.get("auth_username", "medecin")
                                }
                                
                                # Mise à jour du patient
                                try:
                                    update_patient(patient_id, {"medical_data": medical_data})
                                    st.success("✅ Informations médicales enregistrées avec succès !")
                                    st.info(f"Le dossier de {patient.get('prenom', '')} {patient.get('nom', '')} est maintenant complet.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Erreur lors de l'enregistrement: {str(e)}")
    
    # ── Tab 2: Rechercher Patient ─────────────────────────────────────────────
    with tab2:
        section_label("🔍 Recherche de Dossiers Patients")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("Rechercher un patient", 
                                       placeholder="Nom, prénom, ID patient...",
                                       key="search_med")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_btn = st.button("🔎 Rechercher", use_container_width=True, key="btn_search_med")
        
        if search_btn and search_term:
            results = search_patients(search_term)
            if results:
                st.success(f"📊 {len(results)} dossier(s) trouvé(s)")
                
                # Afficher les résultats
                for patient in results:
                    with st.expander(f"📁 {patient.get('prenom', '')} {patient.get('nom', '')} - Statut: {patient.get('status', '').replace('_', ' ').title()}", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Informations de Base:**")
                            st.text(f"ID: {patient.get('patient_id', '')}")
                            st.text(f"Nom: {patient.get('nom', '')}")
                            st.text(f"Prénom: {patient.get('prenom', '')}")
                            st.text(f"Âge: {patient.get('age', '')}")
                            st.text(f"Motif: {patient.get('motif_consultation', '')}")
                        
                        with col2:
                            st.markdown("**Statut et Dates:**")
                            st.text(f"Statut: {patient.get('status', '').replace('_', ' ').title()}")
                            st.text(f"Créé le: {patient.get('created_at', '').split('T')[0] if patient.get('created_at') else ''}")
                            if patient.get('updated_at'):
                                st.text(f"Mis à jour le: {patient.get('updated_at', '').split('T')[0]}")
                        
                        # Afficher les données médicales si disponibles
                        if patient.get('medical_data'):
                            st.markdown("---")
                            st.markdown("**📋 Données Médicales:**")
                            medical = patient.get('medical_data', {})
                            
                            if medical.get('diagnostic'):
                                st.text(f"Diagnostic: {medical.get('diagnostic', '')}")
                            
                            if medical.get('traitement'):
                                st.text(f"Traitement: {medical.get('traitement', '')}")
            else:
                st.info("Aucun dossier trouvé avec ces critères.")
    
    # ── Tab 3: Dossiers Complets ─────────────────────────────────────────────
    with tab3:
        section_label("📊 Dossiers Médicaux Complets")
        
        complete_patients = get_patients_by_status("complete")
        
        if not complete_patients:
            st.info("📭 Aucun dossier médical complet pour le moment.")
        else:
            st.success(f"✅ {len(complete_patients)} dossier(s) médical(aux) complet(s)")
            
            # Statistiques
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Dossiers Complets", len(complete_patients))
            with col2:
                # Compter par médecin
                medecins = {}
                for patient in complete_patients:
                    medical_data = patient.get('medical_data', {})
                    medecin = medical_data.get('medecin', 'Inconnu')
                    medecins[medecin] = medecins.get(medecin, 0) + 1
                if medecins:
                    top_medecin = max(medecins.items(), key=lambda x: x[1])
                    st.metric("Médecin le plus actif", f"{top_medecin[0]} ({top_medecin[1]})")
            
            # Liste des dossiers
            st.markdown("---")
            for patient in complete_patients:
                with st.expander(f"✅ {patient.get('prenom', '')} {patient.get('nom', '')} - Complété par: {patient.get('medical_data', {}).get('medecin', 'Inconnu')}", expanded=False):
                    # Informations résumées
                    medical = patient.get('medical_data', {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Diagnostic:**")
                        st.write(medical.get('diagnostic', 'Non spécifié'))
                    
                    with col2:
                        st.markdown("**Traitement:**")
                        st.write(medical.get('traitement', 'Non spécifié') if medical.get('traitement') else "Aucun traitement spécifié")
                    
                    # Bouton pour voir les détails
                    if st.button(f"📄 Voir le dossier complet", key=f"view_{patient.get('patient_id', '')}"):
                        st.session_state.selected_patient_id = patient.get('patient_id', '')
                        st.rerun()
    
    # ── Tab 4: Tous les Patients ─────────────────────────────────────────────
    with tab4:
        section_label("👥 Liste Complète des Patients")
        
        all_patients = get_all_patients()
        
        if not all_patients:
            st.info("📭 Aucun patient enregistré dans le système.")
        else:
            st.success(f"👥 {len(all_patients)} patient(s) enregistré(s) au total")
            
            # Statistiques
            col1, col2, col3 = st.columns(3)
            with col1:
                en_attente = len([p for p in all_patients if p.get('status') == 'en_attente'])
                st.metric("En Attente", en_attente)
            with col2:
                en_cours = len([p for p in all_patients if p.get('status') == 'en_cours'])
                st.metric("En Cours", en_cours)
            with col3:
                complets = len([p for p in all_patients if p.get('status') == 'complete'])
                st.metric("Complets", complets)
            
            # Tableau des patients
            st.markdown("---")
            st.markdown("### 📋 Liste des Patients")
            
            # Créer un DataFrame pour l'affichage
            patients_data = []
            for patient in all_patients:
                patients_data.append({
                    "ID": patient.get('patient_id', ''),
                    "Nom": patient.get('nom', ''),
                    "Prénom": patient.get('prenom', ''),
                    "Âge": patient.get('age', ''),
                    "Sexe": patient.get('sexe', ''),
                    "Statut": patient.get('status', '').replace('_', ' ').title(),
                    "Créé le": patient.get('created_at', '').split('T')[0] if patient.get('created_at') else '',
                    "Motif": patient.get('motif_consultation', '')[:30] + "..." if len(patient.get('motif_consultation', '')) > 30 else patient.get('motif_consultation', '')
                })
            
            if patients_data:
                df = pd.DataFrame(patients_data)
                
                # Ajouter un filtre par statut
                statuts = df['Statut'].unique()
                selected_statuts = st.multiselect(
                    "Filtrer par statut",
                    options=statuts,
                    default=statuts
                )
                
                if selected_statuts:
                    df_filtered = df[df['Statut'].isin(selected_statuts)]
                else:
                    df_filtered = df
                
                # Afficher le tableau
                st.dataframe(
                    df_filtered,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "ID": st.column_config.TextColumn("ID", width="small"),
                        "Nom": st.column_config.TextColumn("Nom", width="small"),
                        "Prénom": st.column_config.TextColumn("Prénom", width="small"),
                        "Âge": st.column_config.NumberColumn("Âge", width="small"),
                        "Sexe": st.column_config.TextColumn("Sexe", width="small"),
                        "Statut": st.column_config.TextColumn("Statut", width="small"),
                        "Créé le": st.column_config.TextColumn("Créé le", width="small"),
                        "Motif": st.column_config.TextColumn("Motif", width="medium")
                    }
                )
                
                # Options de détail
                st.markdown("---")
                st.markdown("### 🔍 Détails des Patients")
                
                selected_patient_id = st.selectbox(
                    "Sélectionner un patient pour voir les détails",
                    options=df_filtered['ID'].tolist(),
                    format_func=lambda x: f"{df_filtered[df_filtered['ID'] == x]['Prénom'].iloc[0]} {df_filtered[df_filtered['ID'] == x]['Nom'].iloc[0]} (ID: {x})"
                )
                
                if selected_patient_id:
                    patient = get_patient(selected_patient_id)
                    if patient:
                        with st.expander(f"📄 Dossier complet de {patient.get('prenom', '')} {patient.get('nom', '')}", expanded=True):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Informations Personnelles:**")
                                st.text(f"ID: {patient.get('patient_id', 'N/A')}")
                                st.text(f"Nom: {patient.get('nom', 'N/A')}")
                                st.text(f"Prénom: {patient.get('prenom', 'N/A')}")
                                st.text(f"Âge: {patient.get('age', 'N/A')}")
                                st.text(f"Sexe: {patient.get('sexe', 'N/A')}")
                                st.text(f"Téléphone: {patient.get('telephone', 'N/A')}")
                                st.text(f"Email: {patient.get('email', 'N/A')}")
                                st.text(f"Adresse: {patient.get('adresse', 'N/A')}")
                                st.text(f"Profession: {patient.get('profession', 'N/A')}")
                            
                            with col2:
                                st.markdown("**Informations Médicales:**")
                                st.text(f"Statut: {patient.get('status', 'N/A').replace('_', ' ').title()}")
                                st.text(f"Motif: {patient.get('motif_consultation', 'N/A')}")
                                if patient.get('poids'):
                                    st.text(f"Poids: {patient.get('poids', '')} kg")
                                if patient.get('taille'):
                                    st.text(f"Taille: {patient.get('taille', '')} cm")
                                
                                # Antécédents
                                antecedents = patient.get('antecedents', {})
                                if antecedents:
                                    st.markdown("**Antécédents:**")
                                    for key, value in antecedents.items():
                                        if value:
                                            st.text(f"- {key.replace('_', ' ').title()}: {value}")
                                
                                # Données médicales si disponibles
                                medical_data = patient.get('medical_data', {})
                                if medical_data:
                                    st.markdown("**Données Médicales:**")
                                    if medical_data.get('diagnostic'):
                                        st.text(f"Diagnostic: {medical_data.get('diagnostic', '')}")
                                    if medical_data.get('medecin'):
                                        st.text(f"Médecin: {medical_data.get('medecin', '')}")
    
    # ── Tab 5: Thyroïde ─────────────────────────────────────────────
    with tab5:
        page_header("🦋 Diagnostic Thyroïdien",
                   "Analyse des Marqueurs Hormonaux",
                   "Renseignez les paramètres biologiques pour une analyse prédictive en temps réel")
        from modules.prediction import render
        render()
    
    # ── Tab 6: Cancer Cérébral ─────────────────────────────────────────────
    with tab6:
        page_header("🧠 Diagnostic Tumeur Cérébrale",
                   "Analyse d'Images IRM par Deep Learning",
                   "Analysez une image IRM cérébrale — classification en 4 classes par EfficientNet-B0")
        from modules.brain_tumor import render
        render()
    
    # ── Tab 7: PTDM ─────────────────────────────────────────────
    with tab7:
        page_header("🩸 Prédiction Risque PTDM",
                   "Diabète Post-Transplantation",
                   "Évaluation du risque de développer un diabète après transplantation rénale")
        from modules.ptdm_prediction import render
        render()