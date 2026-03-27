"""
Application de prédiction du diabète - Version corrigée
Sans la colonne Insulin (supprimée lors du prétraitement)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Diabète",
    page_icon="🩺",
    layout="wide"
)

# Chargement du modèle et des métadonnées
@st.cache_resource
def load_model():
    """Charge le modèle sauvegardé depuis Colab"""
    try:
        model = joblib.load('model_diabete.pkl')
        metadata = joblib.load('model_metadata.pkl')
        return model, metadata
    except FileNotFoundError:
        st.error("""
        ❌ **Fichiers du modèle non trouvés!**
        
        Assurez-vous que:
        1. `model_diabete.pkl` est dans le même dossier que cette application
        2. `model_metadata.pkl` est également présent
        """)
        return None, None
    except Exception as e:
        st.error(f"❌ Erreur de chargement: {str(e)}")
        return None, None

# Charger
model, metadata = load_model()

# Titre
st.title("🩺 Prédiction du Diabète - Interface Médicale")
st.markdown("---")

# Sidebar avec les métriques
with st.sidebar:
    st.header("📊 Performance du modèle")
    
    if metadata:
        if 'auc_test' in metadata:
            st.metric("AUC-ROC", f"{metadata['auc_test']:.4f}")
        if 'accuracy_test' in metadata:
            st.metric("Accuracy", f"{metadata['accuracy_test']:.2%}")
        if 'n_features' in metadata:
            st.metric("Variables utilisées", metadata['n_features'])
    
    st.markdown("---")
    st.markdown("### 📚 Variables cliniques")
    st.markdown("""
    - **Pregnancies** (nombre de grossesses)
    - **Glucose** (glycémie à jeun)
    - **BloodPressure** (pression artérielle)
    - **SkinThickness** (épaisseur cutanée)
    - **BMI** (indice de masse corporelle)
    - **DiabetesPedigreeFunction** (risque génétique)
    - **Age** (âge)
    """)
    
    st.markdown("---")
    st.markdown("### ⚠️ Avertissement")
    st.markdown("*Application d'aide à la décision - Ne remplace pas un diagnostic médical*")

# Formulaire de saisie
st.header("📝 Saisie des données cliniques")

# Organisation en colonnes
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Informations générales**")
    pregnancies = st.number_input(
        "Nombre de grossesses",
        min_value=0,
        max_value=17,
        value=1,
        help="Nombre total de grossesses antérieures"
    )
    
    age = st.number_input(
        "Âge (années)",
        min_value=21,
        max_value=81,
        value=31,
        help="Âge de la patiente (≥ 21 ans)"
    )
    
    bmi = st.number_input(
        "IMC (kg/m²)",
        min_value=10.0,
        max_value=70.0,
        value=26.6,
        step=0.1,
        format="%.1f",
        help="Indice de masse corporelle = poids / taille²"
    )

with col2:
    st.markdown("**🩸 Paramètres sanguins**")
    glucose = st.number_input(
        "Glucose (mg/dL)",
        min_value=44,
        max_value=199,
        value=85,
        help="Concentration de glucose plasmatique à jeun"
    )
    
    dpf = st.number_input(
        "Fonction pedigree diabète",
        min_value=0.078,
        max_value=2.42,
        value=0.351,
        step=0.01,
        format="%.3f",
        help="Risque génétique de diabète"
    )

with col3:
    st.markdown("**📏 Mesures physiques**")
    blood_pressure = st.number_input(
        "Pression artérielle (mm Hg)",
        min_value=44,
        max_value=122,
        value=66,
        help="Pression artérielle diastolique"
    )
    
    skin_thickness = st.number_input(
        "Épaisseur cutanée (mm)",
        min_value=14,
        max_value=99,
        value=29,
        help="Épaisseur du pli cutané tricipital"
    )

# Bouton de prédiction
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🔍 PRÉDIRE LE DIABÈTE", type="primary", use_container_width=True)

# Prédiction
if predict_button:
    if model is None:
        st.error("❌ Modèle non chargé. Vérifiez que les fichiers sont présents.")
    else:
        # IMPORTANT: Créer le DataFrame avec les BONNES colonnes (sans Insulin)
        input_data = pd.DataFrame([{
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }])
        
        # Afficher les colonnes utilisées pour vérification
        st.write("**Colonnes utilisées pour la prédiction:**")
        st.write(list(input_data.columns))
        
        # Prédiction
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            proba_diabete = probability[1] * 100
            proba_sain = probability[0] * 100
            
            # Affichage des résultats
            st.markdown("---")
            st.markdown("## 📊 Résultat de la prédiction")
            
            # Deux colonnes pour les résultats
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                if prediction == 1:
                    st.error(f"""
                    ### ⚠️ RISQUE ÉLEVÉ
                    
                    **Probabilité de diabète:** `{proba_diabete:.1f}%`
                    
                    **Recommandation:** 
                    - ✅ Consultation médicale urgente
                    - ✅ Dosage Hba1c
                    - ✅ Test de tolérance au glucose
                    """)
                else:
                    st.success(f"""
                    ### ✅ RISQUE FAIBLE
                    
                    **Probabilité de non-diabète:** `{proba_sain:.1f}%`
                    
                    **Recommandation:** 
                    - ✅ Suivi médical annuel
                    - ✅ Alimentation équilibrée
                    - ✅ Activité physique régulière
                    """)
            
            with col_res2:
                # Graphique de jauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = proba_diabete,
                    title = {'text': "Risque de diabète (%)"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#ff4b4b"},
                        'steps': [
                            {'range': [0, 30], 'color': "#90EE90"},
                            {'range': [30, 70], 'color': "#FFD700"},
                            {'range': [70, 100], 'color': "#FF6B6B"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Tableau récapitulatif
            st.markdown("### 📋 Récapitulatif des données")
            
            recap_data = {
                'Variable': ['Grossesses', 'Glucose', 'Pression artérielle', 'Épaisseur cutanée', 
                            'IMC', 'Risque génétique', 'Âge'],
                'Valeur': [pregnancies, glucose, blood_pressure, skin_thickness,
                          f"{bmi:.1f}", f"{dpf:.3f}", age],
                'Unité': ['', 'mg/dL', 'mm Hg', 'mm', 'kg/m²', '', 'ans']
            }
            
            recap_df = pd.DataFrame(recap_data)
            st.dataframe(recap_df, use_container_width=True, hide_index=True)
            
            # Interprétation par variable
            with st.expander("🔍 Interprétation clinique détaillée"):
                st.markdown("**🩸 Glycémie**")
                if glucose < 100:
                    st.success(f"✅ {glucose} mg/dL - Normale")
                elif glucose < 126:
                    st.warning(f"⚠️ {glucose} mg/dL - Prédiabète")
                else:
                    st.error(f"❌ {glucose} mg/dL - Diabète suspecté")
                
                st.markdown("**⚖️ IMC**")
                if bmi < 18.5:
                    st.warning(f"⚠️ {bmi:.1f} - Insuffisance pondérale")
                elif bmi < 25:
                    st.success(f"✅ {bmi:.1f} - Poids normal")
                elif bmi < 30:
                    st.warning(f"⚠️ {bmi:.1f} - Surpoids")
                else:
                    st.error(f"❌ {bmi:.1f} - Obésité")
                
                st.markdown("**❤️ Pression artérielle**")
                if blood_pressure < 80:
                    st.success(f"✅ {blood_pressure} mmHg - Normale")
                else:
                    st.warning(f"⚠️ {blood_pressure} mmHg - Élevée")
                
                st.markdown("**🎂 Âge**")
                if age < 35:
                    st.success(f"✅ {age} ans - Jeune adulte")
                elif age < 45:
                    st.warning(f"⚠️ {age} ans - Risque modéré")
                else:
                    st.error(f"❌ {age} ans - Risque élevé")
                
                st.markdown("**🧬 Risque génétique**")
                if dpf < 0.5:
                    st.success(f"✅ {dpf:.3f} - Faible")
                else:
                    st.warning(f"⚠️ {dpf:.3f} - Élevé")
        
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction: {str(e)}")
            st.info("""
            **Détails techniques:** Vérifiez que:
            1. Les colonnes correspondent exactement à celles utilisées lors de l'entraînement
            2. Le modèle a été correctement sauvegardé
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>🩺 Application développée avec Streamlit</p>
    <p>Modèle entraîné sur le dataset PIMA Indians Diabetes Database (sans la variable Insulin)</p>
    <p>⚠️ Usage médical professionnel uniquement - Ne remplace pas un avis médical</p>
</div>
""", unsafe_allow_html=True)