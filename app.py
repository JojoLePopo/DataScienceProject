import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


st.set_page_config(page_title="Dashboard Maintenance Prédictive", layout="wide")



st.title("🧰 Dashboard de Maintenance Prédictive Industrielle")
st.write("Ce dashboard permet d'analyser les données capteurs et de prédire les pannes des machines.")

@st.cache_data
def load_data():
    try:
        return pd.read_csv('data/predictive_maintenance_v3.csv')
    except Exception:
        return None

@st.cache_data
def load_metrics():
    try:
        df = pd.read_csv('models/model_metrics.csv')
        return df
    except FileNotFoundError:
        return None

@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/best_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Le modèle n'a pas été trouvé. Veuillez lancer `python main.py` d'abord.")
        return None

model = load_model()

if model:
    st.sidebar.header("Paramètres Capteurs de la Machine")
    
    st.header("⚙️ Simulation de Prédiction en direct")
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        # Widgets adaptés aux nouvelles variables du modèle
        vibration_rms = st.sidebar.slider("Vibration (RMS)", min_value=0.0, max_value=15.0, value=2.5, step=0.1)
        temperature_motor = st.sidebar.slider("Température Moteur (°C)", min_value=20.0, max_value=200.0, value=80.0, step=1.0)
        current_phase_avg = st.sidebar.slider("Courant moyen (Phase)", min_value=10.0, max_value=100.0, value=25.0, step=0.5)
        pressure_level = st.sidebar.slider("Niveau de Pression", min_value=0.0, max_value=150.0, value=30.0, step=1.0)
        rpm = st.sidebar.slider("Vitesse de rotation (RPM)", min_value=500.0, max_value=5000.0, value=2000.0, step=100.0)
        hours_since_maintenance = st.sidebar.number_input("Heures depuis dernière maintenance", min_value=0.0, max_value=20000.0, value=250.0, step=10.0)
        operating_mode = st.sidebar.selectbox("Mode Opératoire", ["normal", "idle", "peak"])

        st.subheader("📝 Données simulées")
        input_data = {
            'vibration_rms': vibration_rms,
            'temperature_motor': temperature_motor,
            'current_phase_avg': current_phase_avg,
            'pressure_level': pressure_level,
            'rpm': rpm,
            'hours_since_maintenance': hours_since_maintenance,
            'operating_mode': operating_mode
        }
        input_df = pd.DataFrame([input_data])
        st.write(input_df)

    with col_result:
        if st.button("Lancer la Prédiction"):
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None
        
            st.subheader("🔍 Résultat de la prédiction")
            if prediction[0] == 1:
                st.error("⚠️ ALERTE : Risque de panne détecté dans les 24 heures !")
            else:
                st.success("✅ Fonctionnement normal de la machine. Aucun risque détecté.")
                
            if probability is not None:
                st.write(f"**Probabilité de panne :** {probability[1] * 100:.2f} %")
                
    st.divider()
    
    with st.expander("📊 Afficher les Coefficients / Poids des variables du modèle"):
        try:
            coef_df = pd.read_csv('models/model_coefficients.csv')
            if not coef_df.empty:
                # Bar chart horizontal
                fig_fi = px.bar(coef_df, x='Coefficients', y='Feature', orientation='h',
                                title="Poids de chaque capteur dans la prédiction",
                                color='Coefficients', color_continuous_scale='viridis')
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Le modèle retenu ne remonte pas l'importance des variables.")
        except FileNotFoundError:
            st.warning("Fichier de coefficients indisponible. Relancez l'entraînement (`python main.py`).")
