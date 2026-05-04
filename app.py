import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


st.set_page_config(page_title="Dashboard Maintenance Prédictive", layout="wide")



st.title("Dashboard de Maintenance Prédictive Industrielle")
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
    
    # --- Création des onglets ---
    tab_simulation, tab_metrics, tab_viz, tab_data = st.tabs([
        "⚙️ Simulation de Prédiction", 
        "📊 Performances des modèles", 
        "📈 Analyses et Corrélations", 
        "📋 Vue d'ensemble des données"
    ])
    
    # --- Onglet 1 : Simulation ---
    with tab_simulation:
        st.header("Simulation de Prédiction en direct")
        
        # Widgets adaptés aux nouvelles variables du modèle
        vibration_rms = st.sidebar.slider("Vibration (RMS)", min_value=0.0, max_value=15.0, value=2.5, step=0.1)
        temperature_motor = st.sidebar.slider("Température Moteur (°C)", min_value=20.0, max_value=200.0, value=80.0, step=1.0)
        current_phase_avg = st.sidebar.slider("Courant moyen (Phase)", min_value=10.0, max_value=100.0, value=25.0, step=0.5)
        pressure_level = st.sidebar.slider("Niveau de Pression", min_value=0.0, max_value=150.0, value=30.0, step=1.0)
        rpm = st.sidebar.slider("Vitesse de rotation (RPM)", min_value=500.0, max_value=5000.0, value=2000.0, step=100.0)
        hours_since_maintenance = st.sidebar.number_input("Heures depuis dernière maintenance", min_value=0.0, max_value=20000.0, value=1500.0, step=10.0)
        operating_mode = st.sidebar.selectbox("Mode Opératoire", ["normal", "idle", "peak"])

        st.subheader("Données simulées")
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

        if st.button("Lancer la Prédiction"):
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None
        
            st.subheader("Résultat de la prédiction")
            if prediction[0] == 1:
                st.error("⚠️ ALERTE : Risque de panne détecté dans les 24 heures !")
            else:
                st.success("✅ Fonctionnement normal de la machine. Aucun risque détecté.")
                
            if probability is not None:
                st.write(f"**Probabilité de panne :** {probability[1] * 100:.2f} %")
                
    # --- Onglet 2 : Métriques ---
    with tab_metrics:
        st.header("Performances comparées des modèles")
        metrics_df = load_metrics()
        
        if metrics_df is not None:
            st.write("Voici les indicateurs qui ont permis d'évaluer et de choisir le meilleur modèle lors de l'entraînement :")
            
            # Réorganisation pour afficher Model et Global en premier
            cols = ['Model']
            if 'Global' in metrics_df.columns:
                cols.append('Global')
            cols += [c for c in metrics_df.columns if c not in cols]
            
            metrics_df = metrics_df[cols]
            
            # On surligne uniquement les colonnes numériques
            numeric_cols = [c for c in metrics_df.columns if metrics_df[c].dtype in ['float64', 'int64']]
            st.dataframe(metrics_df.style.highlight_max(subset=numeric_cols, color='lightgreen', axis=0), use_container_width=True)
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Matrice de Confusion du meilleur modèle")
                try:
                    cm = np.load('models/confusion_matrix.npy')
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                     labels=dict(x="Prédiction", y="Valeur Réelle", color="Nombre"),
                                     x=['Normal (0)', 'Panne (1)'], y=['Normal (0)', 'Panne (1)'],
                                     title="Analyse d'Erreur (Matrice de Confusion)")
                    st.plotly_chart(fig_cm, use_container_width=True)
                except FileNotFoundError:
                    st.warning("Matrice de confusion indisponible. Relancez l'entraînement.")
            
            with col2:
                st.write("### Importance des Variables (Feature Importance)")
                try:
                    fi_df = pd.read_csv('models/feature_importance.csv')
                    if not fi_df.empty:
                        # Bar chart horizontal
                        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                                        title="Poids de chaque capteur dans la prédiction",
                                        color='Importance', color_continuous_scale='viridis')
                        st.plotly_chart(fig_fi, use_container_width=True)
                    else:
                        st.info("Le modèle retenu ne remonte pas l'importance des variables.")
                except FileNotFoundError:
                    st.warning("Fichier d'importance des variables indisponible.")
        else:
            st.info("Lancez `python main.py` pour générer les métriques d'entraînement.")

    # Chargement du dataset brut pour les onglets 3 et 4
    raw_data = load_data()

    # --- Onglet 3 : Analyses et Corrélations ---
    with tab_viz:
        st.header("Analyses Exploratoires et Corrélations")
        if raw_data is not None:
            st.write("### Matrice de corrélation (Variables numériques)")
            # On ne garde que les variables numériques
            numeric_cols_data = raw_data.select_dtypes(include=['float64', 'int64']).columns
            corr_matrix = raw_data[numeric_cols_data].corr()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax, linewidths=0.5)
            st.pyplot(fig)
            
            st.write("### Distribution des Pannes selon le Mode Opératoire")
            fig_fail = px.histogram(raw_data, x="operating_mode", color="failure_within_24h", text_auto=True, 
                                    barmode="group", title="Pannes par statut de fonctionnement")
            st.plotly_chart(fig_fail, use_container_width=True)
            
            st.write("### Relation Température vs Vibration")
            fig_scatter = px.scatter(raw_data, x="vibration_rms", y="temperature_motor", color="failure_within_24h", 
                                     opacity=0.6, title="Vibration vs Température coloré par l'état (Panne ou Non)")
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible. Placez le fichier 'predictive_maintenance_v3.csv' dans le dossier /data.")

    # --- Onglet 4 : Vue d'ensemble des données ---
    with tab_data:
        st.header("Vue d'ensemble des données brutes")
        if raw_data is not None:
            st.write("Aperçu des 50 premières lignes :")
            st.dataframe(raw_data.head(50))
            
            st.write("### Statistiques Descriptives")
            st.dataframe(raw_data.describe())
            
            st.write("### Répartition de la variable cible (nombre de pannes vs fonctionnement normal)")
            target_counts = raw_data["failure_within_24h"].value_counts().reset_index()
            target_counts.columns = ["Statut de Panne", "Nombre"]
            
            # Remplacement du tableau par un graphique (Pie chart)
            target_counts["Statut de Panne"] = target_counts["Statut de Panne"].map({0: "Fonctionnement Normal", 1: "Panne Imminente"})
            fig_pie = px.pie(target_counts, values="Nombre", names="Statut de Panne", 
                             title="Distribution des classes (Équilibre des données)",
                             color="Statut de Panne",
                             color_discrete_map={"Fonctionnement Normal": "#2ca02c", "Panne Imminente": "#d62728"},
                             hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("Aucune donnée disponible. Placez le fichier 'predictive_maintenance_v3.csv' dans le dossier /data.")
