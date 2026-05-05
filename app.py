import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Maintenance Prédictive - Dashboard Opérationnel", layout="wide")

# CSS pour améliorer l'interface
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Dashboard Maintenance Prédictive - Mode Opérationnel")
st.write("**Outil d'aide à la décision** : Visualisez les KPIs, simulez des scénarios et obtenez des prédictions en temps réel.")

# ============ CHARGEMENT DES DONNÉES ET MODÈLE ============
@st.cache_data
def load_data():
    try:
        return pd.read_csv('data/predictive_maintenance_v3.csv')
    except Exception:
        return None

@st.cache_data
def load_metrics():
    try:
        df = pd.read_csv('models/model_metrics_1.csv')
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

@st.cache_data
def load_feature_importance():
    try:
        return pd.read_csv('models/feature_importance_analysis.csv')
    except FileNotFoundError:
        return None

model = load_model()
data = load_data()
metrics = load_metrics()
feature_importance = load_feature_importance()

if model and data is not None:
    
    # ============ SECTION 1 : KPIs GLOBAUX ============
    st.header("Indicateurs Clés de Performance (KPIs)")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        failure_rate = (data['failure_within_24h'].sum() / len(data)) * 100
        st.metric("Taux de Panne", f"{failure_rate:.1f}%", delta=None)
    
    with kpi_col2:
        if metrics is not None and not metrics.empty:
            # Accès sécurisé au F1-Score
            if 'F1-Score' in metrics.columns:
                f1_score = metrics['F1-Score'].iloc[2]
                st.metric("Score F1 du Modèle", f"{f1_score:.3f}")
            else:
                st.metric("Score F1 du Modèle", "Calcul...")
        else:
            st.metric("Score F1 du Modèle", "N/A")
    
    with kpi_col3:
        avg_vibration = data['vibration_rms'].mean()
        st.metric("Vibration Moyenne (RMS)", f"{avg_vibration:.2f}", delta="Baseline")
    
    with kpi_col4:
        avg_temp = data['temperature_motor'].mean()
        st.metric("Température Moyenne (°C)", f"{avg_temp:.1f}°C", delta="Normale")
    
    st.divider()
    
    
    st.sidebar.header("Paramètres de la Machine")
    st.sidebar.write("Ajustez les paramètres pour simuler différents scénarios.")
    
    # Récupérer les statistiques du dataset pour les seuils
    vibration_max = data['vibration_rms'].max()
    temp_max = data['temperature_motor'].max()
    current_max = data['current_phase_avg'].max()
    pressure_max = data['pressure_level'].max()
    rpm_max = data['rpm'].max()
    
    # Widgets dans la barre latérale
    vibration_rms = st.sidebar.slider("Vibration (RMS) [m/s]", 
                                       min_value=0.0, max_value=vibration_max, 
                                       value=2.5, step=0.1)
    temperature_motor = st.sidebar.slider("Température Moteur [°C]", 
                                          min_value=20.0, max_value=temp_max, 
                                          value=80.0, step=1.0)
    current_phase_avg = st.sidebar.slider("Courant moyen (Phase) [A]", 
                                          min_value=10.0, max_value=current_max, 
                                          value=25.0, step=0.5)
    pressure_level = st.sidebar.slider("Niveau de Pression [bar]", 
                                       min_value=0.0, max_value=pressure_max, 
                                       value=30.0, step=1.0)
    rpm = st.sidebar.slider("Vitesse de rotation [RPM]", 
                            min_value=500.0, max_value=rpm_max, 
                            value=2000.0, step=100.0)
    hours_since_maintenance = st.sidebar.number_input("Heures depuis maintenance", 
                                                       min_value=0.0, max_value=20000.0, 
                                                       value=250.0, step=10.0)
    operating_mode = st.sidebar.selectbox("Mode Opératoire", ["normal", "idle", "peak"])
    
    # Préparer les données d'entrée
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
    
    # ============ SECTION 3 : RÉSULTAT DE PRÉDICTION AVEC VISUALISATION (BOUTON UNIQUE) ============
    st.header("Analyse et Prédiction - Aide à la Décision")
    
    if st.button("Lancer la Prédiction", key="unique_prediction_btn"):
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None
        
        risk_prob = probability[1] * 100 if probability is not None else 0
        
        # Affichage du résultat
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            # Jauge de risque
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risque de Panne (%)"},
                delta={'reference': 50, 'decreasing': {'color': '#06a77d'}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "#90EE90"},
                        {'range': [30, 70], 'color': "#FFD700"},
                        {'range': [70, 100], 'color': "#FF6B6B"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with result_col2:
            st.subheader("Décision d'Intervention")
            
            if prediction[0] == 1:
                st.error(f"ALERTE CRITIQUE\n\nRisque de panne : **{risk_prob:.1f}%**")
                st.write("**Recommandation** : **INTERVENTION IMMÉDIATE REQUISE**")
                st.write("- Arrêt prévu dans les 24h")
                st.write("- Maintenance urgente recommandée")
                st.write("- Impact opérationnel : Élevé")
            else:
                if risk_prob > 30:
                    st.warning(f"ALERTE MODÉRÉE\n\nRisque élevé : **{risk_prob:.1f}%**")
                    st.write("**Recommandation** : **SURVEILLANCE RENFORCÉE + INTERVENTION PRÉVENTIVE**")
                    st.write("- Planifier une intervention dans les 48-72h")
                    st.write("- Réduire la charge opérationnelle si possible")
                    st.write("- Impact opérationnel : Modéré")
                else:
                    st.success(f"FONCTIONNEMENT NORMAL\n\nRisque faible : **{risk_prob:.1f}%**")
                    st.write("**Recommandation** : **SURVEILLANCE RÉGULIÈRE**")
                    st.write("- Maintenance planifiée standard")
                    st.write("- Aucune action urgente requise")
                    st.write("- Impact opérationnel : Minimal")
    
    st.divider()
    
    # ============ SECTION 4 : COMPARAISON AVEC LES SEUILS ============
    st.header("Analyse Comparative - Paramètres vs Seuils")
    
    # Définir les seuils de risque
    thresholds = {
        'vibration_rms': {'normal': 3.0, 'warning': 5.0, 'critical': 8.0},
        'temperature_motor': {'normal': 90, 'warning': 130, 'critical': 160},
        'current_phase_avg': {'normal': 40, 'warning': 60, 'critical': 80},
        'pressure_level': {'normal': 40, 'warning': 70, 'critical': 100},
        'rpm': {'normal': 3000, 'warning': 4000, 'critical': 4800},
        'hours_since_maintenance': {'normal': 500, 'warning': 1000, 'critical': 1500}
    }
    
    # Créer un tableau comparatif
    comparison_data = []
    for param, value in input_data.items():
        if param != 'operating_mode' and param in thresholds:
            thresh = thresholds[param]
            if value <= thresh['normal']:
                status = "Normal"
                color_val = "lightgreen"
            elif value <= thresh['warning']:
                status = "Attention"
                color_val = "gold"
            else:
                status = "Critique"
                color_val = "lightcoral"
            
            comparison_data.append({
                'Paramètre': param.replace('_', ' ').title(),
                'Valeur Actuelle': f"{value:.2f}",
                'Seuil Normal': f"{thresh['normal']:.2f}",
                'Seuil Alerte': f"{thresh['warning']:.2f}",
                'Statut': status
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # ============ SECTION 5 : IMPORTANCE DES VARIABLES ============
    if feature_importance is not None:
        st.header("Importance des Variables")
        
        fig_importance = px.bar(
            feature_importance.head(10),
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 10 Variables les Plus Influentes",
            labels={'Feature': 'Variable', 'Importance': 'Importance (%)'}
        )
        fig_importance.update_layout(height=400)
        st.plotly_chart(fig_importance, use_container_width=True)
    
    st.divider()
    
    # ============ SECTION 6 : MÉTRIQUES DU MODÈLE ============
    if metrics is not None and not metrics.empty:
        st.header("Performance du Modèle")
        
        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
        
        metrics_dict = {
            'Accuracy': metrics['Accuracy'].iloc[0] if 'Accuracy' in metrics.columns else None,
            'Precision': metrics['Precision'].iloc[0] if 'Precision' in metrics.columns else None,
            'Recall': metrics['Recall'].iloc[0] if 'Recall' in metrics.columns else None,
            'F1-Score': metrics['F1-Score'].iloc[0] if 'F1-Score' in metrics.columns else None,
            'ROC-AUC': metrics['ROC-AUC'].iloc[0] if 'ROC-AUC' in metrics.columns else None
        }
        
        with metric_col1:
            if metrics_dict['Accuracy'] is not None:
                st.metric("Accuracy", f"{metrics_dict['Accuracy']:.3f}")
        with metric_col2:
            if metrics_dict['Precision'] is not None:
                st.metric("Precision", f"{metrics_dict['Precision']:.3f}")
        with metric_col3:
            if metrics_dict['Recall'] is not None:
                st.metric("Recall", f"{metrics_dict['Recall']:.3f}")
        with metric_col4:
            if metrics_dict['F1-Score'] is not None:
                st.metric("F1-Score", f"{metrics_dict['F1-Score']:.3f}")
        with metric_col5:
            if metrics_dict['ROC-AUC'] is not None:
                st.metric("ROC-AUC", f"{metrics_dict['ROC-AUC']:.3f}")
    
    st.divider()
    
    # ============ SECTION 7 : AIDE À LA DÉCISION ============
    st.header("Guide de Décision d'Intervention")
    
    col_decision1, col_decision2 = st.columns(2)
    
    with col_decision1:
        st.subheader("Pas d'Intervention Requise")
        st.write("""
        - Risque de panne < 30%
        - Tous les paramètres en zone normale
        - **Action** : Surveillance régulière
        """)
    
    with col_decision2:
        st.subheader("Intervention Immédiate")
        st.write("""
        - Risque de panne > 70%
        - Paramètres en zone critique
        - **Action** : Arrêt et maintenance urgente
        """)
    
    st.info("**Mode Opérationnel** : Cet outil aide le responsable maintenance à prioriser les interventions et optimiser l'utilisation des ressources.")

else:
    st.error("Erreur : Impossible de charger le modèle ou les données.")
    
    