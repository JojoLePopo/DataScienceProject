from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API Maintenance Prédictive",
    description="API REST pour prédire les pannes des machines industrielles.",
    version="1.0"
)

# Chargement du modèle au démarrage
try:
    model = joblib.load('models/best_model.pkl')
except FileNotFoundError:
    model = None
    print("Attention: Modèle non trouvé. Veuillez lancer l'entraînement d'abord.")

# Schéma des données d'entrée attendues (validation automatique par Pydantic)
class SensorData(BaseModel):
    vibration_rms: float
    temperature_motor: float
    current_phase_avg: float
    pressure_level: float
    rpm: float
    hours_since_maintenance: float
    operating_mode: str

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de Maintenance Prédictive. Utilisez /docs pour voir la documentation."}

@app.get("/health")
def health_check():
    """Vérifie que l'API est en ligne et que le modèle est chargé."""
    if model is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas chargé.")
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
def predict(data: SensorData):
    """Reçoit les données capteurs et renvoie une prédiction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle indisponible pour la prédiction.")
    
    # Transformation des données en DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Prédiction
    prediction = int(model.predict(input_df)[0])
    
    # Récupération de la probabilité si disponible
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_df)[0][1])
        
    return {
        "prediction": prediction,
        "probability_of_failure": probability,
        "status": "Panne detectée (Dans les 24h)" if prediction == 1 else "Fonctionnement Normal"
    }
