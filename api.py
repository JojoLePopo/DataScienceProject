from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import Optional


app = FastAPI(
    title="API Maintenance Prédictive",
    description="API REST pour prédire les pannes des machines industrielles.",
    version="1.0",
)


try:
    model = joblib.load("models/best_model.pkl")
except FileNotFoundError:
    model = None
    print("Attention: Modèle non trouvé. Veuillez lancer l'entraînement d'abord.")


class SensorData(BaseModel):
    vibration_rms: float
    temperature_motor: float
    current_phase_avg: float
    pressure_level: float
    rpm: float
    hours_since_maintenance: float
    operating_mode: str


class PredictionResponse(BaseModel):
    prediction: int
    probability_of_failure: Optional[float]
    status: str


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Les données envoyées sont invalides.",
            "errors": exc.errors(),
            "path": str(request.url.path),
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Une erreur interne est survenue pendant le traitement.",
            "path": str(request.url.path),
        },
    )


@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de Maintenance Prédictive. Utilisez /docs pour voir la documentation."}


@app.get("/health")
def health_check():
    """Vérifie que l'API est en ligne et que le modèle est chargé."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas chargé. Lancez d'abord l'entraînement avec python main.py.",
        )
    return {"status": "ok", "model_loaded": True, "service": "api-maintenance-predictive"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: SensorData):
    """Reçoit les données capteurs et renvoie une prédiction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle indisponible pour la prédiction.")

    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = int(model.predict(input_df)[0])

        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_df)[0][1])

        return {
            "prediction": prediction,
            "probability_of_failure": probability,
            "status": "Panne détectée dans les 24h" if prediction == 1 else "Fonctionnement normal",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur pendant la prédiction: {exc}") from exc