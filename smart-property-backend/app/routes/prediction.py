from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException

from app.services.prediction_service import (
    HouseFeatures,
    MODEL_METRICS,
    build_model_row,
    get_places_data,
    model,
    use_log_target,
)

router = APIRouter()


@router.get("/places")
def get_places():
    """
    Retourne uniquement les lieux géographiques pour le frontend.
    Le champ state n'est pas retourné car il représente l'état du bien.
    """
    return get_places_data()


@router.get("/metrics")
def get_metrics():
    return MODEL_METRICS or {
        "r2": None,
        "mae": None,
        "rmse": None,
        "mape": None,
    }


@router.post("/predict")
def predict_price(data: Optional[HouseFeatures] = None):
    """
    Route de prédiction. Tous les champs sont optionnels afin de conserver
    la compatibilité avec un formulaire incomplet.
    """
    try:
        if data is None:
            data = HouseFeatures()

        row = build_model_row(data)
        prediction = model.predict(row)[0]

        if use_log_target:
            prediction = np.expm1(prediction)

        prediction = max(float(prediction), 0.0)

        response = {
            "predicted_price_tnd": round(prediction, 2),
        }

        if MODEL_METRICS is not None:
            response["metrics"] = MODEL_METRICS

        return response

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la prédiction : {str(exc)}",
        ) from exc

