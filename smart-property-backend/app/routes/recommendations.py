from fastapi import APIRouter, HTTPException

from app.schema import ClientPreferences
from app.core.data_loader import df
from app.services.recommender import recommend_houses


router = APIRouter(tags=["recommendation"])


@router.post("/recommendation/recommend")
@router.post("/recommend")
def recommend(pref: ClientPreferences):
    try:
        results = recommend_houses(pref, df)
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))