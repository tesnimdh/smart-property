from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.prediction import router as prediction_router
from app.routes.recommendations import router as recommendations_router


app = FastAPI(title="House Price Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "API de prédiction immobilière active",
        "docs": "/docs",
        "routes_debug": "/debug/routes",
       
    }



app.include_router(prediction_router)
app.include_router(recommendations_router)





@app.get("/debug/routes")
def debug_routes():
    return [
        {
            "path": route.path,
            "methods": sorted(list(route.methods or [])),
            "name": route.name,
        }
        for route in app.routes
    ]
