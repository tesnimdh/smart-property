# ImmoTunisia — Backend API

API FastAPI unifiée combinant :
- **`POST /predict`** — Estimation de prix via modèle XGBoost
- **`GET /predict/places`** — Liste des zones géographiques disponibles
- **`POST /recommendation/recommend`** — Recommandation de biens par similarité cosinus
- **`GET /stats/...`** — Statistiques immobilières par gouvernorat, type, etc.
- **`GET /docs`** — Documentation Swagger interactive

## Structure

```
immo-tunisia-backend/
├── main.py                        # Point d'entrée FastAPI, CORS, routers
├── requirements.txt
├── data/                          # ← Placez vos datasets ici
│   ├── dataset.csv                # Dataset principal (recommandation + preprocessing)
│   ├── dataset_clean_3.csv        # Dataset avec coordonnées GPS (latt, long)
│   └── tunisia-real-estate.csv    # Dataset statistiques
├── house_price_xgboost_model.pkl  # ← Placez votre modèle ici (racine du projet)
└── app/
    ├── schema.py                  # Modèles Pydantic (HouseFeatures, ClientPreferences)
    ├── core/
    │   ├── data_loader.py         # Chargement dataset + fit préprocesseur sklearn
    │   └── model_loader.py        # Chargement modèle XGBoost (.pkl)
    ├── services/
    │   └── recommender.py         # Logique recommandation (filtres + cosine similarity)
    └── routes/
        ├── prediction.py          # Routes /predict
        ├── recommendations.py     # Routes /recommendation
        └── stats.py               # Routes /stats
```

## Installation et démarrage

```bash
# 1. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Placer les fichiers nécessaires
#    - data/dataset.csv
#    - data/dataset_clean_3.csv
#    - data/tunisia-real-estate.csv
#    - house_price_xgboost_model.pkl  (à la racine)

# 4. Démarrer le serveur
uvicorn main:app --reload --port 8000
```

L'API sera accessible sur : http://127.0.0.1:8000
Documentation Swagger : http://127.0.0.1:8000/docs

## Datasets attendus

| Fichier | Utilisé par | Colonnes minimales |
|---|---|---|
| `dataset.csv` | Recommandation + préprocesseur | `price_tnd, Area, pieces, room, bathroom, distance_to_capital, age_num, location, city, governorate, state, garage, garden, ...` |
| `dataset_clean_3.csv` | Places (GPS) | `location, city, governorate, state, latt, long` |
| `tunisia-real-estate.csv` | Statistiques | `governorate, delegation, locality, nature, real_estate_type, surface, price, inserted_on` |

## CORS

Par défaut autorisé pour `http://localhost:5173` et `http://localhost:5174`.
Pour modifier, éditez `main.py` → `allow_origins`.
