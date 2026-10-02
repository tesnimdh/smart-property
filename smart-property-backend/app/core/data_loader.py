from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / 'smart-property-backend' /  "data" / "dataset_clean.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset introuvable : {DATA_PATH}\n"
        "Placez votre fichier CSV dans le dossier data/ à la racine du projet."
    )

df = pd.read_csv(DATA_PATH)



numeric_features = [
    "price_tnd",
    "Area",
    "pieces",
    "room",
    "bathroom",
    "distance_to_capital",
    "age_num",
]

categorical_features = [
    "location",
    "city",
    "governorate",
    "state",
]

binary_features = [
    "garage",
    "garden",
    "concierge",
    "beach_view",
    "mountain_view",
    "pool",
    "elevator",
    "furnished",
    "equipped_kitchen",
    "central_heating",
    "air_conditioning",
]


for col in numeric_features + binary_features:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

for col in categorical_features:
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str).str.strip()


try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), numeric_features + binary_features),
        ("cat", ohe, categorical_features),
    ],
    remainder="drop",
)

preprocessor.fit(df[numeric_features + binary_features + categorical_features])

print(f"[data_loader] Dataset chargé — {df.shape[0]} lignes, {df.shape[1]} colonnes")
