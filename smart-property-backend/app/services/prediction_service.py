from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import unicodedata

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pydantic import BaseModel




BASE_NUMERIC_FEATURES = [
    "Area",
    "pieces",
    "room",
    "bathroom",
    "age_num",
    "latt",
    "long",
    "distance_to_capital",
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

ENGINEERED_NUMERIC_FEATURES = [
    "log_area",
    "area_per_room",
    "bathroom_per_room",
    "pieces_per_area",
    "amenity_score",
]

DEFAULT_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES


BASE_CATEGORICAL_FEATURES = [
    "location",
    "city",
    "governorate",
    "state",
]

ENGINEERED_CATEGORICAL_FEATURES = [
    "city_location",
    "governorate_city",
]

DEFAULT_CATEGORICAL_FEATURES = BASE_CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES
DEFAULT_MODEL_FEATURES = DEFAULT_NUMERIC_FEATURES + DEFAULT_CATEGORICAL_FEATURES

AMENITY_COLUMNS = [
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

TARGET_COLUMN_CANDIDATES = [
    "price_tnd",
    "price",
    "Price",
    "Prix",
    "prix",
    "prix_tnd",
    "price_in_tnd",
    "target",
    "y",
]

MISSING_TEXT_VALUES = {
    "",
    "none",
    "nan",
    "null",
    "undefined",
    "missing",
    "non renseigné",
    "non renseigne",
    "ex: 10",
    "ex: cité des oliviers",
    "ex: cite des oliviers",
}


DEFAULT_PROPERTY_STATE_ALIASES = {
    "state_0": "neuf",
    "state_1": "bon_etat",
    "state_2": "a_renover",

    "neuf": "neuf",
    "neuve": "neuf",
    "nouveau": "neuf",
    "nouvelle": "neuf",

    "bon etat": "bon_etat",
    "bon état": "bon_etat",
    "bon_etat": "bon_etat",
    "bon état général": "bon_etat",
    "bon etat general": "bon_etat",

    "a renover": "a_renover",
    "à rénover": "a_renover",
    "a rénover": "a_renover",
    "à renover": "a_renover",
    "a_renover": "a_renover",
    "renover": "a_renover",
    "rénover": "a_renover",
}




def find_project_paths() -> tuple[Path, Path, Path]:
    
    current = Path(__file__).resolve()

    for parent in [current.parent, *current.parents]:
        data_candidates = [
            parent / "data" / "dataset_clean.csv",
            parent / "dataset_clean.csv",
        ]

        for data_path in data_candidates:
            if not data_path.exists():
                continue

            model_candidates = [
                data_path.parent / "house_price_model.pkl",
                parent / "data" / "house_price_model.pkl",
                parent / "house_price_model.pkl",
            ]

            for model_path in model_candidates:
                if model_path.exists():
                    return parent, model_path, data_path

    raise FileNotFoundError(
        "Impossible de trouver simultanément dataset_clean.csv et house_price_model.pkl. "
        "Placez idéalement les deux fichiers dans le dossier data/."
    )


BASE_DIR, MODEL_PATH, DATA_PATH = find_project_paths()




def read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, regex=True)]
    df.columns = [str(col).strip() for col in df.columns]
    return df




bundle = joblib.load(MODEL_PATH)

if isinstance(bundle, dict) and "model" in bundle:
    model = bundle["model"]
    use_log_target = bool(bundle.get("use_log_target", True))
    MODEL_FEATURES = list(bundle.get("feature_columns", DEFAULT_MODEL_FEATURES))
    NUMERIC_FEATURES = list(bundle.get("numeric_features", DEFAULT_NUMERIC_FEATURES))
    CATEGORICAL_FEATURES = list(bundle.get("categorical_features", DEFAULT_CATEGORICAL_FEATURES))
    PROPERTY_STATE_ALIASES = dict(bundle.get("property_state_aliases", DEFAULT_PROPERTY_STATE_ALIASES))
else:
    model = bundle
    use_log_target = True
    MODEL_FEATURES = DEFAULT_MODEL_FEATURES
    NUMERIC_FEATURES = DEFAULT_NUMERIC_FEATURES
    CATEGORICAL_FEATURES = DEFAULT_CATEGORICAL_FEATURES
    PROPERTY_STATE_ALIASES = DEFAULT_PROPERTY_STATE_ALIASES




class HouseFeatures(BaseModel):
    area: Any = None
    pieces: Any = None
    room: Any = None
    bathroom: Any = None
    distance_to_capital: Any = None

    location: Any = None
    city: Any = None
    governorate: Any = None

   
    state: Any = None
    property_state: Any = None 

    garage: Any = None
    garden: Any = None
    concierge: Any = None
    beach_view: Any = None
    mountain_view: Any = None
    pool: Any = None
    elevator: Any = None
    furnished: Any = None
    equipped_kitchen: Any = None
    central_heating: Any = None
    air_conditioning: Any = None

    age_num: Any = None
    latt: Any = None
    long: Any = None




def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def is_missing_value(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        return value.strip().lower() in MISSING_TEXT_VALUES

    return False


def normalize_raw_text(value: Any) -> str:
    if is_missing_value(value):
        return "missing"

    text = str(value).strip().lower()
    return "missing" if text in MISSING_TEXT_VALUES else text


def normalize_property_state(value: Any) -> str:
    """
    Normalise l'état du bien pour être identique à l'entraînement.
    Le frontend peut envoyer : 'neuf', 'bon état', 'à rénover'.
    """
    text = normalize_raw_text(value)
    if text == "missing":
        return "missing"

    no_accent = strip_accents(text)
    candidates = [
        text,
        no_accent,
        text.replace(" ", "_"),
        no_accent.replace(" ", "_"),
    ]

    for candidate in candidates:
        if candidate in PROPERTY_STATE_ALIASES:
            return PROPERTY_STATE_ALIASES[candidate]

    return no_accent.replace(" ", "_")


def to_float_or_nan(value: Any) -> float:
    if is_missing_value(value):
        return np.nan

    try:
        if isinstance(value, str):
            value = value.strip().replace(",", ".")
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def to_binary_or_zero(value: Any) -> float:
    if is_missing_value(value):
        return 0.0

    try:
        if isinstance(value, bool):
            return 1.0 if value else 0.0

        if isinstance(value, str):
            normalized_value = value.strip().lower()

            if normalized_value in {"true", "yes", "oui", "checked", "1"}:
                return 1.0

            if normalized_value in {"false", "no", "non", "unchecked", "0"}:
                return 0.0

            value = normalized_value.replace(",", ".")

        numeric_value = float(value)
        return 1.0 if numeric_value > 0 else 0.0

    except (TypeError, ValueError):
        return 0.0




places_df = read_csv_robust(DATA_PATH)


for col in ["location", "city", "governorate", "latt", "long"]:
    if col not in places_df.columns:
        places_df[col] = np.nan if col in {"latt", "long"} else "missing"

places_df = places_df[["location", "city", "governorate", "latt", "long"]].copy()

for col in ["location", "city", "governorate"]:
    places_df[col] = places_df[col].map(normalize_raw_text)

places_df["latt"] = pd.to_numeric(places_df["latt"], errors="coerce")
places_df["long"] = pd.to_numeric(places_df["long"], errors="coerce")
places_df = places_df.dropna(subset=["latt", "long"])

places_df = (
    places_df
    .groupby(["location", "city", "governorate"], as_index=False)
    .agg({"latt": "mean", "long": "mean"})
)

places = places_df.to_dict(orient="records")


def get_places_data():
    return places





def get_coordinates_from_places(location: str, city: str, governorate: str) -> tuple[float, float]:
    """
    Recherche les coordonnées moyennes selon la localisation.
    state n'intervient volontairement pas ici, car il désigne l'état du bien.
    """
    if places_df.empty:
        return np.nan, np.nan

    filters = [
        (
            places_df["location"].eq(location)
            & places_df["city"].eq(city)
            & places_df["governorate"].eq(governorate)
        ),
        places_df["city"].eq(city) & places_df["governorate"].eq(governorate),
        places_df["governorate"].eq(governorate),
    ]

    for mask in filters:
        matched = places_df[mask]
        if not matched.empty:
            return float(matched["latt"].mean()), float(matched["long"].mean())

    return np.nan, np.nan


def add_engineered_features(row: pd.DataFrame) -> pd.DataFrame:
    row = row.copy()

    for col in ["Area", "room", "bathroom", "pieces", "age_num", "distance_to_capital"]:
        if col not in row.columns:
            row[col] = np.nan
        row[col] = pd.to_numeric(row[col], errors="coerce")

    for col in AMENITY_COLUMNS:
        if col not in row.columns:
            row[col] = 0.0
        row[col] = pd.to_numeric(row[col], errors="coerce").fillna(0.0)
        row[col] = (row[col] > 0).astype(float)

    row["log_area"] = np.log1p(row["Area"].clip(lower=0))
    row["area_per_room"] = row["Area"] / row["room"].replace(0, np.nan)
    row["bathroom_per_room"] = row["bathroom"] / row["room"].replace(0, np.nan)
    row["pieces_per_area"] = row["pieces"] / row["Area"].replace(0, np.nan)
    row["amenity_score"] = row[AMENITY_COLUMNS].sum(axis=1)

    row["city_location"] = row["city"].astype(str) + "_" + row["location"].astype(str)
    row["governorate_city"] = row["governorate"].astype(str) + "_" + row["city"].astype(str)

    return row


def build_model_row(data: HouseFeatures) -> pd.DataFrame:
    """
    Construit une ligne de prédiction contenant exactement les colonnes attendues
    par le pipeline entraîné.
    """
    location = normalize_raw_text(data.location)
    city = normalize_raw_text(data.city)
    governorate = normalize_raw_text(data.governorate)

    # Compatibilité avec l'ancien nom property_state, tout en privilégiant state.
    state_value = data.state if not is_missing_value(data.state) else data.property_state
    state = normalize_property_state(state_value)

    latt = to_float_or_nan(data.latt)
    long = to_float_or_nan(data.long)

    if np.isnan(latt) or np.isnan(long):
        inferred_latt, inferred_long = get_coordinates_from_places(
            location=location,
            city=city,
            governorate=governorate,
        )

        if np.isnan(latt):
            latt = inferred_latt

        if np.isnan(long):
            long = inferred_long

    row = pd.DataFrame(
        [
            {
                "Area": to_float_or_nan(data.area),
                "pieces": to_float_or_nan(data.pieces),
                "room": to_float_or_nan(data.room),
                "bathroom": to_float_or_nan(data.bathroom),
                "age_num": to_float_or_nan(data.age_num),
                "latt": latt,
                "long": long,
                "distance_to_capital": to_float_or_nan(data.distance_to_capital),
                "garage": to_binary_or_zero(data.garage),
                "garden": to_binary_or_zero(data.garden),
                "concierge": to_binary_or_zero(data.concierge),
                "beach_view": to_binary_or_zero(data.beach_view),
                "mountain_view": to_binary_or_zero(data.mountain_view),
                "pool": to_binary_or_zero(data.pool),
                "elevator": to_binary_or_zero(data.elevator),
                "furnished": to_binary_or_zero(data.furnished),
                "equipped_kitchen": to_binary_or_zero(data.equipped_kitchen),
                "central_heating": to_binary_or_zero(data.central_heating),
                "air_conditioning": to_binary_or_zero(data.air_conditioning),
                "location": location,
                "city": city,
                "governorate": governorate,
                "state": state,
            }
        ]
    )

    row = add_engineered_features(row)

    for col in MODEL_FEATURES:
        if col not in row.columns:
            row[col] = np.nan if col in NUMERIC_FEATURES else "missing"

    row = row[MODEL_FEATURES].copy()

    numeric_cols = [col for col in NUMERIC_FEATURES if col in row.columns]
    categorical_cols = [col for col in CATEGORICAL_FEATURES if col in row.columns]

    row[numeric_cols] = row[numeric_cols].apply(pd.to_numeric, errors="coerce")

    for col in categorical_cols:
        if col == "state":
            row[col] = row[col].map(normalize_property_state)
        else:
            row[col] = row[col].map(normalize_raw_text)

    return row




def _round_or_none(value: Any, digits: int = 4) -> Optional[float]:
    try:
        value = float(value)
        if not np.isfinite(value):
            return None
        return round(value, digits)
    except (TypeError, ValueError):
        return None


def _normalize_metrics(metrics: Any) -> Optional[dict[str, Optional[float]]]:
    if not isinstance(metrics, dict):
        return None

    aliases = {
        "r2": ["r2", "R2", "r2_score", "R2_score"],
        "mae": ["mae", "MAE", "mean_absolute_error"],
        "rmse": ["rmse", "RMSE", "root_mean_squared_error"],
        "mape": ["mape", "MAPE", "mean_absolute_percentage_error"],
    }

    normalized: dict[str, Optional[float]] = {}

    for public_name, possible_names in aliases.items():
        found_value = None
        for possible_name in possible_names:
            if possible_name in metrics:
                found_value = metrics[possible_name]
                break

        digits = 4 if public_name == "r2" else 2
        normalized[public_name] = _round_or_none(found_value, digits=digits)

    if all(value is None for value in normalized.values()):
        return None

    return normalized


def _extract_metrics_from_bundle() -> Optional[dict[str, Optional[float]]]:
    if not isinstance(bundle, dict):
        return None

    for key in ["metrics", "model_metrics", "performance", "test_metrics"]:
        metrics = _normalize_metrics(bundle.get(key))
        if metrics is not None:
            return metrics

    return None


def _find_target_column(df: pd.DataFrame) -> Optional[str]:
    columns_by_lower_name = {col.lower(): col for col in df.columns}

    for candidate in TARGET_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate

        lower_candidate = candidate.lower()
        if lower_candidate in columns_by_lower_name:
            return columns_by_lower_name[lower_candidate]

    return None


def _prepare_metric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare le dataset pour un calcul de métriques de secours lorsque les
    métriques n'ont pas été sauvegardées dans le bundle.
    """
    prepared_rows = []

    for _, item in df.iterrows():
        features = HouseFeatures(
            area=item.get("Area", item.get("area")),
            pieces=item.get("pieces"),
            room=item.get("room"),
            bathroom=item.get("bathroom"),
            age_num=item.get("age_num"),
            latt=item.get("latt"),
            long=item.get("long"),
            distance_to_capital=item.get("distance_to_capital"),
            location=item.get("location"),
            city=item.get("city"),
            governorate=item.get("governorate"),
            state=item.get("state"),
            garage=item.get("garage"),
            garden=item.get("garden"),
            concierge=item.get("concierge"),
            beach_view=item.get("beach_view"),
            mountain_view=item.get("mountain_view"),
            pool=item.get("pool"),
            elevator=item.get("elevator"),
            furnished=item.get("furnished"),
            equipped_kitchen=item.get("equipped_kitchen"),
            central_heating=item.get("central_heating"),
            air_conditioning=item.get("air_conditioning"),
        )
        prepared_rows.append(build_model_row(features).iloc[0])

    if not prepared_rows:
        return pd.DataFrame(columns=MODEL_FEATURES)

    return pd.DataFrame(prepared_rows)[MODEL_FEATURES]


def _compute_metrics_from_dataset() -> Optional[dict[str, Optional[float]]]:
    """
    Calcule des métriques de secours à partir du dataset local.
    La méthode recommandée reste de lire les métriques du bundle sauvegardé
    après l'entraînement sur l'échantillon de test.
    """
    try:
        df = read_csv_robust(DATA_PATH)
        target_col = _find_target_column(df)

        if target_col is None:
            return None

        y_true = pd.to_numeric(df[target_col], errors="coerce")
        valid_mask = y_true.notna() & (y_true > 0)

        if valid_mask.sum() == 0:
            return None

        y_true = y_true.loc[valid_mask].astype(float)
        x = _prepare_metric_features(df.loc[valid_mask])

        y_pred = model.predict(x)

        if use_log_target:
            y_pred = np.expm1(y_pred)

        y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
        y_true_values = y_true.to_numpy(dtype=float)

        non_zero_mask = y_true_values != 0
        mape = None
        if non_zero_mask.any():
            mape = (
                np.mean(
                    np.abs(
                        (y_true_values[non_zero_mask] - y_pred[non_zero_mask])
                        / y_true_values[non_zero_mask]
                    )
                )
                * 100
            )

        rmse = np.sqrt(mean_squared_error(y_true_values, y_pred))

        return {
            "r2": _round_or_none(r2_score(y_true_values, y_pred), digits=4),
            "mae": _round_or_none(mean_absolute_error(y_true_values, y_pred), digits=2),
            "rmse": _round_or_none(rmse, digits=2),
            "mape": _round_or_none(mape, digits=2),
        }

    except Exception as exc:
        print(f"Métriques non disponibles : {exc}")
        return None


def get_model_metrics() -> Optional[dict[str, Optional[float]]]:
    metrics = _extract_metrics_from_bundle()
    if metrics is not None:
        return metrics

    return _compute_metrics_from_dataset()


MODEL_METRICS = get_model_metrics()

