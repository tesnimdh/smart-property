import unicodedata

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "Area",
    "pieces",
    "room",
    "bathroom",
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

CATEGORICAL_FEATURES = [
    "location",
    "city",
    "governorate",
    "state",
]

AMENITY_FEATURES = [
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


def _normalize_text(value):
    if value is None:
        return None

    value = str(value).strip().lower()

    if value in ["", "none", "null", "nan", "undefined"]:
        return None

    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))

    return value.strip()


def _normalize_series(series: pd.Series) -> pd.Series:
    return series.apply(_normalize_text).fillna("missing")


def _is_requested(value) -> bool:
    if isinstance(value, bool):
        return value

    try:
        return float(value) >= 1
    except Exception:
        return False


def _find_column_case_insensitive(df: pd.DataFrame, name: str) -> str | None:
    if name in df.columns:
        return name

    lower_map = {col.lower(): col for col in df.columns}

    return lower_map.get(name.lower())


def _find_price_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "price_tnd",
        "predicted_price_tnd",
        "price",
        "Price",
        "prix",
        "Prix",
        "price_dt",
        "Price_TND",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def _mode_or_default(series: pd.Series, default: str = "missing") -> str:
    values = _normalize_series(series)
    values = values[values != "missing"]

    if values.empty:
        return default

    return str(values.mode().iloc[0])


def _median_or_default(series: pd.Series, default: float = 0.0) -> float:
    values = pd.to_numeric(series, errors="coerce")
    median = values.median()

    if pd.isna(median):
        return float(default)

    return float(median)


def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    area_col = _find_column_case_insensitive(data, "Area")
    if area_col is not None and area_col != "Area":
        data["Area"] = data[area_col]

    for col in CATEGORICAL_FEATURES:
        if col not in data.columns:
            data[col] = "missing"

        data[col] = _normalize_series(data[col])

    for col in NUMERIC_FEATURES:
        if col not in data.columns:
            data[col] = 0.0

        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)

    return data


def _apply_filters(data: pd.DataFrame, pref) -> pd.DataFrame:
    filtered = data.copy()

    price_col = _find_price_column(filtered)

    if price_col is not None:
        filtered[price_col] = pd.to_numeric(filtered[price_col], errors="coerce")

        if pref.budget_max is not None:
            filtered = filtered[filtered[price_col] <= float(pref.budget_max)]

    text_filters = {
        "governorate": _normalize_text(pref.governorate),
        "city": _normalize_text(pref.city),
        "location": _normalize_text(pref.location),
        "state": _normalize_text(pref.state),
    }

    for col, value in text_filters.items():
        if value is not None and col in filtered.columns:
            filtered = filtered[filtered[col] == value]

    numeric_min_filters = {
        "Area": pref.min_area,
        "pieces": pref.min_pieces,
        "room": pref.min_room,
        "bathroom": pref.min_bathroom,
    }

    for col, value in numeric_min_filters.items():
        if value is not None and col in filtered.columns:
            filtered = filtered[filtered[col] >= float(value)]

    for col in AMENITY_FEATURES:
        if hasattr(pref, col) and _is_requested(getattr(pref, col)):
            filtered = filtered[filtered[col] >= 1]

    return filtered


def _pref_to_row(pref, reference_data: pd.DataFrame) -> pd.DataFrame:
    row = {
        "Area": (
            float(pref.min_area)
            if pref.min_area is not None
            else _median_or_default(reference_data["Area"])
        ),
        "pieces": (
            float(pref.min_pieces)
            if pref.min_pieces is not None
            else _median_or_default(reference_data["pieces"])
        ),
        "room": (
            float(pref.min_room)
            if pref.min_room is not None
            else _median_or_default(reference_data["room"])
        ),
        "bathroom": (
            float(pref.min_bathroom)
            if pref.min_bathroom is not None
            else _median_or_default(reference_data["bathroom"])
        ),
        "location": (
            _normalize_text(pref.location)
            if _normalize_text(pref.location) is not None
            else _mode_or_default(reference_data["location"])
        ),
        "city": (
            _normalize_text(pref.city)
            if _normalize_text(pref.city) is not None
            else _mode_or_default(reference_data["city"])
        ),
        "governorate": (
            _normalize_text(pref.governorate)
            if _normalize_text(pref.governorate) is not None
            else _mode_or_default(reference_data["governorate"])
        ),
        "state": (
            _normalize_text(pref.state)
            if _normalize_text(pref.state) is not None
            else _mode_or_default(reference_data["state"])
        ),
    }

    for col in AMENITY_FEATURES:
        row[col] = 1.0 if hasattr(pref, col) and _is_requested(getattr(pref, col)) else 0.0

    return pd.DataFrame([row])


def _clean_json_records(records):
    cleaned = []

    for record in records:
        item = {}

        for key, value in record.items():
            if isinstance(value, np.integer):
                item[key] = int(value)

            elif isinstance(value, np.floating):
                if np.isnan(value) or np.isinf(value):
                    item[key] = None
                else:
                    item[key] = float(value)

            elif value is None:
                item[key] = None

            elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                item[key] = None

            else:
                item[key] = value

        cleaned.append(item)

    return cleaned


def recommend_houses(pref, df: pd.DataFrame, top_k: int = 10):
    data = _prepare_data(df)

    if data.empty:
        return []

    filtered = _apply_filters(data, pref)

    if filtered.empty:
        return []

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    data_model = filtered[features].copy()
    pref_row = _pref_to_row(pref, filtered)[features].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    X = preprocessor.fit_transform(data_model)
    x_pref = preprocessor.transform(pref_row)

    n_neighbors = min(top_k, len(filtered))

    knn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute",
    )

    knn.fit(X)

    distances, indices = knn.kneighbors(x_pref)

    results = filtered.iloc[indices[0]].copy()

    similarity = 1 - distances[0]
    similarity = np.clip(similarity, 0, 1)

    results["score"] = similarity * 100

    price_col = _find_price_column(results)

    if price_col is not None:
        results["price_tnd"] = pd.to_numeric(results[price_col], errors="coerce")
    else:
        results["price_tnd"] = np.nan

    output_cols = [
        "price_tnd",
        "location",
        "city",
        "governorate",
        "state",
        "Area",
        "pieces",
        "room",
        "bathroom",
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
        "score",
    ]

    output_cols = [col for col in output_cols if col in results.columns]

    records = results[output_cols].to_dict(orient="records")

    return _clean_json_records(records)