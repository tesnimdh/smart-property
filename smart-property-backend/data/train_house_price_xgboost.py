from __future__ import annotations

from pathlib import Path
import inspect
import unicodedata
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor



BASE_DIR = Path(__file__).resolve().parent

DATA_CANDIDATES = [
    BASE_DIR / "data" / "dataset_clean_3.csv",
    BASE_DIR / "dataset_clean_3.csv",
]

DATA_PATH = next((path for path in DATA_CANDIDATES if path.exists()), DATA_CANDIDATES[0])
MODEL_PATH = DATA_PATH.parent / "house_price_model.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.20
USE_LOG_TARGET = True


USE_RANDOM_SEARCH = True
N_ITER_SEARCH = 50
CV_FOLDS = 5




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

NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES


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

CATEGORICAL_FEATURES = BASE_CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

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


PROPERTY_STATE_ALIASES = {
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




def strip_accents(value: str) -> str:
    """Supprime les accents pour faciliter la comparaison des modalités."""
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_raw_text(value: object) -> str:
    """Normalisation textuelle générale."""
    if value is None:
        return "missing"

    text = str(value).strip().lower()
    if text in MISSING_TEXT_VALUES:
        return "missing"

    return text


def normalize_property_state(value: object) -> str:
    """
    Normalise l'état du bien.

    Le frontend envoie par exemple : 'neuf', 'bon état', 'à rénover'.
    Certains datasets contiennent plutôt : state_0, state_1, state_2.
    Cette fonction impose une représentation commune entre l'entraînement
    et la prédiction.
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


def read_csv_robust(path: Path) -> pd.DataFrame:
    """
    Lit le fichier CSV en privilégiant UTF-8. La colonne d'index créée
    automatiquement par pandas/Excel, par exemple 'Unnamed: 0', est supprimée.
    """
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, regex=True)]
    df.columns = [str(col).strip() for col in df.columns]
    return df


def find_target_column(df: pd.DataFrame) -> str:
    """Détecte automatiquement la colonne cible correspondant au prix."""
    possible_targets = [
        "price_tnd",
        "price",
        "Price",
        "prix",
        "Prix",
        "Price_TND",
        "prix_tnd",
        "price_in_tnd",
        "target",
        "y",
    ]

    columns_by_lower_name = {col.lower(): col for col in df.columns}

    for col in possible_targets:
        if col in df.columns:
            return col
        if col.lower() in columns_by_lower_name:
            return columns_by_lower_name[col.lower()]

    raise ValueError(
        "Impossible de détecter la colonne cible du prix. "
        "Ajoutez son nom dans possible_targets ou renommez-la en 'price_tnd'."
    )


def make_one_hot_encoder() -> OneHotEncoder:
    """Construit un OneHotEncoder compatible avec plusieurs versions de scikit-learn."""
    params = inspect.signature(OneHotEncoder).parameters

    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_numeric_imputer() -> SimpleImputer:
    """Construit un imputer numérique compatible avec plusieurs versions de scikit-learn."""
    params = inspect.signature(SimpleImputer).parameters

    if "keep_empty_features" in params:
        return SimpleImputer(strategy="median", keep_empty_features=True)

    return SimpleImputer(strategy="median")


def ensure_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garantit la présence des colonnes explicatives de base.
    Les variables dérivées sont créées ensuite par add_engineered_features().
    """
    df = df.copy()

    if "Area" not in df.columns and "area" in df.columns:
        df["Area"] = df["area"]

    for col in BASE_NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    for col in BASE_CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "missing"

    return df


def normalize_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les colonnes catégorielles utilisées par le modèle."""
    df = df.copy()

    for col in ["location", "city", "governorate"]:
        df[col] = df[col].map(normalize_raw_text)

    # state = état du bien, pas localisation.
    df["state"] = df["state"].map(normalize_property_state)

    return df


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les variables numériques. Les valeurs invalides deviennent NaN."""
    df = df.copy()

    for col in BASE_NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def normalize_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Force les variables d'équipements en 0/1 lorsque ces colonnes existent."""
    df = df.copy()

    for col in AMENITY_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df[col] = (df[col] > 0).astype(float)

    return df


def remove_outliers(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Supprime les observations manifestement aberrantes.

    Cette étape est volontairement conservatrice : elle nettoie les prix,
    les surfaces et le prix au mètre carré par des bornes réalistes et des
    quantiles. Elle réduit généralement le RMSE, très sensible aux valeurs extrêmes.
    """
    df = df.copy()

    df = df.dropna(subset=[target_col, "Area"])
    df = df[df[target_col] > 0]
    df = df[df["Area"] > 0]

    
    df = df[df["Area"].between(20, 5000)]
    df = df[df[target_col].between(20_000, 5_000_000)]

    if len(df) < 30:
        return df

    price_low, price_high = df[target_col].quantile([0.01, 0.99])
    area_low, area_high = df["Area"].quantile([0.01, 0.99])

    df = df[
        (df[target_col].between(price_low, price_high))
        & (df["Area"].between(area_low, area_high))
    ]

    df["price_per_m2_tmp"] = df[target_col] / df["Area"]
    ppm2_low, ppm2_high = df["price_per_m2_tmp"].quantile([0.01, 0.99])

    df = df[df["price_per_m2_tmp"].between(ppm2_low, ppm2_high)]
    df = df.drop(columns=["price_per_m2_tmp"])

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des variables explicatives supplémentaires destinées à améliorer
    la capacité prédictive du modèle.
    """
    df = df.copy()

    for col in ["Area", "room", "bathroom", "pieces", "age_num", "distance_to_capital"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["log_area"] = np.log1p(df["Area"].clip(lower=0))
    df["area_per_room"] = df["Area"] / df["room"].replace(0, np.nan)
    df["bathroom_per_room"] = df["bathroom"] / df["room"].replace(0, np.nan)
    df["pieces_per_area"] = df["pieces"] / df["Area"].replace(0, np.nan)
    df["amenity_score"] = df[AMENITY_COLUMNS].sum(axis=1)

    # Variables de localisation composées. Elles n'incluent pas state,
    # car state décrit l'état du bien et non sa position géographique.
    df["city_location"] = df["city"].astype(str) + "_" + df["location"].astype(str)
    df["governorate_city"] = df["governorate"].astype(str) + "_" + df["city"].astype(str)

    for col in ENGINEERED_CATEGORICAL_FEATURES:
        df[col] = df[col].map(normalize_raw_text)

    return df


def prepare_dataset(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Exécute toutes les étapes de préparation avant l'entraînement."""
    df = ensure_base_columns(df)
    df = normalize_categorical_columns(df)
    df = clean_numeric_columns(df)
    df = normalize_binary_columns(df)

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])
    df = df[df[target_col] > 0]
    df = remove_outliers(df, target_col)
    df = add_engineered_features(df)

    required_columns = FEATURE_COLUMNS + [target_col]
    df = df[required_columns].copy()

    return df


def make_stratification_bins(y: pd.Series) -> Optional[pd.Series]:
    """Crée des classes de prix pour obtenir un train/test split plus représentatif."""
    try:
        bins = pd.qcut(y, q=10, duplicates="drop")
        if bins.value_counts().min() < 2:
            return None
        return bins
    except ValueError:
        return None


def build_model() -> Pipeline:
    """Construit le pipeline complet : prétraitement + XGBoost."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", make_numeric_imputer()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    regressor = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=3,
        subsample=0.90,
        colsample_bytree=0.90,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=3.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def neg_original_mae_scorer(estimator: Pipeline, x_val: pd.DataFrame, y_val: pd.Series) -> float:
    """
    Score utilisé pendant la recherche d'hyperparamètres.
    Comme le modèle peut être entraîné sur log(price), le score est calculé
    dans l'unité originale du prix, en TND.
    """
    y_pred = estimator.predict(x_val)

    if USE_LOG_TARGET:
        y_true_original = np.expm1(y_val)
        y_pred_original = np.expm1(y_pred)
    else:
        y_true_original = y_val
        y_pred_original = y_pred

    y_pred_original = np.maximum(np.asarray(y_pred_original, dtype=float), 0.0)
    return -mean_absolute_error(y_true_original, y_pred_original)


def tune_or_fit_model(model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series) -> tuple[Pipeline, Optional[dict]]:
    """
    Entraîne le modèle. Si USE_RANDOM_SEARCH=True, une recherche aléatoire
    d'hyperparamètres est effectuée.
    """
    if not USE_RANDOM_SEARCH:
        print("\nEntraînement du modèle sans recherche d'hyperparamètres...")
        model.fit(x_train, y_train)
        return model, None

    param_distributions = {
        "regressor__n_estimators": [500, 800, 1000, 1200, 1500],
        "regressor__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.07],
        "regressor__max_depth": [3, 4, 5, 6, 7, 8],
        "regressor__min_child_weight": [1, 3, 5, 7],
        "regressor__subsample": [0.70, 0.80, 0.90, 1.00],
        "regressor__colsample_bytree": [0.70, 0.80, 0.90, 1.00],
        "regressor__gamma": [0.0, 0.01, 0.05, 0.10, 0.20],
        "regressor__reg_alpha": [0.0, 0.01, 0.10, 1.00],
        "regressor__reg_lambda": [1.0, 3.0, 5.0, 10.0],
    }

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=N_ITER_SEARCH,
        scoring=neg_original_mae_scorer,
        cv=cv,
        verbose=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nRecherche des meilleurs hyperparamètres...")
    search.fit(x_train, y_train)

    print("\nMeilleurs hyperparamètres :")
    print(search.best_params_)

    return search.best_estimator_, search.best_params_


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Évalue le modèle sur l'échantillon de test en unité originale du prix."""
    y_pred = model.predict(x_test)

    if USE_LOG_TARGET:
        y_test_original = np.expm1(y_test)
        y_pred_original = np.expm1(y_pred)
    else:
        y_test_original = y_test
        y_pred_original = y_pred

    y_test_original = np.asarray(y_test_original, dtype=float)
    y_pred_original = np.maximum(np.asarray(y_pred_original, dtype=float), 0.0)

    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    r2 = r2_score(y_test_original, y_pred_original)

    non_zero_mask = y_test_original != 0
    if non_zero_mask.any():
        mape = (
            np.mean(
                np.abs(
                    (y_test_original[non_zero_mask] - y_pred_original[non_zero_mask])
                    / y_test_original[non_zero_mask]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape) if np.isfinite(mape) else np.nan,
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATA_PATH}. "
            "Placez dataset_clean_3.csv dans le dossier data/ ou dans le dossier du script."
        )

    print(f"Dataset utilisé : {DATA_PATH}")
    print(f"Modèle sauvegardé dans : {MODEL_PATH}")

    df_raw = read_csv_robust(DATA_PATH)

    print("\nColonnes disponibles dans le dataset :")
    print(df_raw.columns.tolist())

    target_column = find_target_column(df_raw)
    print(f"\nColonne cible détectée : {target_column}")

    df = prepare_dataset(df_raw, target_column)

    if df.empty:
        raise ValueError(
            "Le dataset est vide après nettoyage. "
            "Vérifiez la colonne cible, les prix, les surfaces et les valeurs manquantes."
        )

    print(f"\nNombre d'observations après nettoyage : {len(df)}")
    print("\nDistribution de la variable state après normalisation :")
    print(df["state"].value_counts(dropna=False))

    x = df[FEATURE_COLUMNS]
    y = df[target_column].astype(float)

    y_model = np.log1p(y) if USE_LOG_TARGET else y
    stratify_bins = make_stratification_bins(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_model,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_bins,
    )

    model = build_model()
    model, best_params = tune_or_fit_model(model, x_train, y_train)

    metrics = evaluate_model(model, x_test, y_test)

    print("\nPerformance sur l'échantillon de test :")
    print(f"MAE  : {metrics['mae']:,.2f} TND")
    print(f"RMSE : {metrics['rmse']:,.2f} TND")
    print(f"R²   : {metrics['r2']:.4f}")
    print(f"MAPE : {metrics['mape']:.2f} %")

    bundle = {
        "model": model,
        "use_log_target": USE_LOG_TARGET,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "base_numeric_features": BASE_NUMERIC_FEATURES,
        "base_categorical_features": BASE_CATEGORICAL_FEATURES,
        "engineered_numeric_features": ENGINEERED_NUMERIC_FEATURES,
        "engineered_categorical_features": ENGINEERED_CATEGORICAL_FEATURES,
        "target_column": target_column,
        "metrics": metrics,
        "best_params": best_params,
        "property_state_aliases": PROPERTY_STATE_ALIASES,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    print(f"\nModèle sauvegardé avec succès dans : {MODEL_PATH}")


if __name__ == "__main__":
    main()
