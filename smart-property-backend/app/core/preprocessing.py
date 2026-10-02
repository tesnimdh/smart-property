import pandas as pd
import numpy as np


INPUT_PATH = "./data/dataset.csv"

OUTPUT_PATH = "./data/dataset_clean.csv"




df = pd.read_csv(INPUT_PATH)

print("Shape initial :", df.shape)



required_raw_cols = [

    "price_tnd",
    "location",
    "city",
    "governorate",
    "state",
    "Area",
    "pieces",
    "room",
    "bathroom",
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
    "age_num"

]

missing_raw_cols = [
    col for col in required_raw_cols
    if col not in df.columns
]

if missing_raw_cols:

    raise ValueError(
        f"Colonnes manquantes dans le dataset source : {missing_raw_cols}"
    )




df = df.drop(
    columns=["id", "price_eur", "age"],
    errors="ignore"
)




text_cols = [
    "location",
    "city",
    "governorate"
]

for col in text_cols:

    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(
            ["nan", "none", "null", ""],
            np.nan
        )
    )






numeric_cols = [

    "Area",
    "pieces",
    "room",
    "bathroom",
    "age_num",
    "latt",
    "long",
    "distance_to_capital",
    "price_tnd"

]




binary_cols = [

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
    "air_conditioning"

]




for col in numeric_cols + binary_cols:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )



categorical_cols = [

    "location",
    "city",
    "governorate",
    "state"

]




before_drop = len(df)

df = df.dropna(
    subset=["price_tnd"] + categorical_cols
)

after_drop = len(df)




before_dupes = len(df)

df = df.drop_duplicates()

after_dupes = len(df)




print(
    "\nLignes supprimées pour valeurs critiques manquantes :",
    before_drop - after_drop
)

print(
    "Lignes supprimées pour doublons :",
    before_dupes - after_dupes
)

print(
    "Shape final :",
    df.shape
)




print("\nValeurs manquantes par colonne :")

print(
    df.isnull().sum()
)




print("\nAperçu du dataset nettoyé :")

print(
    df.head()
)




df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nDataset saved to {OUTPUT_PATH}"
)