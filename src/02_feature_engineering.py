"""
Transforma los CSV crudos en features ML-ready.
Salidas:
  - data/processed/features.parquet  -> X + y listos para entrenar
  - models/encoders.pkl              -> LabelEncoders para inference
"""
import os
import pickle
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder

RAW_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Fecha minima del dataset = proxy de fecha de scraping para calcular dias_anticipacion
SCRAPE_DATE = pd.Timestamp("2022-02-11")

STOP_MAP  = {"non-stop": 0, "1-stop": 1, "2+-stop": 2}
CLASS_MAP = {"Economy": 0, "Business": 1}

FEATURE_COLS = [
    "month", "day_of_week", "is_weekend", "days_until_flight",
    "dep_hour", "time_of_day", "duration_minutes",
    "stop_enc", "class_enc",
    "airline_enc", "source_enc", "dest_enc", "route_enc",
    "price",
]


def _parse_duration_minutes(s: str) -> float | None:
    m = re.match(r"(\d+)h\s*(\d+)m", str(s).strip())
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return None


def build_features(df_raw: pd.DataFrame, encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Transforma el df crudo (columnas originales del CSV) en features ML-ready.
    - Si encoders es None  -> ajusta LabelEncoders (training).
    - Si encoders se pasa  -> los aplica sin re-entrenar (inference).
    Retorna (features_df, encoders_dict).
    """
    df = df_raw.copy()

    # Precio (target)
    df["price"] = df["price"].astype(str).str.replace(",", "", regex=False).astype(float)

    # Fecha
    df["date"]              = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["month"]             = df["date"].dt.month
    df["day_of_week"]       = df["date"].dt.dayofweek        # 0=Lunes, 6=Domingo
    df["is_weekend"]        = (df["day_of_week"] >= 5).astype(int)
    df["days_until_flight"] = (df["date"] - SCRAPE_DATE).dt.days

    # Hora de salida
    df["dep_hour"] = df["dep_time"].str.split(":").str[0].astype(int)
    df["time_of_day"] = pd.cut(
        df["dep_hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=[0, 1, 2, 3],     # 0=madrugada, 1=manana, 2=tarde, 3=noche
    ).astype(int)

    # Duracion en minutos
    df["duration_minutes"] = df["time_taken"].apply(_parse_duration_minutes)
    median_dur = df["duration_minutes"].median()
    df["duration_minutes"] = df["duration_minutes"].fillna(median_dur).astype(int)

    # Paradas (ordinal)
    stop_clean = df["stop"].astype(str).str.extract(r"(non-stop|1-stop|2\+-stop)", expand=False)
    df["stop_enc"] = stop_clean.map(STOP_MAP).fillna(1).astype(int)

    # Clase (binaria)
    df["class_enc"] = df["class"].map(CLASS_MAP)

    # Ruta combinada
    df["route"] = df["from"].str.strip() + "_" + df["to"].str.strip()

    # Encoders categoriales
    cat_cols = {"airline": "airline_enc", "from": "source_enc", "to": "dest_enc", "route": "route_enc"}
    if encoders is None:
        encoders = {}
        for col, enc_col in cat_cols.items():
            le = LabelEncoder()
            df[enc_col] = le.fit_transform(df[col].str.strip())
            encoders[col] = le
    else:
        for col, enc_col in cat_cols.items():
            df[enc_col] = encoders[col].transform(df[col].str.strip())

    return df[FEATURE_COLS].copy(), encoders


def _assert_features(df: pd.DataFrame) -> None:
    null_counts = df.isnull().sum()
    assert null_counts.sum() == 0, f"Nulls encontrados:\n{null_counts[null_counts > 0]}"
    assert df["price"].min() > 0, "Precio <= 0 detectado"
    assert set(df["stop_enc"].unique()).issubset({0, 1, 2}), "stop_enc fuera de rango"
    assert set(df["class_enc"].unique()).issubset({0, 1}), "class_enc fuera de rango"
    assert df["duration_minutes"].min() > 0, "duration_minutes invalido"
    assert df["days_until_flight"].min() >= 0, "days_until_flight negativo"
    print("  Asserts OK")


if __name__ == "__main__":
    print("=== Feature Engineering ===")

    print("\n[1/4] Leyendo CSVs...")
    eco = pd.read_csv(os.path.join(RAW_DIR, "economy.csv"), encoding="utf-8-sig")
    bus = pd.read_csv(os.path.join(RAW_DIR, "business.csv"), encoding="utf-8-sig")
    eco["class"] = "Economy"
    bus["class"] = "Business"
    df_raw = pd.concat([eco, bus], ignore_index=True)
    print(f"  {len(df_raw):,} filas cargadas")

    print("\n[2/4] Construyendo features...")
    features, encoders = build_features(df_raw)
    print(f"  Shape: {features.shape}")
    print(f"  Columnas: {list(features.columns)}")
    print(f"  Precio medio: {features['price'].mean():,.0f} INR")
    print(f"  Dias anticipacion - min: {features['days_until_flight'].min()}, max: {features['days_until_flight'].max()}")
    print(f"  Clases aereas: {features['airline_enc'].nunique()} aerolineas encodeadas")
    print(f"  Rutas unicas: {features['route_enc'].nunique()}")

    print("\n[3/4] Validando...")
    _assert_features(features)

    print("\n[4/4] Guardando artefactos...")
    os.makedirs(PROC_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    out_path = os.path.join(PROC_DIR, "features.parquet")
    features.to_parquet(out_path, index=False)
    print(f"  features.parquet guardado ({features.shape[0]:,} filas x {features.shape[1]} cols)")

    enc_path = os.path.join(MODEL_DIR, "encoders.pkl")
    with open(enc_path, "wb") as f:
        pickle.dump(encoders, f)
    print(f"  encoders.pkl guardado — keys: {list(encoders.keys())}")

    print("\nDone.")
