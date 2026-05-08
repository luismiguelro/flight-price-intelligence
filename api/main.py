"""
FastAPI — Flight Price Intelligence
POST /predict: recibe datos del vuelo + precio actual de SerpAPI
               devuelve prediccion, señal Buy/Wait y confianza
"""
import json
import os
import pickle
from datetime import date, datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Rutas de artefactos
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_v1.joblib")
ENC_PATH   = os.path.join(BASE_DIR, "models", "encoders.pkl")
COLS_PATH  = os.path.join(BASE_DIR, "models", "feature_cols.json")

# ---------------------------------------------------------------------------
# Constantes del dataset de entrenamiento
# ---------------------------------------------------------------------------
SCRAPE_DATE = pd.Timestamp("2022-02-11")

STOP_MAP  = {"non-stop": 0, "1-stop": 1, "2+-stop": 2}
CLASS_MAP = {"Economy": 0, "Business": 1}

# SerpAPI devuelve nombres distintos a los del dataset de entrenamiento
AIRLINE_NORMALIZE = {
    "indigo":             "Indigo",
    "air india":          "Air India",
    "air india express":  "Air India",
    "airasia":            "AirAsia",
    "air asia":           "AirAsia",
    "go first":           "GO FIRST",
    "go!":                "GO FIRST",
    "spicejet":           "SpiceJet",
    "starair":            "StarAir",
    "star air":           "StarAir",
    "trujet":             "Trujet",
    "vistara":            "Vistara",
}

CITY_TO_IATA = {
    "Delhi": "DEL", "Mumbai": "BOM", "Bangalore": "BLR",
    "Kolkata": "CCU", "Hyderabad": "HYD", "Chennai": "MAA",
}
VALID_CITIES = set(CITY_TO_IATA.keys())

# ---------------------------------------------------------------------------
# Carga de artefactos al iniciar (una sola vez)
# ---------------------------------------------------------------------------
model    = joblib.load(MODEL_PATH)
encoders = pickle.load(open(ENC_PATH, "rb"))
with open(COLS_PATH) as f:
    FEATURE_COLS = json.load(f)

app = FastAPI(
    title="Flight Price Intelligence API",
    description="Predice si el precio actual de un vuelo es bueno para comprar o conviene esperar.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    origin:        str          # "Delhi", "Mumbai", etc.
    destination:   str
    flight_date:   date         # YYYY-MM-DD
    departure_time: str         # "HH:MM"
    airline:       str          # "IndiGo", "Air India", etc.
    flight_class:  str = "Economy"   # "Economy" | "Business"
    stops:         str = "1-stop"    # "non-stop" | "1-stop" | "2+-stop"
    duration_minutes: int       # duración total en minutos
    current_price: float        # precio actual de SerpAPI en INR

    @field_validator("origin", "destination")
    @classmethod
    def validate_city(cls, v):
        if v not in VALID_CITIES:
            raise ValueError(f"Ciudad no soportada: '{v}'. Válidas: {sorted(VALID_CITIES)}")
        return v

    @field_validator("flight_class")
    @classmethod
    def validate_class(cls, v):
        if v not in CLASS_MAP:
            raise ValueError(f"Clase inválida: '{v}'. Usar 'Economy' o 'Business'")
        return v

    @field_validator("stops")
    @classmethod
    def validate_stops(cls, v):
        if v not in STOP_MAP:
            raise ValueError(f"Stops inválido: '{v}'. Usar 'non-stop', '1-stop' o '2+-stop'")
        return v


class PredictResponse(BaseModel):
    predicted_price:  float
    current_price:    float
    signal:           str    # "BUY" | "WAIT"
    signal_es:        str    # "Compra ahora" | "Espera"
    confidence:       float  # 0–1
    price_diff_pct:   float  # % diferencia actual vs prediccion
    explanation:      str


# ---------------------------------------------------------------------------
# Lógica de features (espeja 02_feature_engineering.py)
# ---------------------------------------------------------------------------
def _normalize_airline(name: str) -> str:
    normalized = AIRLINE_NORMALIZE.get(name.lower().strip())
    if normalized is None:
        raise ValueError(f"Aerolínea no reconocida: '{name}'. Válidas: {sorted(set(AIRLINE_NORMALIZE.values()))}")
    return normalized


def build_input(req: PredictRequest) -> pd.DataFrame:
    flight_dt = pd.Timestamp(req.flight_date)
    dep_hour  = int(req.departure_time.split(":")[0])

    time_of_day = pd.cut(
        pd.Series([dep_hour]),
        bins=[-1, 5, 11, 17, 23],
        labels=[0, 1, 2, 3],
    ).astype(int).iloc[0]

    route = f"{req.origin}_{req.destination}"

    row = {
        "month":             flight_dt.month,
        "day_of_week":       flight_dt.dayofweek,
        "is_weekend":        int(flight_dt.dayofweek >= 5),
        "days_until_flight": (flight_dt - SCRAPE_DATE).days,
        "dep_hour":          dep_hour,
        "time_of_day":       int(time_of_day),
        "duration_minutes":  req.duration_minutes,
        "stop_enc":          STOP_MAP[req.stops],
        "class_enc":         CLASS_MAP[req.flight_class],
        "airline_enc":       encoders["airline"].transform([_normalize_airline(req.airline)])[0],
        "source_enc":        encoders["from"].transform([req.origin])[0],
        "dest_enc":          encoders["to"].transform([req.destination])[0],
        "route_enc":         encoders["route"].transform([route])[0],
    }
    return pd.DataFrame([row])[FEATURE_COLS]


def compute_signal(current: float, predicted: float) -> tuple[str, str, float, float]:
    diff_pct = (current - predicted) / predicted * 100

    # Confianza basada en qué tan lejos está el precio actual del predicho
    # >20% más caro  → alta confianza para WAIT
    # <-10% más barato → alta confianza para BUY
    abs_diff = abs(diff_pct)
    confidence = min(abs_diff / 30, 1.0)   # normalizado a [0, 1]

    if current > predicted * 1.05:
        return "WAIT", "Espera", round(confidence, 2), round(diff_pct, 1)
    else:
        return "BUY", "Compra ahora", round(confidence, 2), round(diff_pct, 1)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "model": "xgboost_v1", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        X = build_input(req)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    predicted = float(model.predict(X)[0])
    signal, signal_es, confidence, diff_pct = compute_signal(req.current_price, predicted)

    if signal == "WAIT":
        explanation = (
            f"El precio actual ({req.current_price:,.0f} INR) está "
            f"{abs(diff_pct):.1f}% por encima del precio esperado "
            f"({predicted:,.0f} INR). Conviene esperar."
        )
    else:
        explanation = (
            f"El precio actual ({req.current_price:,.0f} INR) está "
            f"en línea o por debajo del precio esperado ({predicted:,.0f} INR). "
            f"Es buen momento para comprar."
        )

    return PredictResponse(
        predicted_price=round(predicted, 0),
        current_price=req.current_price,
        signal=signal,
        signal_es=signal_es,
        confidence=confidence,
        price_diff_pct=diff_pct,
        explanation=explanation,
    )
