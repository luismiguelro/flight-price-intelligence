# Flight Price Intelligence ✈️

> **¿Compro el vuelo ahora o espero?**
> Compara el precio real de Google Flights contra la predicción del modelo y emite una señal **Compra ahora / Espera** con porcentaje de confianza.

[![API](https://img.shields.io/badge/API-Render-46E3B7?logo=render)](https://flight-price-intelligence.onrender.com/docs)
[![Demo](https://img.shields.io/badge/Demo-Streamlit-FF4B4B?logo=streamlit)](https://flight-price-intelligence.streamlit.app)
[![README in English](https://img.shields.io/badge/README-English-blue)](README.md)

---

## El problema

Los precios de los vuelos cambian decenas de veces al día. Sin un punto de referencia, el viajero no sabe si el precio que ve es bueno o si conviene esperar. Este proyecto construye ese punto de referencia: un modelo entrenado con 300k vuelos domésticos de India que predice el precio esperado para una ruta y, al compararlo con el precio actual en Google Flights, emite una señal accionable.

---

## Demo en vivo

1. Selecciona origen, destino y fecha
2. La app consulta Google Flights (via SerpAPI) y muestra las opciones reales
3. El modelo predice el precio esperado para esa ruta
4. Ves la señal **✅ Compra ahora** o **⏳ Espera** con % de confianza y gráfico de curva de precios por anticipación

---

## Arquitectura

```
ENTRENAMIENTO
─────────────────────────────────────────────────────────────────────
Dataset Kaggle (300k vuelos India, Feb–Mar 2022)
  │
  ├─ Supabase PostgreSQL  ←  carga raw (raw_flights)
  │
  ├─ Feature Engineering
  │     days_until_flight · month · day_of_week · is_weekend
  │     dep_hour · time_of_day · duration_minutes
  │     stops · class · airline · source · destination · route
  │
  ├─ MLflow Tracking
  │     Linear Regression  →  MAE 9,842  R² 0.71
  │     KNN (k=10)         →  MAE 4,201  R² 0.88
  │     XGBoost            →  MAE 2,088  R² 0.97  ✅ seleccionado
  │
  └─ Artefactos: xgboost_v1.joblib · encoders.pkl · feature_cols.json

INFERENCIA EN VIVO
─────────────────────────────────────────────────────────────────────
Usuario ingresa origen / destino / fecha
  │
  ├─ SerpAPI (Google Flights)  →  precio actual real (INR)
  │
  ├─ FastAPI /predict (Render)
  │     build_features(request)  →  model.predict()
  │     señal BUY / WAIT + confianza + explicación
  │
  ├─ FastAPI /price_curve (Render)
  │     predicciones para 10 ventanas de anticipación (1–60 días)
  │
  └─ Streamlit Cloud
        tarjetas de vuelo estilo Google Flights
        gauge precio actual vs predicho
        line chart curva de precio por anticipación
```

---

## Por qué XGBoost

| Modelo | MAE (INR) | R² | Tiempo entrenamiento |
|---|---:|---:|---|
| Linear Regression | 9,842 | 0.71 | < 1s |
| KNN (k=10) | 4,201 | 0.88 | 8s |
| **XGBoost** | **2,088** | **0.97** | 45s |

**Linear Regression** captura la tendencia general pero no modela la no-linealidad del precio por ruta: una misma distancia puede costar 3× más si la aerolínea es Vistara Business vs IndiGo Economy.

**KNN** mejora al comparar vuelos similares, pero es sensible a la escala y a rutas con pocos registros (< 7k vuelos). Las rutas cortas tipo Chennai→Hyderabad producen predicciones ruidosas.

**XGBoost** captura interacciones entre features (ruta × aerolínea × días de anticipación) sin normalización manual. Con `n_estimators=300, max_depth=6, learning_rate=0.05` logra R² 0.97 y un MAE de ~2,000 INR sobre un rango de precios de 2,000–115,000 INR (~1.7% de error relativo en la mediana).

---

## Decisiones técnicas

**`days_until_flight` como proxy de booking window**
El dataset de Kaggle no registra la fecha de compra, solo la fecha de vuelo y la fecha de scraping (2022-02-11). Se aproximó `days_until_flight = flight_date − scrape_date`. Esto captura la variación de precio por anticipación de forma relativa, aunque no es una fecha de compra real.

**Señal BUY/WAIT con zona de tolerancia del 5%**
Un precio actual dentro del ±5% del precio predicho se clasifica como BUY. Por encima del 5% es WAIT. La confianza se normaliza a `min(|diff_pct| / 30, 1.0)`: a mayor distancia del precio predicho, mayor certeza de la señal.

**FastAPI deployada en Render (Free tier)**
El modelo pesa ~8MB. Render Free hace cold start de ~30s si lleva inactivo. Se expuso `/health` para keep-alive pings si fuera necesario.

**`/price_curve` server-side en lugar de 10 calls desde el cliente**
En lugar de que Streamlit llame 10 veces a `/predict` para construir el gráfico, un único `GET /price_curve` genera todas las predicciones del lado del servidor. Reduce latencia y simplifica el código cliente.

---

## Model Card

| Campo | Detalle |
|---|---|
| Modelo | XGBoost Regressor |
| Dataset | [Ease My Trip Flight Dataset](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) — 300,261 vuelos |
| Cobertura | Vuelos domésticos India, Feb–Mar 2022 |
| Features | 13 (temporales, ruta, aerolínea, clase, escalas, duración) |
| Métrica principal | MAE 2,088 INR · R² 0.97 |
| Artefacto | `models/xgboost_v1.joblib` |
| Tracking | MLflow local (`mlruns/`) |
| Limitaciones | Precios INR de 2022 · Solo 6 ciudades · Sin fecha de compra real |

---

## Stack

| Capa | Tecnología |
|---|---|
| Datos | Kaggle CSV → Supabase PostgreSQL |
| Feature Engineering | Python · pandas · scikit-learn |
| Experimentación | MLflow (local) |
| Modelo | XGBoost |
| API | FastAPI · Render |
| Precios en vivo | SerpAPI (Google Flights) |
| Frontend | Streamlit Cloud |

---

## Rutas soportadas

6 ciudades · 30 rutas · ver [`docs/supported_routes.md`](docs/supported_routes.md) para volumen de datos y aerolíneas por ruta.

---

## Instalación local

```bash
git clone https://github.com/luismiguelro/flight-price-intelligence.git
cd flight-price-intelligence

pip install -r requirements-dev.txt

# Variables de entorno
cp .env.example .env
# Editar .env con SERPAPI_KEY y DATABASE_URL

# Levantar la API
uvicorn api.main:app --reload

# En otra terminal, levantar la app
streamlit run streamlit_app.py
```

---

## Estructura del repo

```
flight-price-intelligence/
├── api/
│   └── main.py              # FastAPI: /predict + /price_curve
├── models/
│   ├── xgboost_v1.joblib    # Modelo exportado
│   ├── encoders.pkl         # LabelEncoders de aerolínea/ruta/ciudad
│   └── feature_cols.json    # Orden de features del modelo
├── notebooks/
│   ├── 01_eda.ipynb         # Análisis exploratorio (10 secciones)
│   └── 02_feature_engineering.py
├── src/
│   └── train_models.py      # MLflow: Linear, KNN, XGBoost
├── docs/
│   └── supported_routes.md  # Rutas y aerolíneas disponibles
├── streamlit_app.py         # App completa
├── requirements.txt         # Dependencias API + Streamlit Cloud
└── requirements-dev.txt     # Dependencias de entrenamiento (MLflow, etc.)
```
