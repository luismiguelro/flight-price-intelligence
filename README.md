# Flight Price Intelligence ✈️

> **Should I buy this flight now or wait?**
> Compares real Google Flights prices against a machine learning model prediction and outputs a **Buy Now / Wait** signal with confidence score.

[![API](https://img.shields.io/badge/API-Render-46E3B7?logo=render)](https://flight-price-intelligence.onrender.com/docs)
[![Demo](https://img.shields.io/badge/Demo-Streamlit-FF4B4B?logo=streamlit)](https://flight-price-intelligence.streamlit.app)
[![README en Español](https://img.shields.io/badge/README-Español-blue)](README.es.md)

---

## The Problem

Flight prices change dozens of times a day. Without a reference point, travelers have no way to know if the price they see is fair or if waiting would save them money. This project builds that reference point: a model trained on 300k domestic India flights that predicts the expected price for a route and, when compared to the live Google Flights price, outputs an actionable signal.

---

## Live Demo

1. Select origin, destination, and travel date
2. The app queries Google Flights (via SerpAPI) and displays real flight options
3. The model predicts the expected price for that route
4. You get a **✅ Buy Now** or **⏳ Wait** signal with confidence % and a price curve chart showing how prices typically behave by booking window

---

## Architecture

```
TRAINING
─────────────────────────────────────────────────────────────────────
Kaggle Dataset (300k India domestic flights, Feb–Mar 2022)
  │
  ├─ Supabase PostgreSQL  ←  raw load (raw_flights table)
  │
  ├─ Feature Engineering
  │     days_until_flight · month · day_of_week · is_weekend
  │     dep_hour · time_of_day · duration_minutes
  │     stops · class · airline · source · destination · route
  │
  ├─ MLflow Experiment Tracking
  │     Linear Regression  →  MAE 9,842  R² 0.71
  │     KNN (k=10)         →  MAE 4,201  R² 0.88
  │     XGBoost            →  MAE 2,088  R² 0.97  ✅ selected
  │
  └─ Artifacts: xgboost_v1.joblib · encoders.pkl · feature_cols.json

LIVE INFERENCE
─────────────────────────────────────────────────────────────────────
User inputs origin / destination / date
  │
  ├─ SerpAPI (Google Flights)  →  live price in INR
  │
  ├─ FastAPI /predict (Render)
  │     build_features(request)  →  model.predict()
  │     BUY / WAIT signal + confidence + explanation
  │
  ├─ FastAPI /price_curve (Render)
  │     predictions for 10 booking windows (1–60 days ahead)
  │
  └─ Streamlit Cloud
        Google Flights-style flight cards
        gauge: current price vs predicted price
        line chart: price curve by booking window
```

---

## Why XGBoost

| Model | MAE (INR) | R² | Training time |
|---|---:|---:|---|
| Linear Regression | 9,842 | 0.71 | < 1s |
| KNN (k=10) | 4,201 | 0.88 | 8s |
| **XGBoost** | **2,088** | **0.97** | 45s |

**Linear Regression** captures the general trend but fails to model the non-linearity of flight pricing: the same route can cost 3× more depending on airline (Vistara Business vs IndiGo Economy) or booking window.

**KNN** improves by comparing similar flights, but is sensitive to feature scale and underperforms on routes with sparse data (< 7k flights). Short routes like Chennai→Hyderabad produce noisy predictions.

**XGBoost** captures interactions between features (route × airline × days until flight) without manual normalization. With `n_estimators=300, max_depth=6, learning_rate=0.05` it achieves R² 0.97 and a MAE of ~2,000 INR over a price range of 2,000–115,000 INR (~1.7% relative error at the median).

---

## Technical Decisions

**`days_until_flight` as a booking window proxy**
The Kaggle dataset does not record the purchase date — only the flight date and the scraping date (2022-02-11). `days_until_flight = flight_date − scrape_date` was used as a proxy. This captures relative price variation by anticipation, though it is not a true purchase date.

**BUY/WAIT signal with a 5% tolerance band**
A current price within ±5% of the predicted price is classified as BUY. Above 5% is WAIT. Confidence is normalized as `min(|diff_pct| / 30, 1.0)`: the further the current price is from the prediction, the higher the signal confidence.

**FastAPI on Render (Free tier)**
The model weighs ~8MB. Render Free has a ~30s cold start after inactivity. A `/health` endpoint is exposed for keep-alive pings if needed.

**Server-side `/price_curve` instead of 10 client calls**
Instead of Streamlit calling `/predict` 10 times to build the booking-window chart, a single `GET /price_curve` generates all predictions server-side. Reduces latency and simplifies the client.

---

## Model Card

| Field | Detail |
|---|---|
| Model | XGBoost Regressor |
| Dataset | [Ease My Trip Flight Dataset](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) — 300,261 flights |
| Coverage | Domestic India flights, Feb–Mar 2022 |
| Features | 13 (temporal, route, airline, class, stops, duration) |
| Primary metric | MAE 2,088 INR · R² 0.97 |
| Artifact | `models/xgboost_v1.joblib` |
| Experiment tracking | MLflow local (`mlruns/`) |
| Limitations | INR prices from 2022 · 6 cities only · No real purchase date |

---

## Stack

| Layer | Technology |
|---|---|
| Data | Kaggle CSV → Supabase PostgreSQL |
| Feature Engineering | Python · pandas · scikit-learn |
| Experimentation | MLflow (local) |
| Model | XGBoost |
| API | FastAPI · Render |
| Live prices | SerpAPI (Google Flights) |
| Frontend | Streamlit Cloud |

---

## Supported Routes

6 cities · 30 routes · see [`docs/supported_routes.md`](docs/supported_routes.md) for data volume and airlines per route.

---

## Local Setup

```bash
git clone https://github.com/luismiguelro/flight-price-intelligence.git
cd flight-price-intelligence

pip install -r requirements-dev.txt

# Environment variables
cp .env.example .env
# Edit .env with SERPAPI_KEY and DATABASE_URL

# Start the API
uvicorn api.main:app --reload

# In another terminal, start the app
streamlit run streamlit_app.py
```

---

## Repository Structure

```
flight-price-intelligence/
├── api/
│   └── main.py              # FastAPI: /predict + /price_curve
├── models/
│   ├── xgboost_v1.joblib    # Exported model
│   ├── encoders.pkl         # LabelEncoders for airline/route/city
│   └── feature_cols.json    # Feature order expected by the model
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory analysis (10 sections)
│   └── 02_feature_engineering.py
├── src/
│   └── train_models.py      # MLflow: Linear, KNN, XGBoost
├── docs/
│   └── supported_routes.md  # Available routes and airlines
├── streamlit_app.py         # Full Streamlit app
├── requirements.txt         # API + Streamlit Cloud dependencies
└── requirements-dev.txt     # Training dependencies (MLflow, etc.)
```
