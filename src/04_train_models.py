"""
Entrena 3 modelos con MLflow tracking: Linear Regression, KNN, XGBoost.
Compara contra el baseline de 03_train_baseline.py.

Uso:
    python src/04_train_models.py
"""
import os
import time

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

PROC_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MLFLOW_DB = f"sqlite:///{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlflow.db'))}"

EXPERIMENT_NAME = "flight-price-intelligence"
RANDOM_STATE    = 42
TEST_SIZE       = 0.2

FEATURE_COLS = [
    "month", "day_of_week", "is_weekend", "days_until_flight",
    "dep_hour", "time_of_day", "duration_minutes",
    "stop_enc", "class_enc",
    "airline_enc", "source_enc", "dest_enc", "route_enc",
]


def metrics(y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"mae": round(mae, 2), "r2": round(r2, 4), "mape": round(mape, 2)}


def run_experiment(name: str, model, params: dict, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test",  len(X_test))

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - t0, 1)

        y_pred = model.predict(X_test)
        m = metrics(y_test.to_numpy(), y_pred)

        mlflow.log_metrics(m)
        mlflow.log_metric("train_time_sec", train_time)

        # Loguear modelo
        if "xgb" in name.lower():
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"  {name:<25} MAE: {m['mae']:>10,.0f} INR  |  R2: {m['r2']:.4f}  |  MAPE: {m['mape']:.1f}%  |  {train_time}s")
        return m


def main():
    print("=== Entrenamiento 3 modelos ===\n")

    mlflow.set_tracking_uri(MLFLOW_DB)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("[1/2] Cargando features...")
    df = pd.read_parquet(os.path.join(PROC_DIR, "features.parquet"))
    X = df[FEATURE_COLS]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    print("[2/2] Corriendo experimentos...")
    print(f"  {'Modelo':<25} {'MAE':>14}        R2       MAPE    Tiempo")
    print("  " + "-" * 70)

    models = [
        (
            "linear_regression",
            LinearRegression(),
            {"model_type": "linear_regression"},
        ),
        (
            "knn_k15",
            KNeighborsRegressor(n_neighbors=15, n_jobs=-1),
            {"model_type": "knn", "n_neighbors": 15},
        ),
        (
            "xgboost",
            XGBRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            ),
            {
                "model_type":       "xgboost",
                "n_estimators":     400,
                "max_depth":        6,
                "learning_rate":    0.05,
                "subsample":        0.8,
                "colsample_bytree": 0.8,
            },
        ),
    ]

    results = {}
    for name, model, params in models:
        results[name] = run_experiment(name, model, params, X_train, X_test, y_train, y_test)

    print("\n=== Resumen ===")
    best = min(results, key=lambda k: results[k]["mae"])
    print(f"  Mejor modelo: {best}  (MAE: {results[best]['mae']:,.0f} INR)")
    print(f"\n  Ver en MLflow UI:")
    print(f"  .venv\\Scripts\\mlflow ui --backend-store-uri {MLFLOW_DB} --port 5000")


if __name__ == "__main__":
    main()
