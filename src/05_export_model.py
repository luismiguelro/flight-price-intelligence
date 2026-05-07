"""
Exporta el mejor modelo (XGBoost) a joblib y lo registra en MLflow Model Registry.
Salidas:
  - models/xgboost_v1.joblib     -> modelo para FastAPI
  - models/feature_cols.json     -> orden exacto de features para inference
"""
import json
import os

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

PROC_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MLFLOW_DB = f"sqlite:///{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlflow.db'))}"

EXPERIMENT_NAME  = "flight-price-intelligence"
REGISTERED_NAME  = "flight-price-xgboost"
RANDOM_STATE     = 42
TEST_SIZE        = 0.2

FEATURE_COLS = [
    "month", "day_of_week", "is_weekend", "days_until_flight",
    "dep_hour", "time_of_day", "duration_minutes",
    "stop_enc", "class_enc",
    "airline_enc", "source_enc", "dest_enc", "route_enc",
]

BEST_PARAMS = {
    "n_estimators":     400,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
    "verbosity":        0,
}


def main():
    print("=== Exportar modelo XGBoost ===\n")

    mlflow.set_tracking_uri(MLFLOW_DB)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("[1/4] Cargando features...")
    df = pd.read_parquet(os.path.join(PROC_DIR, "features.parquet"))
    X = df[FEATURE_COLS]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    print("\n[2/4] Entrenando XGBoost final...")
    model = XGBRegressor(**BEST_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test.to_numpy() - y_pred) / y_test.to_numpy())) * 100
    print(f"  MAE : {mae:,.0f} INR")
    print(f"  R2  : {r2:.4f}")
    print(f"  MAPE: {mape:.1f}%")

    print("\n[3/4] Registrando en MLflow Model Registry...")
    with mlflow.start_run(run_name="xgboost_v1_final"):
        mlflow.log_params(BEST_PARAMS)
        mlflow.log_metric("mae",  round(mae, 2))
        mlflow.log_metric("r2",   round(r2, 4))
        mlflow.log_metric("mape", round(mape, 2))
        mlflow.log_param("feature_cols", FEATURE_COLS)

        model_info = mlflow.xgboost.log_model(
            model,
            name=REGISTERED_NAME,
            registered_model_name=REGISTERED_NAME,
        )
        print(f"  Modelo registrado: {REGISTERED_NAME}")
        print(f"  URI: {model_info.model_uri}")

    print("\n[4/4] Guardando artefactos locales...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib_path = os.path.join(MODEL_DIR, "xgboost_v1.joblib")
    joblib.dump(model, joblib_path)
    print(f"  {joblib_path}")

    cols_path = os.path.join(MODEL_DIR, "feature_cols.json")
    with open(cols_path, "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    print(f"  {cols_path}")

    print("\nDone.")
    print(f"\nVer en MLflow UI:")
    print(f"  .venv\\Scripts\\mlflow ui --backend-store-uri {MLFLOW_DB} --port 5000")
    print(f"  -> Models tab -> {REGISTERED_NAME}")


if __name__ == "__main__":
    main()
