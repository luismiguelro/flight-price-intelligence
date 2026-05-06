"""
Experimento baseline con MLflow.
Estrategia: predice el precio usando la media por ruta (sin ML).
Sirve como punto de referencia para comparar contra los modelos reales.

Salida: experimento 'flight-price-intelligence' visible en MLflow UI
Ejecutar UI: mlflow ui --port 5000
"""
import os

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

PROC_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MLRUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "mlruns")

EXPERIMENT_NAME = "flight-price-intelligence"
MLFLOW_DB       = f"sqlite:///{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mlflow.db'))}"
RANDOM_STATE    = 42
TEST_SIZE       = 0.2


def load_features() -> pd.DataFrame:
    path = os.path.join(PROC_DIR, "features.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError("Ejecuta primero 02_feature_engineering.py")
    return pd.read_parquet(path)


def mean_by_route_baseline(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    """Predice el precio como la media historica de la misma ruta."""
    route_means = y_train.groupby(X_train["route_enc"]).mean()
    global_mean = y_train.mean()
    preds = X_test["route_enc"].map(route_means).fillna(global_mean)
    return preds.to_numpy()


def main():
    print("=== Baseline MLflow ===")

    # Configurar tracking con SQLite (recomendado MLflow 3.x)
    mlflow.set_tracking_uri(MLFLOW_DB)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\n[1/3] Cargando features...")
    df = load_features()
    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    print("\n[2/3] Corriendo experimento baseline...")
    with mlflow.start_run(run_name="baseline_mean_by_route"):
        # Parametros
        mlflow.log_param("model_type",   "baseline_mean_by_route")
        mlflow.log_param("test_size",    TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_train",      len(X_train))
        mlflow.log_param("n_test",       len(X_test))

        # Prediccion
        y_pred = mean_by_route_baseline(X_train, y_train, X_test)

        # Metricas
        mae    = mean_absolute_error(y_test, y_pred)
        r2     = r2_score(y_test, y_pred)
        mape   = np.mean(np.abs((y_test.to_numpy() - y_pred) / y_test.to_numpy())) * 100

        mlflow.log_metric("mae",  round(mae, 2))
        mlflow.log_metric("r2",   round(r2,  4))
        mlflow.log_metric("mape", round(mape, 2))

        run_id = mlflow.active_run().info.run_id
        print(f"  MAE  : {mae:,.0f} INR")
        print(f"  R2   : {r2:.4f}")
        print(f"  MAPE : {mape:.1f}%")
        print(f"  Run ID: {run_id}")

    print(f"\n[3/3] Listo.")
    print(f"  Ver resultados: mlflow ui --backend-store-uri {MLFLOW_DB} --port 5000")


if __name__ == "__main__":
    main()
