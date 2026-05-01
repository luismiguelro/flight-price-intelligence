"""
Carga economy.csv + business.csv a Supabase tabla raw_flights.
Usa bulk insert (execute_values) — 300k filas en ~85 segundos.
"""
import os
import time
import psycopg2
import psycopg2.extras
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def get_db_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL no definido en .env")
    return url


def get_engine():
    url = get_db_url().replace("postgresql://", "postgresql+psycopg2://")
    url += ("&" if "?" in url else "?") + "sslmode=require"
    return create_engine(url)


def load_csv() -> pd.DataFrame:
    eco = pd.read_csv(os.path.join(RAW_DIR, "economy.csv"), encoding="utf-8-sig")
    bus = pd.read_csv(os.path.join(RAW_DIR, "business.csv"), encoding="utf-8-sig")
    eco["class"] = "Economy"
    bus["class"] = "Business"
    df = pd.concat([eco, bus], ignore_index=True)

    # Limpiar precio
    df["price"] = df["price"].astype(str).str.replace(",", "", regex=False).astype(float)

    # Limpiar stop (extraer solo el tipo)
    df["stop"] = df["stop"].astype(str).str.extract(r"(non-stop|1-stop|2\+-stop)", expand=False)

    # Renombrar columnas a snake_case
    df = df.rename(columns={
        "ch_code":    "airline_code",
        "num_code":   "flight_number",
        "dep_time":   "departure_time",
        "from":       "source_city",
        "time_taken": "duration_raw",
        "arr_time":   "arrival_time",
        "to":         "destination_city",
    })

    print(f"  Filas cargadas: {len(df):,}  |  Columnas: {list(df.columns)}")
    return df


def create_table(engine):
    ddl = """
    CREATE TABLE IF NOT EXISTS raw_flights (
        id               SERIAL PRIMARY KEY,
        date             DATE,
        airline          TEXT,
        airline_code     TEXT,
        flight_number    INTEGER,
        departure_time   TEXT,
        source_city      TEXT,
        duration_raw     TEXT,
        stop             TEXT,
        arrival_time     TEXT,
        destination_city TEXT,
        class            TEXT,
        price            NUMERIC,
        ingested_at      TIMESTAMPTZ DEFAULT NOW()
    );
    """
    with engine.connect() as conn:
        conn.execute(text(ddl))
        conn.execute(text("TRUNCATE TABLE raw_flights RESTART IDENTITY;"))
        conn.commit()
    print("  Tabla raw_flights lista (truncada)")


def insert_bulk(engine, df: pd.DataFrame):
    cols = [
        "date", "airline", "airline_code", "flight_number",
        "departure_time", "source_city", "duration_raw", "stop",
        "arrival_time", "destination_city", "class", "price",
    ]
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y").dt.date
    rows = df[cols].to_records(index=False).tolist()

    sql = f"INSERT INTO raw_flights ({', '.join(cols)}) VALUES %s"

    db_url = get_db_url() + ("&" if "?" in get_db_url() else "?") + "sslmode=require"
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cursor:
            psycopg2.extras.execute_values(cursor, sql, rows, page_size=5000)

    print(f"  {len(rows):,} filas insertadas")


def verify(engine):
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM raw_flights")).scalar()
        sample = conn.execute(text(
            "SELECT airline, source_city, destination_city, class, price "
            "FROM raw_flights LIMIT 3"
        )).fetchall()
    print(f"  Verificacion: {count:,} filas en raw_flights")
    for row in sample:
        print(f"    {row}")


if __name__ == "__main__":
    print("=== Flight Price Ingest ===")

    print("\n[1/4] Leyendo CSVs...")
    df = load_csv()

    print("\n[2/4] Conectando a Supabase...")
    engine = get_engine()
    get_db_url()  # valida que DATABASE_URL existe antes de continuar

    print("\n[3/4] Creando tabla e insertando...")
    create_table(engine)
    t0 = time.time()
    insert_bulk(engine, df)
    print(f"  Tiempo: {time.time() - t0:.1f}s")

    print("\n[4/4] Verificando...")
    verify(engine)

    print("\nDone.")
