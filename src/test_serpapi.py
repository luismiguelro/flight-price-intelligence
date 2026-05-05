"""
Prueba la conexion con SerpAPI Google Flights.
Consulta vuelos Delhi (DEL) -> Mumbai (BOM) para una fecha futura
y devuelve los precios reales en JSON.

Uso:
    python src/test_serpapi.py
    python src/test_serpapi.py --origin DEL --dest BOM --date 2026-06-01
"""
import argparse
import json
import os
from datetime import date, timedelta

from dotenv import load_dotenv
from serpapi import GoogleSearch

load_dotenv()

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Ciudades del dataset -> codigos IATA
CITY_TO_IATA = {
    "Delhi":     "DEL",
    "Mumbai":    "BOM",
    "Bangalore": "BLR",
    "Kolkata":   "CCU",
    "Hyderabad": "HYD",
    "Chennai":   "MAA",
}


def fetch_flights(origin: str, dest: str, outbound_date: str) -> dict:
    """
    Consulta SerpAPI Google Flights y retorna el resultado crudo.
    origin / dest: codigos IATA (ej. 'DEL', 'BOM')
    outbound_date: 'YYYY-MM-DD'
    """
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        raise ValueError("SERPAPI_KEY no definido en .env")

    params = {
        "engine":         "google_flights",
        "departure_id":   origin,
        "arrival_id":     dest,
        "outbound_date":  outbound_date,
        "currency":       "INR",
        "hl":             "en",
        "type":           "2",      # solo ida
        "api_key":        api_key,
    }

    search = GoogleSearch(params)
    return search.get_dict()


def extract_summary(result: dict) -> list[dict]:
    """
    Extrae los campos relevantes de best_flights + other_flights.
    Retorna lista de opciones ordenadas por precio.
    """
    flights = []
    for group in ("best_flights", "other_flights"):
        for option in result.get(group, []):
            leg = option["flights"][0]
            flights.append({
                "airline":          leg.get("airline"),
                "flight_number":    leg.get("flight_number"),
                "departure_time":   leg.get("departure_airport", {}).get("time"),
                "arrival_time":     leg.get("arrival_airport", {}).get("time"),
                "duration_minutes": option.get("total_duration"),
                "stops":            len(option.get("flights", [])) - 1,
                "price_inr":        option.get("price"),
                "class":            leg.get("travel_class", "Economy"),
            })

    return sorted(flights, key=lambda x: x["price_inr"] or 9_999_999)


def main(origin: str, dest: str, outbound_date: str):
    print(f"\n=== SerpAPI Google Flights ===")
    print(f"Ruta: {origin} -> {dest}  |  Fecha: {outbound_date}\n")

    result = fetch_flights(origin, dest, outbound_date)

    # Guardar respuesta cruda
    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = os.path.join(RAW_DIR, "serpapi_test.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Respuesta cruda guardada en: {out_path}")

    # Mostrar resumen
    summary = extract_summary(result)
    if not summary:
        print("Sin resultados. Revisa la fecha o los codigos IATA.")
        return

    print(f"\nOpciones encontradas: {len(summary)}\n")
    print(f"{'Aerolinea':<20} {'Salida':<8} {'Llegada':<8} {'Dur(min)':<10} {'Escalas':<8} {'Precio INR'}")
    print("-" * 70)
    for f in summary[:10]:
        print(
            f"{str(f['airline']):<20} "
            f"{str(f['departure_time']):<8} "
            f"{str(f['arrival_time']):<8} "
            f"{str(f['duration_minutes']):<10} "
            f"{str(f['stops']):<8} "
            f"{f['price_inr']:,}"
        )

    cheapest = summary[0]
    print(f"\nMas barato: {cheapest['airline']} a {cheapest['price_inr']:,} INR")
    print(f"Resumen JSON:\n{json.dumps(cheapest, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    default_date = (date.today() + timedelta(days=15)).strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", default="DEL", help="Codigo IATA origen (default: DEL)")
    parser.add_argument("--dest",   default="BOM", help="Codigo IATA destino (default: BOM)")
    parser.add_argument("--date",   default=default_date, help="Fecha YYYY-MM-DD (default: hoy+15d)")
    args = parser.parse_args()

    main(args.origin, args.dest, args.date)
