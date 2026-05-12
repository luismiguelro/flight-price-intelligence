"""
Flight Price Intelligence — Streamlit App
Consulta precio real (SerpAPI) + prediccion del modelo (FastAPI)
y emite señal Buy Now / Wait.
"""
import os
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from serpapi import GoogleSearch

load_dotenv()

# Streamlit Cloud expone secrets via st.secrets, no como env vars.
# Este bloque los mapea a os.environ para que os.getenv() funcione igual en local y en cloud.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
API_URL = os.getenv("API_URL", "https://flight-price-intelligence.onrender.com")

CITIES = ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"]

CITY_TO_IATA = {
    "Delhi": "DEL", "Mumbai": "BOM", "Bangalore": "BLR",
    "Kolkata": "CCU", "Hyderabad": "HYD", "Chennai": "MAA",
}

AIRLINES = ["IndiGo", "Air India", "Air India Express", "Vistara",
            "SpiceJet", "AirAsia", "GO FIRST", "StarAir", "Trujet"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_serpapi_price(origin_iata: str, dest_iata: str, flight_date: str) -> list[dict]:
    """Consulta Google Flights via SerpAPI y retorna lista de opciones."""
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        st.error("SERPAPI_KEY no configurada.")
        return []

    params = {
        "engine":        "google_flights",
        "departure_id":  origin_iata,
        "arrival_id":    dest_iata,
        "outbound_date": flight_date,
        "currency":      "INR",
        "hl":            "en",
        "type":          "2",
        "api_key":       api_key,
    }
    result = GoogleSearch(params).get_dict()

    options = []
    for group in ("best_flights", "other_flights"):
        is_best = group == "best_flights"
        for opt in result.get(group, []):
            leg_first = opt["flights"][0]
            leg_last  = opt["flights"][-1]
            options.append({
                "airline":          leg_first.get("airline", ""),
                "airline_logo":     leg_first.get("airline_logo", ""),
                "flight_number":    leg_first.get("flight_number", ""),
                "departure_time":   leg_first.get("departure_airport", {}).get("time", "")[-5:],
                "arrival_time":     leg_last.get("arrival_airport",  {}).get("time", "")[-5:],
                "duration_minutes": opt.get("total_duration", 120),
                "stops":            len(opt.get("flights", [])) - 1,
                "price_inr":        opt.get("price", 0),
                "is_best":          is_best,
            })
    return sorted(options, key=lambda x: x["price_inr"])


def stops_label(n: int) -> str:
    return {0: "non-stop", 1: "1-stop"}.get(n, "2+-stop")


def call_predict_api(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error llamando a la API: {e}")
        return None


def fetch_price_curve(
    origin: str, destination: str, airline: str,
    flight_class: str, stops: str, duration_minutes: int, departure_time: str,
) -> list[dict]:
    try:
        r = requests.get(
            f"{API_URL}/price_curve",
            params={
                "origin":            origin,
                "destination":       destination,
                "departure_time":    departure_time or "08:00",
                "airline":           airline,
                "flight_class":      flight_class,
                "stops":             stops,
                "duration_minutes":  duration_minutes,
            },
            timeout=15,
        )
        r.raise_for_status()
        return r.json().get("curve", [])
    except Exception as e:
        st.warning(f"No se pudo cargar la curva de precios: {e}")
        return []


def price_curve_chart(curve: list[dict], current_price: float) -> go.Figure:
    days   = [p["days_until_flight"] for p in curve]
    prices = [p["predicted_price"]   for p in curve]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=days, y=prices,
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.08)",
        line=dict(color="#3b82f6", width=2.5),
        mode="lines+markers",
        marker=dict(size=6, color="#3b82f6"),
        name="Precio predicho",
        hovertemplate="<b>%{x} días antes</b><br>₹%{y:,.0f}<extra></extra>",
    ))

    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="#ef4444",
        line_width=1.5,
        annotation_text=f"Precio actual  ₹{current_price:,.0f}",
        annotation_position="top right",
        annotation_font_color="#ef4444",
    )

    fig.update_layout(
        title=dict(text="Precio predicho según días de anticipación de compra", font_size=14),
        xaxis=dict(
            title="Días de anticipación",
            autorange="reversed",
            tickvals=days,
        ),
        yaxis=dict(title="Precio (INR)", tickformat=",.0f"),
        height=300,
        margin=dict(t=50, b=40, l=70, r=20),
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def signal_gauge(current: float, predicted: float) -> go.Figure:
    diff_pct = (current - predicted) / predicted * 100
    color = "#ef4444" if diff_pct > 5 else "#22c55e"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current,
        delta={"reference": predicted, "valueformat": ",.0f",
               "prefix": "vs predicho: ₹"},
        number={"prefix": "₹", "valueformat": ",.0f"},
        title={"text": "Precio actual vs predicho (INR)"},
        gauge={
            "axis":  {"range": [0, max(current, predicted) * 1.4]},
            "bar":   {"color": color},
            "steps": [
                {"range": [0, predicted * 0.95],          "color": "#dcfce7"},
                {"range": [predicted * 0.95, predicted * 1.05], "color": "#fef9c3"},
                {"range": [predicted * 1.05, max(current, predicted) * 1.4], "color": "#fee2e2"},
            ],
            "threshold": {
                "line":  {"color": "#6b7280", "width": 3},
                "thickness": 0.75,
                "value": predicted,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
    return fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Flight Price Intelligence",
    page_icon="✈️",
    layout="centered",
)

st.title("✈️ Flight Price Intelligence")
st.caption("¿Es buen momento para comprar tu vuelo? Consulta el precio actual y decide con datos.")

st.divider()

# Selectores de ciudad fuera del form para que el destino se filtre dinamicamente
col1, col2 = st.columns(2)
with col1:
    origin = st.selectbox("Ciudad origen", CITIES, index=0)
with col2:
    dest_options = [c for c in CITIES if c != origin]
    destination = st.selectbox("Ciudad destino", dest_options, index=0)

if origin == destination:
    st.error("⚠️ El origen y el destino no pueden ser la misma ciudad.")
    st.stop()

# Resto del formulario
with st.form("flight_form"):
    col3, col4 = st.columns(2)
    with col3:
        flight_date = st.date_input(
            "Fecha del vuelo",
            value=date.today() + timedelta(days=15),
            min_value=date.today() + timedelta(days=1),
        )
    with col4:
        flight_class = st.selectbox("Clase", ["Economy", "Business"])

    submitted = st.form_submit_button("🔍 Consultar precios", use_container_width=True)

# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------
if submitted:
    if origin == destination:
        st.warning("Origen y destino deben ser distintos.")
        st.stop()

    with st.spinner("Consultando Google Flights..."):
        options = fetch_serpapi_price(
            CITY_TO_IATA[origin],
            CITY_TO_IATA[destination],
            flight_date.strftime("%Y-%m-%d"),
        )

    if not options:
        st.warning("No se encontraron vuelos para esa ruta y fecha.")
        st.stop()

    # Filtrar por clase (SerpAPI no filtra por clase en el free tier; usamos el más barato)
    cheapest = options[0]

    st.subheader(f"✈️ Vuelos encontrados: {origin} → {destination}")
    st.caption(f"{len(options)} opciones · ordenadas por precio · análisis basado en el más económico")

    STOPS_COLOR = {0: "#16a34a", 1: "#ca8a04", 2: "#dc2626"}
    STOPS_TEXT  = {0: "Non-stop", 1: "1 escala", 2: "2+ escalas"}

    cheapest_price = options[0]["price_inr"]

    for opt in options[:8]:
        dur_h   = opt["duration_minutes"] // 60
        dur_m   = opt["duration_minutes"] % 60
        stops_c = STOPS_COLOR.get(opt["stops"], "#dc2626")
        stops_t = STOPS_TEXT.get(opt["stops"], "2+ escalas")
        is_cheapest = opt["price_inr"] == cheapest_price
        badge = "<br><span style='background:rgba(34,197,94,0.15);color:#22c55e;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600'>Más económico</span>" if is_cheapest else ""
        logo_html = f"<img src='{opt['airline_logo']}' height='22' style='vertical-align:middle;margin-right:6px'>" if opt["airline_logo"] else ""

        st.markdown(f"""
<div style='
  border:1px solid rgba(128,128,128,0.2);
  border-radius:12px;
  padding:14px 18px;
  margin-bottom:10px;
  background:var(--secondary-background-color);
'>
  <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>
    <div style='display:flex;align-items:center;gap:10px;min-width:160px'>
      {logo_html}
      <div>
        <div style='font-weight:600;font-size:14px;color:var(--text-color)'>{opt['airline']}</div>
        <div style='opacity:0.5;font-size:12px;color:var(--text-color)'>{opt['flight_number']}</div>
      </div>
    </div>
    <div style='text-align:center'>
      <div style='font-size:20px;font-weight:700;color:var(--text-color)'>
        {opt['departure_time']}
        <span style='opacity:0.4;font-weight:400'> → </span>
        {opt['arrival_time']}
      </div>
      <div style='opacity:0.55;font-size:12px;color:var(--text-color)'>
        {dur_h}h {dur_m}m &nbsp;·&nbsp;
        <span style='color:{stops_c};font-weight:600;opacity:1'>{stops_t}</span>
      </div>
    </div>
    <div style='text-align:right'>
      <div style='font-size:20px;font-weight:700;color:#3b82f6'>₹{opt['price_inr']:,}{badge}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # Llamar a la API de predicción con el vuelo más barato
    with st.spinner("Analizando con el modelo..."):
        payload = {
            "origin":            origin,
            "destination":       destination,
            "flight_date":       flight_date.strftime("%Y-%m-%d"),
            "departure_time":    cheapest["departure_time"] or "08:00",
            "airline":           cheapest["airline"],
            "flight_class":      flight_class,
            "stops":             stops_label(cheapest["stops"]),
            "duration_minutes":  cheapest["duration_minutes"],
            "current_price":     cheapest["price_inr"],
        }
        result = call_predict_api(payload)

    if result:
        signal   = result["signal"]
        is_buy   = signal == "BUY"
        color    = "green" if is_buy else "red"
        emoji    = "✅" if is_buy else "⏳"
        label    = result["signal_es"]
        conf_pct = int(result["confidence"] * 100)

        # Señal principal
        st.markdown(
            f"<h2 style='text-align:center; color:{color}'>{emoji} {label}</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='text-align:center; color:gray'>Confianza: {conf_pct}%</p>",
            unsafe_allow_html=True,
        )

        # Gauge
        st.plotly_chart(
            signal_gauge(result["current_price"], result["predicted_price"]),
            use_container_width=True,
        )

        # Métricas
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Precio actual",    f"₹{result['current_price']:,.0f}")
        col_b.metric("Precio predicho",  f"₹{result['predicted_price']:,.0f}")
        col_c.metric("Diferencia",       f"{result['price_diff_pct']:+.1f}%")

        # Explicación
        st.info(result["explanation"])

        # Line chart: curva de precio por anticipación
        with st.spinner("Cargando curva de precios..."):
            curve = fetch_price_curve(
                origin, destination,
                cheapest["airline"],
                flight_class,
                stops_label(cheapest["stops"]),
                cheapest["duration_minutes"],
                cheapest["departure_time"] or "08:00",
            )

        if curve:
            st.subheader("📈 ¿Cuándo conviene comprar en esta ruta?")
            st.plotly_chart(price_curve_chart(curve, result["current_price"]), use_container_width=True)
            st.caption(
                "Predicciones del modelo XGBoost según días de anticipación · "
                "Entrenado con vuelos domésticos India Feb–Mar 2022 · "
                "La línea roja es el precio actual de Google Flights"
            )

st.divider()
st.caption("Modelo entrenado con datos de vuelos domésticos India (Feb–Mar 2022) · "
           "Precios en INR · Solo rutas entre Delhi, Mumbai, Bangalore, Kolkata, Hyderabad y Chennai")
