# Rutas soportadas por el modelo

El modelo fue entrenado con datos de **vuelos domésticos de India** (Feb–Mar 2022, Kaggle).
Solo puede predecir precios con confianza para las **30 rutas entre estas 6 ciudades**.

## Ciudades disponibles

| Ciudad     | Código IATA | Usar en la app |
|------------|-------------|----------------|
| Delhi      | DEL         | ✅              |
| Mumbai     | BOM         | ✅              |
| Bangalore  | BLR         | ✅              |
| Kolkata    | CCU         | ✅              |
| Hyderabad  | HYD         | ✅              |
| Chennai    | MAA         | ✅              |

Rutas internacionales o con otras ciudades indias → **no soportadas**.

## Rutas con mayor volumen de datos (mayor confianza)

| Ruta                    | Vuelos en dataset | Precio min (INR) | Precio medio (INR) | Precio max (INR) |
|-------------------------|:-----------------:|:----------------:|:------------------:|:----------------:|
| Delhi → Mumbai          | 15,291            | 2,281            | 19,354             | 95,657           |
| Mumbai → Delhi          | 14,809            | 2,336            | 18,725             | 111,437          |
| Delhi → Bangalore       | 14,012            | 3,090            | 17,880             | 85,353           |
| Bangalore → Delhi       | 13,756            | 2,723            | 17,723             | 111,883          |
| Bangalore → Mumbai      | 12,940            | 2,150            | 23,127             | 103,819          |
| Mumbai → Bangalore      | 12,885            | 2,074            | 23,148             | 114,523          |
| Mumbai → Kolkata        | 12,602            | 2,835            | 22,379             | 100,909          |
| Delhi → Kolkata         | 11,934            | 2,480            | 20,566             | 117,307          |
| Kolkata → Mumbai        | 11,467            | 3,379            | 22,079             | 110,936          |
| Delhi → Chennai         | 10,780            | 1,998            | 19,370             | 104,466          |

## Rutas con menor volumen (predicción menos confiable)

| Ruta                        | Vuelos en dataset |
|-----------------------------|:-----------------:|
| Chennai → Hyderabad         | 6,103             |
| Hyderabad → Chennai         | 6,395             |
| Bangalore → Chennai         | 6,410             |
| Chennai → Bangalore         | 6,493             |
| Kolkata → Chennai           | 6,653             |
| Chennai → Kolkata           | 6,983             |

> Rutas con menos de 7,000 registros producen predicciones con mayor margen de error.
> Para el demo de Streamlit, prioriza las rutas del primer bloque.

## Aerolíneas en el dataset

| Aerolínea   | Clase disponible  |
|-------------|-------------------|
| IndiGo      | Economy           |
| Air India   | Economy, Business |
| Vistara     | Economy, Business |
| SpiceJet    | Economy           |
| GO FIRST    | Economy           |
| AirAsia     | Economy           |
| StarAir     | Economy           |
| Trujet      | Economy           |

> Business class solo disponible en Air India y Vistara.

## Limitaciones del modelo

- **Temporalidad:** datos de Feb–Mar 2022. Los precios absolutos en INR pueden no reflejar tarifas actuales.
- **Sin booking date real:** `days_until_flight` se aproximó como días desde el inicio del dataset, no desde la fecha de compra real.
- **Señal "Compra/Espera":** compara el precio actual de SerpAPI (INR de hoy) contra la predicción del modelo (INR 2022). La señal es relativa dentro del rango histórico de la ruta, no un pronóstico absoluto.
