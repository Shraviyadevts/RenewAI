# RenewAI – Production Renewable Energy Operations & Decision Support Platform

RenewAI is an operational solar power forecasting, dynamic energy balance, battery storage simulation, and prescriptive decision support platform. It integrates **live Open-Meteo weather and geocoding** with an **XGBoost solar forecaster**, providing continuous rolling 24-hour predictions, dynamic demand curves, physical battery SOC trajectory simulations, energy risk mitigation, and tailored advisories for 5 distinct facility operational roles.

---

## 1. End-to-End System Architecture

```
                [Dynamic Location: Auto-Detect / City Search / Preset Hubs]
                                            ↓
               [Live Open-Meteo API: Temperature, Pressure, Precip, GHI, DNI]
                                            ↓
                [Feature Engineering: Ideal Gas Law Rhoa, Temporal Features]
                                            ↓
               [Trained XGBoost Regressor: Rolling 24-Hour Solar Forecast]
                                            ↓
               ┌────────────────────────────┼──────────────────────────┐
               ▼                            ▼                          ▼
     [Dynamic Demand Engine]      [Battery SOC Simulation]     [Energy Balance Engine]
     (Diurnal Feeder/Industrial/  (Hour-by-hour storage flow,  (Surplus/Deficit, streaks,
      Commercial/EV Curves)        reserve breach detection)    5 Balance Tiers)
               └────────────────────────────┬──────────────────────────┘
                                            ↓
                   [Multi-Role Prescriptive Decision Support Engine]
    ┌─────────────────┬───────────────────┬──────────────────┬─────────────────┬──────────────────┐
    ▼                 ▼                   ▼                  ▼                 ▼
 ⚡ Grid Operator  🔋 Battery Storage  🏭 Industrial Facility ☀️ Solar Plant  🚗 EV Charging
    └─────────────────┴───────────────────┴──────────────────┴─────────────────┴──────────────────┘
                                            ↓
             [Streamlit Web Operations Center & FastAPI RESTful Backend]
```

---

## 2. Key Production Capabilities

### 📍 Live Dynamic Geolocation & Weather
- **Browser/IP Auto-Detection**: Automatically identifies the user's city and geographic coordinates.
- **Worldwide City Search**: Instant geocoding lookup for any global city via Open-Meteo Geocoding API.
- **Live Weather Telemetry**: Temperature, Pressure, Precipitation, Global Horizontal Irradiance (GHI), and Direct Normal Irradiance (DNI).
- **Exact 24-Hour Horizon**: Exactly 24 future hourly forecast points rolling continuously forward with the local clock.

### ⚡ Dynamic Diurnal Demand Curves
- Replaces fixed load assumptions with 5 selectable operational demand profiles:
  1. `⚡ Grid Feeder Curve` (Dual morning & evening distribution peaks)
  2. `🏭 Industrial Shift Curve` (Heavy daytime manufacturing with shift ramps)
  3. `🏢 Commercial Building Curve` (Occupancy & HVAC profile)
  4. `🚗 EV Charging Hub Curve` (Morning arrival and evening return spikes)
  5. `📊 Constant Baseline Load` (User-configurable flat baseline)

### 🔋 24-Hour Physical Battery SOC Simulation
- Hour-by-hour simulation of battery charging and discharging:
  - Tracks energy stored ($\text{Wh}$), energy discharged ($\text{Wh}$), and projected minimum reserve (SOC %).
  - Constrained by capacity ($\text{Wh}$), max charge/discharge rates ($\text{W}$), efficiency losses, and $20\%$ minimum reserve protection.
  - Generates a **projected 24-hour SOC curve** displayed in the forecast chart and timeline table.

### 👤 5 Targeted Facility Operating Roles
1. ⚡ **Grid Operator**:
   - `EXPORT SURPLUS TO GRID`
   - `IMPORT FROM GRID & REDUCE GRID STRESS`
   - `DISPATCH BATTERY`
   - `CURTAIL EXCESS GENERATION`
   - `MAINTAIN GRID BALANCE`
2. 🔋 **Battery Storage Operator**:
   - `CHARGE BATTERY`
   - `DISCHARGE BATTERY`
   - `HOLD BATTERY RESERVE & PREPARE FOR DEFICIT`
   - `OPTIMIZE CHARGE/DISCHARGE & HOLD`
3. 🏭 **Industrial Operator**:
   - `RUN FLEXIBLE INDUSTRIAL LOADS`
   - `SHIFT FLEXIBLE LOAD & USE STORED ENERGY`
   - `USE RENEWABLE POWER & CHARGE STORAGE`
   - `REDUCE NON-CRITICAL LOAD & IMPORT FROM GRID`
4. ☀️ **Solar Plant Operator**:
   - `EXPORT SURPLUS`
   - `MAXIMIZE SELF-CONSUMPTION & STORE ENERGY`
   - `MAINTAIN GENERATION USING BATTERY`
   - `PREPARE FOR WEATHER REDUCTION`
   - `CURTAIL EXCESS / STANDBY`
5. 🚗 **EV Charging Station Operator**:
   - `START EV CHARGING & CHARGE DURING SURPLUS`
   - `USE RENEWABLE POWER FOR EV CHARGING`
   - `USE STORED ENERGY FOR CHARGING`
   - `REDUCE CHARGING RATE / DELAY EV CHARGING`

### ⚠️ Energy Risk Assessment & Overnight Opportunity Engine
- **Risk Assessment**: Dynamic 4-tier evaluation (`CRITICAL`, `HIGH`, `MODERATE`, `LOW`) integrating multi-hour deficit duration, battery reserve minima, and forecast uncertainty.
- **Opportunity Window**: Computes exact start and end times with day context across midnight, total clean surplus ($\text{Wh}$), total deficit ($\text{Wh}$), and longest continuous streaks.

---

## 3. How to Run the Application

### 1. Launch the Streamlit Live Operations Center:
```bash
python -m streamlit run frontend/app.py
```
👉 Open browser at: **`http://localhost:8501`**

### 2. Launch the FastAPI REST API:
```bash
uvicorn backend.main:app --reload --port 8000
```
👉 Interactive Swagger Documentation: **`http://localhost:8000/docs`**

### 3. Run the CLI Pipeline Runner:
```bash
python main.py --demand 60 --battery 50
```

---

## 4. API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | System overview, status, supported roles & demand profiles |
| `GET` | `/health` | Service health, model status & weather connectivity |
| `GET` | `/location` | Worldwide city search & IP auto-detection |
| `GET` | `/weather/current` | Current weather & model-derived solar estimate for dynamic location |
| `GET` | `/forecast` | Exact 24h rolling forecast with demand curve & battery SOC simulation |
| `GET` | `/opportunity` | Dynamic opportunity window detection for custom demand |
| `GET` | `/recommendation` | Prescriptive role-specific decision advisory |
| `POST` | `/predict` | Single-step solar prediction with reliability score |
| `POST` | `/recommendation` | Prescriptive decision recommendation with 'Why?' deduction |
| `POST` | `/what-if` | Dynamic live What-If scenario simulation |
