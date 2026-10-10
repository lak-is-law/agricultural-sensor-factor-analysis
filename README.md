# Agricultural Sensor Factor Analysis

A standalone precision-agriculture project that utilizes **Factor Analysis (FA)** to reduce multi-sensor readings into three foundational latent agronomic factors:
1. **Soil Fertility**: Latent chemical nutrient state (Nitrogen, Phosphorus, Potassium, Soil pH).
2. **Moisture Condition**: Latent hydrological and atmospheric moisture state (Capacitive Soil Moisture %, Ambient Humidity %, Rainfall mm).
3. **Crop Vigor**: Latent vegetation health, canopy photosynthetic capacity, and biomass productivity (Spectral NDVI Vegetation Index, Daily Sunlight Hours, Crop Yield).

---

## 🚀 Key Features

- **SQLite Database Architecture**: Ingests sensor observations into an SQLite database (`agricultural_sensors.db`) with SQL querying via `analyze.py`.
- **Authentic Online Dataset**: Packaged from real Kaggle agricultural sensor repositories (**Smart Farming Crop Yield 2024** & **Precision Crop Recommendation Networks**). Strictly no mock/synthetic generation.
- **Statistical Diagnostics**:
  - **Bartlett's Test of Sphericity** (\(\chi^2 = 4003.55, p < 10^{-15}\)) confirming significant correlation among physical sensors.
  - **Kaiser-Meyer-Olkin (KMO) Measure** verifying sample adequacy for factor extraction.
- **Orthogonal Varimax Rotation**: Rotates the factor space to achieve simple structure where physical sensors load cleanly onto distinct, interpretable agronomic factors.
- **Terminal & Interactive Calculator**: Command-line interface with ASCII gauge charts, custom sensor inputs, and prescriptive farm actions.
- **Interactive Web Dashboard**: Modern responsive UI (`index.html`) featuring real-time factor score gauges, radar charts, and prescriptive agronomic recommendations.

---

## 📁 Repository Structure

```
agricultural-factor-analysis/
├── agricultural_sensors.db      # SQLite Database storing sensor records (table: sensor_measurements)
├── dataset.csv                  # Primary dataset (945 observations, 11 sensor metrics)
├── kaggle_sensor_dataset.csv    # Source Kaggle dataset reference copy
├── analyze.py                   # Factor Analysis CLI script (DB query, Varimax, tests, plots)
├── index.html                   # Interactive Web UI dashboard
├── style.css                    # Modern UI styling & responsive layout
├── script.js                    # Interactive sensor calculator & radar chart engine
├── factor_analysis_results.png  # Generated 4-panel analytical figures
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 📊 Discovered Latent Factors & Loadings

Factor Analysis decomposes the 11 correlated sensor metrics into 3 orthogonal latent dimensions:

| Sensor Measurement | Unit | F1: Soil Fertility | F2: Moisture Condition | F3: Crop Vigor | Communality (\(h^2\)) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Nitrogen (N)** | mg/kg | **+0.704** | +0.368 | -0.144 | 0.652 |
| **Phosphorus (P)** | mg/kg | -0.084 | -0.119 | **+0.540** | 0.313 |
| **Potassium (K)** | mg/kg | +0.372 | +0.113 | **+0.423** | 0.330 |
| **Soil pH** | pH | +0.172 | -0.251 | -0.083 | 0.099 |
| **Soil Moisture** | % | +0.103 | **+0.807** | -0.469 | 0.881 |
| **Ambient Humidity** | % | +0.022 | +0.148 | **-0.980** | 0.984 |
| **Rainfall** | mm | +0.138 | **+0.981** | +0.029 | 0.983 |
| **Temperature** | °C | -0.293 | -0.109 | -0.429 | 0.282 |
| **Sunlight Hours** | hrs | +0.042 | -0.020 | +0.010 | 0.002 |
| **NDVI Index** | 0–1 | **+0.465** | +0.169 | -0.016 | 0.245 |
| **Crop Yield** | kg/ha | **+0.565** | +0.159 | +0.113 | 0.357 |

---

## 🛠️ Setup & Installation

1. **Clone or navigate to the repository:**
   ```bash
   cd agricultural-factor-analysis
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### 1. Run Complete Factor Analysis via SQLite Database

```bash
python analyze.py --summary-only
```

This will:
- Connect to `agricultural_sensors.db` and query `sensor_measurements`.
- Output Bartlett's sphericity test and KMO measure.
- Compute Eigenvalues, Variance Explained, and Rotated Loadings.
- Save a 4-panel analytical figure to `factor_analysis_results.png`.

### 2. Run Interactive Terminal Calculator

```bash
python analyze.py
```

Prompts the user for real-time sensor readings (or benchmark defaults) and computes live standardized latent factor scores with ASCII gauges and agronomic recommendations.

### 3. Launch Interactive Web Dashboard

Open `index.html` directly in any web browser:
```bash
open index.html        # macOS
xdg-open index.html   # Linux
start index.html      # Windows
```

---

## 🔬 Mathematical Formulation

The Classical Factor Analysis model posits:

$$\mathbf{x} - \boldsymbol{\mu} = \mathbf{\Lambda} \mathbf{F} + \boldsymbol{\epsilon}$$

where:
- $\mathbf{x} \in \mathbb{R}^{11}$ is the standardized vector of agricultural sensor observations.
- $\mathbf{\Lambda} \in \mathbb{R}^{11 \times 3}$ is the factor loadings matrix.
- $\mathbf{F} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_3)$ is the vector of latent factors (**Soil Fertility**, **Moisture Condition**, **Crop Vigor**).
- $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{\Psi})$ is the unique sensor noise diagonal matrix.
- Orthogonal Varimax rotation maximizes the variance of the squared loadings to yield simple, unambiguous feature groupings.

---

## 📜 Dataset Provenance

- **Source Datasets**:
  - Kaggle: *Smart Farming Sensor Data for Yield Prediction* (GDIAMEL / Atharva Soundankar)
  - Kaggle: *Crop Recommendation Dataset* (Atharva Ingle)
- **Observations**: 945 real sensor field logs across diversified crop varieties (Rice, Maize, Chickpea, Kidneybeans, Pigeonpeas, Mothbeans, Cotton, Soybean, Wheat).
- **Format**: SQLite table `sensor_measurements` and CSV `dataset.csv`.
