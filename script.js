/**
 * AgriFactor AI - Interactive Factor Analysis Simulator & Data Inspector
 * Calibrated against Kaggle Agricultural Sensor Data
 */

// Dataset Means and Standard Deviations for Feature Standardization
const SENSOR_STATS = {
  nitrogen: { mean: 50.55, std: 36.92, unit: "mg/kg" },
  phosphorus: { mean: 53.36, std: 32.98, unit: "mg/kg" },
  potassium: { mean: 48.15, std: 50.65, unit: "mg/kg" },
  soil_pH: { mean: 6.47, std: 0.77, unit: "pH" },
  soil_moisture_pct: { mean: 37.80, std: 13.90, unit: "%" },
  humidity_pct: { mean: 71.48, std: 22.26, unit: "%" },
  rainfall_mm: { mean: 103.46, std: 54.59, unit: "mm" },
  temperature_c: { mean: 25.62, std: 5.06, unit: "°C" },
  sunlight_hours: { mean: 6.64, std: 1.48, unit: "hrs" },
  ndvi_index: { mean: 0.63, std: 0.14, unit: "0-1" },
  crop_yield_kg_ha: { mean: 4125.0, std: 1150.0, unit: "kg/ha" }
};

// Rotated Factor Loadings Matrix (from Varimax Factor Analysis)
const FACTOR_LOADINGS = {
  nitrogen:          { f1: 0.704, f2: 0.368, f3: -0.144, h2: 0.652, psi: 0.348 },
  phosphorus:        { f1: -0.084, f2: -0.119, f3: 0.540, h2: 0.313, psi: 0.687 },
  potassium:         { f1: 0.372, f2: 0.113, f3: 0.423, h2: 0.330, psi: 0.670 },
  soil_pH:           { f1: 0.172, f2: -0.251, f3: -0.083, h2: 0.099, psi: 0.901 },
  soil_moisture_pct: { f1: 0.103, f2: 0.807, f3: -0.469, h2: 0.881, psi: 0.119 },
  humidity_pct:      { f1: 0.022, f2: 0.148, f3: -0.980, h2: 0.984, psi: 0.016 },
  rainfall_mm:       { f1: 0.138, f2: 0.981, f3: 0.029, h2: 0.983, psi: 0.017 },
  temperature_c:     { f1: -0.293, f2: -0.109, f3: -0.429, h2: 0.282, psi: 0.718 },
  sunlight_hours:    { f1: 0.042, f2: -0.020, f3: 0.010, h2: 0.002, psi: 0.998 },
  ndvi_index:        { f1: 0.465, f2: 0.169, f3: -0.016, h2: 0.245, psi: 0.755 },
  crop_yield_kg_ha:  { f1: 0.565, f2: 0.159, f3: 0.113, h2: 0.357, psi: 0.643 }
};

// Factor Scoring Coefficients (Standardized Regression Weights)
const FACTOR_WEIGHTS = {
  f1_fertility: {
    nitrogen: 0.42, phosphorus: 0.08, potassium: 0.24, soil_pH: 0.12,
    soil_moisture_pct: 0.04, humidity_pct: 0.01, rainfall_mm: 0.05,
    temperature_c: -0.18, sunlight_hours: 0.02, ndvi_index: 0.28, crop_yield_kg_ha: 0.34
  },
  f2_moisture: {
    nitrogen: 0.12, phosphorus: -0.05, potassium: 0.04, soil_pH: -0.16,
    soil_moisture_pct: 0.48, humidity_pct: 0.18, rainfall_mm: 0.54,
    temperature_c: -0.08, sunlight_hours: -0.02, ndvi_index: 0.09, crop_yield_kg_ha: 0.08
  },
  f3_vigor: {
    nitrogen: 0.15, phosphorus: 0.38, potassium: 0.32, soil_pH: -0.04,
    soil_moisture_pct: -0.14, humidity_pct: -0.52, rainfall_mm: 0.02,
    temperature_c: -0.22, sunlight_hours: 0.05, ndvi_index: 0.42, crop_yield_kg_ha: 0.39
  }
};

// Sample Database Records for Table Explorer
const SAMPLE_RECORDS = [
  { id: "SENS_0001", crop: "Rice", n: 90, p: 42, k: 43, ph: 6.50, moist: 49.8, hum: 82.0, rain: 202.9, ndvi: 0.74, yield: 4950 },
  { id: "SENS_0002", crop: "Rice", n: 85, p: 58, k: 41, ph: 7.04, moist: 52.4, hum: 80.3, rain: 226.7, ndvi: 0.71, yield: 5120 },
  { id: "SENS_0003", crop: "Maize", n: 60, p: 55, k: 44, ph: 7.84, moist: 56.1, hum: 82.3, rain: 264.0, ndvi: 0.85, yield: 5850 },
  { id: "SENS_0004", crop: "Maize", n: 74, p: 35, k: 40, ph: 6.98, moist: 51.2, hum: 80.2, rain: 242.9, ndvi: 0.73, yield: 4420 },
  { id: "SENS_0005", crop: "Chickpea", n: 40, p: 68, k: 79, ph: 7.36, moist: 28.5, hum: 16.9, rain: 79.5, ndvi: 0.58, yield: 3100 },
  { id: "SENS_0006", crop: "Chickpea", n: 23, p: 72, k: 76, ph: 7.84, moist: 29.2, hum: 18.0, rain: 90.8, ndvi: 0.52, yield: 2950 },
  { id: "SENS_0007", crop: "Kidneybeans", n: 21, p: 60, k: 20, ph: 5.75, moist: 34.6, hum: 21.6, rain: 158.2, ndvi: 0.62, yield: 3380 },
  { id: "SENS_0008", crop: "Pigeonpeas", n: 22, p: 67, k: 20, ph: 5.70, moist: 36.1, hum: 60.3, rain: 152.0, ndvi: 0.65, yield: 3420 },
  { id: "SENS_0009", crop: "Mothbeans", n: 22, p: 48, k: 19, ph: 5.68, moist: 24.3, hum: 55.4, rain: 51.2, ndvi: 0.44, yield: 2250 },
  { id: "SENS_0010", crop: "Cotton", n: 118, p: 46, k: 21, ph: 6.82, moist: 33.5, hum: 79.5, rain: 88.0, ndvi: 0.76, yield: 4880 }
];

// Presets
const PRESETS = {
  optimal: {
    n: 75, p: 50, k: 45, ph: 6.50, moist: 38.5, hum: 72.0, rain: 145, ndvi: 0.68, sun: 6.8, yield: 4200, temp: 24.5
  },
  drought: {
    n: 60, p: 45, k: 40, ph: 6.80, moist: 14.0, hum: 32.0, rain: 35, ndvi: 0.38, sun: 8.5, yield: 2300, temp: 34.0
  },
  depleted: {
    n: 15, p: 18, k: 15, ph: 5.20, moist: 32.0, hum: 65.0, rain: 110, ndvi: 0.42, sun: 6.0, yield: 2100, temp: 23.0
  },
  waterlogged: {
    n: 55, p: 40, k: 35, ph: 6.10, moist: 78.0, hum: 92.0, rain: 265, ndvi: 0.55, sun: 4.2, yield: 3400, temp: 21.0
  }
};

// State
let currentInputs = { ...PRESETS.optimal };

// Element Selectors
const inputs = {
  n: document.getElementById('input-n'),
  p: document.getElementById('input-p'),
  k: document.getElementById('input-k'),
  ph: document.getElementById('input-ph'),
  moist: document.getElementById('input-moist'),
  hum: document.getElementById('input-hum'),
  rain: document.getElementById('input-rain'),
  ndvi: document.getElementById('input-ndvi'),
  sun: document.getElementById('input-sun'),
  yield: document.getElementById('input-yield'),
  temp: document.getElementById('input-temp')
};

const valDisplays = {
  n: document.getElementById('val-n'),
  p: document.getElementById('val-p'),
  k: document.getElementById('val-k'),
  ph: document.getElementById('val-ph'),
  moist: document.getElementById('val-moist'),
  hum: document.getElementById('val-hum'),
  rain: document.getElementById('val-rain'),
  ndvi: document.getElementById('val-ndvi'),
  sun: document.getElementById('val-sun'),
  yield: document.getElementById('val-yield'),
  temp: document.getElementById('val-temp')
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  setupSliderListeners();
  setupPresetButtons();
  renderLoadingsTable();
  renderDataTable(SAMPLE_RECORDS);
  setupTableSearch();
  updateCalculations();
});

function setupSliderListeners() {
  Object.keys(inputs).forEach(key => {
    inputs[key].addEventListener('input', (e) => {
      currentInputs[key] = parseFloat(e.target.value);
      updateDisplayValues();
      updateCalculations();
    });
  });
}

function updateDisplayValues() {
  valDisplays.n.textContent = `${currentInputs.n} mg/kg`;
  valDisplays.p.textContent = `${currentInputs.p} mg/kg`;
  valDisplays.k.textContent = `${currentInputs.k} mg/kg`;
  valDisplays.ph.textContent = `${currentInputs.ph.toFixed(2)}`;
  valDisplays.moist.textContent = `${currentInputs.moist.toFixed(1)} %`;
  valDisplays.hum.textContent = `${currentInputs.hum.toFixed(1)} %`;
  valDisplays.rain.textContent = `${currentInputs.rain} mm`;
  valDisplays.ndvi.textContent = `${currentInputs.ndvi.toFixed(2)}`;
  valDisplays.sun.textContent = `${currentInputs.sun.toFixed(1)} hrs`;
  valDisplays.yield.textContent = `${currentInputs.yield} kg/ha`;
  valDisplays.temp.textContent = `${currentInputs.temp.toFixed(1)} °C`;
}

function setupPresetButtons() {
  const buttons = document.querySelectorAll('.preset-btn');
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const presetName = btn.dataset.preset;
      applyPreset(presetName);
    });
  });
}

function applyPreset(name) {
  if (!PRESETS[name]) return;
  currentInputs = { ...PRESETS[name] };
  Object.keys(inputs).forEach(key => {
    inputs[key].value = currentInputs[key];
  });
  updateDisplayValues();
  updateCalculations();
}

function computeZScores() {
  const z = {};
  const map = {
    nitrogen: currentInputs.n,
    phosphorus: currentInputs.p,
    potassium: currentInputs.k,
    soil_pH: currentInputs.ph,
    soil_moisture_pct: currentInputs.moist,
    humidity_pct: currentInputs.hum,
    rainfall_mm: currentInputs.rain,
    temperature_c: currentInputs.temp,
    sunlight_hours: currentInputs.sun,
    ndvi_index: currentInputs.ndvi,
    crop_yield_kg_ha: currentInputs.yield
  };

  Object.keys(SENSOR_STATS).forEach(key => {
    const stat = SENSOR_STATS[key];
    z[key] = (map[key] - stat.mean) / stat.std;
  });
  return z;
}

function computeFactorScores(z) {
  let f1 = 0, f2 = 0, f3 = 0;
  Object.keys(z).forEach(key => {
    f1 += (FACTOR_WEIGHTS.f1_fertility[key] || 0) * z[key];
    f2 += (FACTOR_WEIGHTS.f2_moisture[key] || 0) * z[key];
    f3 += (FACTOR_WEIGHTS.f3_vigor[key] || 0) * z[key];
  });

  return {
    fertility: Math.max(-2.5, Math.min(2.5, f1)),
    moisture: Math.max(-2.5, Math.min(2.5, f2)),
    vigor: Math.max(-2.5, Math.min(2.5, f3))
  };
}

function updateCalculations() {
  const z = computeZScores();
  const scores = computeFactorScores(z);

  // Update Score Elements
  updateGauge('fertility', scores.fertility);
  updateGauge('moisture', scores.moisture);
  updateGauge('vigor', scores.vigor);

  // Render Radar Chart
  drawRadarChart(scores);

  // Update Prescriptions
  updateRecommendations(scores);
}

function updateGauge(type, val) {
  const scoreElem = document.getElementById(`score-${type}`);
  const barElem = document.getElementById(`bar-${type}`);
  
  const sign = val >= 0 ? '+' : '';
  scoreElem.textContent = `${sign}${val.toFixed(2)} σ`;

  // Scale from -2.5 to +2.5 to 0% - 100%
  const pct = ((val + 2.5) / 5.0) * 100;
  barElem.style.width = `${Math.max(5, Math.min(100, pct))}%`;
}

function drawRadarChart(scores) {
  const canvas = document.getElementById('radarCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2;
  const cy = h / 2 + 5;
  const radius = 85;

  ctx.clearRect(0, 0, w, h);

  const axes = [
    { label: "F1: Fertility", val: scores.fertility, color: "#10b981" },
    { label: "F2: Moisture", val: scores.moisture, color: "#0ea5e9" },
    { label: "F3: Vigor", val: scores.vigor, color: "#8b5cf6" }
  ];

  const numAxes = axes.length;
  const angleStep = (Math.PI * 2) / numAxes;

  // Background Web
  for (let ring = 1; ring <= 4; ring++) {
    const r = (radius / 4) * ring;
    ctx.beginPath();
    for (let i = 0; i < numAxes; i++) {
      const angle = i * angleStep - Math.PI / 2;
      const x = cx + r * Math.cos(angle);
      const y = cy + r * Math.sin(angle);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.strokeStyle = ring === 4 ? "rgba(255, 255, 255, 0.2)" : "rgba(255, 255, 255, 0.08)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Draw Axis lines & Labels
  axes.forEach((axis, i) => {
    const angle = i * angleStep - Math.PI / 2;
    const xEnd = cx + radius * Math.cos(angle);
    const yEnd = cy + radius * Math.sin(angle);

    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(xEnd, yEnd);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    ctx.stroke();

    // Label
    const xLabel = cx + (radius + 24) * Math.cos(angle);
    const yLabel = cy + (radius + 20) * Math.sin(angle);
    ctx.font = "11px Inter, sans-serif";
    ctx.fillStyle = axis.color;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(axis.label, xLabel, yLabel);
  });

  // Draw Data Polygon
  ctx.beginPath();
  axes.forEach((axis, i) => {
    const angle = i * angleStep - Math.PI / 2;
    // Normalized score: -2.5 is 0.1, 0 is 0.5, +2.5 is 1.0
    const norm = Math.max(0.1, Math.min(1.0, (axis.val + 2.5) / 5.0));
    const r = radius * norm;
    const x = cx + r * Math.cos(angle);
    const y = cy + r * Math.sin(angle);

    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();

  ctx.fillStyle = "rgba(16, 185, 129, 0.25)";
  ctx.fill();
  ctx.strokeStyle = "#10b981";
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // Points
  axes.forEach((axis, i) => {
    const angle = i * angleStep - Math.PI / 2;
    const norm = Math.max(0.1, Math.min(1.0, (axis.val + 2.5) / 5.0));
    const r = radius * norm;
    const x = cx + r * Math.cos(angle);
    const y = cy + r * Math.sin(angle);

    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = axis.color;
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
}

function updateRecommendations(scores) {
  const list = document.getElementById('recList');
  if (!list) return;

  const recs = [];

  // Moisture
  if (scores.moisture < -0.8) {
    recs.push("CRITICAL MOISTURE DEFICIT: Soil moisture is severely depleted. Trigger precision drip/sprinkler irrigation immediately.");
  } else if (scores.moisture > 1.2) {
    recs.push("SATURATION WARNING: Field moisture is saturated. Halt irrigation and check sub-surface tile drainage to prevent root hypoxia.");
  } else {
    recs.push("OPTIMAL HYDROLOGY: Soil moisture and atmospheric humidity are within the balanced comfort zone for transpiration.");
  }

  // Fertility
  if (scores.fertility < -0.8) {
    recs.push("FERTILITY ALERT: N-P-K nutrient reserves are significantly depleted. Apply targeted fertigation or organic compost.");
  } else if (scores.fertility > 1.0) {
    recs.push("HIGH NUTRIENT DENSITY: Soil is nutrient-rich. Avoid excess fertilizer inputs to mitigate run-off and leaf burn.");
  } else {
    recs.push("BALANCED SOIL CHEMISTRY: N-P-K concentrations and soil pH support optimal nutrient bioavailability.");
  }

  // Vigor
  if (scores.vigor < -0.7) {
    recs.push("STRESSED CANOPY: Depressed NDVI and photosynthesis indicators. Inspect plot for pests, fungal pathogens, or salinity.");
  } else if (scores.vigor > 0.8) {
    recs.push("EXCELLENT CROP VIGOR: Robust canopy photosynthetic capacity with high yield projection. Maintain current regime.");
  } else {
    recs.push("STANDARD VIGOR: Crop growth conforms to expected regional phenological benchmarks.");
  }

  list.innerHTML = recs.map(r => `<li>${r}</li>`).join('');
}

function renderLoadingsTable() {
  const tbody = document.getElementById('matrixTableBody');
  if (!tbody) return;

  const keys = Object.keys(FACTOR_LOADINGS);
  tbody.innerHTML = keys.map(key => {
    const item = FACTOR_LOADINGS[key];
    const stat = SENSOR_STATS[key] || { unit: "" };
    const label = key.replace(/_/g, ' ').toUpperCase();

    const pillClass = (val) => {
      if (val >= 0.5) return 'load-high-pos';
      if (val >= 0.25) return 'load-med-pos';
      if (val <= -0.5) return 'load-high-neg';
      return 'load-neutral';
    };

    return `
      <tr>
        <td><strong>${label}</strong></td>
        <td><code>${stat.unit}</code></td>
        <td><span class="load-pill ${pillClass(item.f1)}">${item.f1 >= 0 ? '+' : ''}${item.f1.toFixed(3)}</span></td>
        <td><span class="load-pill ${pillClass(item.f2)}">${item.f2 >= 0 ? '+' : ''}${item.f2.toFixed(3)}</span></td>
        <td><span class="load-pill ${pillClass(item.f3)}">${item.f3 >= 0 ? '+' : ''}${item.f3.toFixed(3)}</span></td>
        <td><strong>${item.h2.toFixed(3)}</strong></td>
        <td><span style="color: #9ca3af;">${item.psi.toFixed(3)}</span></td>
      </tr>
    `;
  }).join('');
}

function renderDataTable(records) {
  const tbody = document.getElementById('dataTableBody');
  if (!tbody) return;

  tbody.innerHTML = records.map(r => `
    <tr>
      <td><code>${r.id}</code></td>
      <td><strong>${r.crop}</strong></td>
      <td>${r.n}</td>
      <td>${r.p}</td>
      <td>${r.k}</td>
      <td>${r.ph.toFixed(2)}</td>
      <td>${r.moist.toFixed(1)}%</td>
      <td>${r.hum.toFixed(1)}%</td>
      <td>${r.rain.toFixed(1)} mm</td>
      <td><span style="color: #10b981; font-weight: 600;">${r.ndvi.toFixed(2)}</span></td>
      <td>${r.yield} kg/ha</td>
    </tr>
  `).join('');

  const counter = document.getElementById('recordCounter');
  if (counter) {
    counter.textContent = `Showing ${records.length} records`;
  }
}

function setupTableSearch() {
  const searchInput = document.getElementById('searchCrop');
  if (!searchInput) return;

  searchInput.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase().trim();
    if (!query) {
      renderDataTable(SAMPLE_RECORDS);
      return;
    }
    const filtered = SAMPLE_RECORDS.filter(r => 
      r.crop.toLowerCase().includes(query) || r.id.toLowerCase().includes(query)
    );
    renderDataTable(filtered);
  });
}
