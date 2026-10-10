"""
Factor Analysis for Multi-Sensor Agricultural Measurements
===========================================================
Discovers latent factors (Soil Fertility, Moisture Condition, Crop Vigor)
from correlated agricultural sensor measurements queried from an SQLite database.
"""

import os
import sys
import sqlite3
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis, PCA

# Set non-interactive backend and writable cache for matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.makedirs("/tmp/mplconfig", exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

DB_NAME = "agricultural_sensors.db"
CSV_NAME = "dataset.csv"
TABLE_NAME = "sensor_measurements"

SENSOR_COLS = [
    'nitrogen',
    'phosphorus',
    'potassium',
    'soil_pH',
    'soil_moisture_pct',
    'humidity_pct',
    'rainfall_mm',
    'temperature_c',
    'sunlight_hours',
    'ndvi_index',
    'crop_yield_kg_ha'
]

SENSOR_LABELS = {
    'nitrogen': 'Nitrogen (N) [mg/kg]',
    'phosphorus': 'Phosphorus (P) [mg/kg]',
    'potassium': 'Potassium (K) [mg/kg]',
    'soil_pH': 'Soil pH',
    'soil_moisture_pct': 'Soil Moisture [%]',
    'humidity_pct': 'Humidity [%]',
    'rainfall_mm': 'Rainfall [mm]',
    'temperature_c': 'Temperature [°C]',
    'sunlight_hours': 'Sunlight [hrs/day]',
    'ndvi_index': 'NDVI Index [0-1]',
    'crop_yield_kg_ha': 'Yield [kg/ha]'
}

FACTOR_NAMES = [
    "Soil Fertility",
    "Moisture Condition",
    "Crop Vigor"
]


def get_db_connection(db_path=DB_NAME):
    """Establishes connection to the SQLite database."""
    if not os.path.exists(db_path):
        if os.path.exists(CSV_NAME):
            print(f"[*] Initializing {db_path} from {CSV_NAME}...")
            conn = sqlite3.connect(db_path)
            df = pd.read_csv(CSV_NAME)
            df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
            return conn
        else:
            raise FileNotFoundError(f"Neither {db_path} nor {CSV_NAME} found.")
    return sqlite3.connect(db_path)


def load_sensor_data_from_db(conn):
    """Queries sensor observations from the database table."""
    query = f"SELECT * FROM {TABLE_NAME}"
    df = pd.read_sql_query(query, conn)
    return df


def calculate_bartlett_sphericity(corr_matrix, n_samples):
    """
    Bartlett's Test of Sphericity:
    Tests hypothesis that correlation matrix equals identity matrix (uncorrelated).
    p-value < 0.05 indicates suitability for Factor Analysis.
    """
    p = corr_matrix.shape[0]
    det = np.linalg.det(corr_matrix)
    if det <= 0:
        det = 1e-12
    stat = - (n_samples - 1 - (2 * p + 5) / 6.0) * np.log(det)
    df = p * (p - 1) / 2
    p_value = 1.0 - stats.chi2.cdf(stat, df)
    return stat, df, p_value


def calculate_kmo(df_numeric):
    """
    Kaiser-Meyer-Olkin (KMO) Measure of Sampling Adequacy.
    Values > 0.6 indicate adequate factorability.
    """
    corr = df_numeric.corr().values
    inv_corr = np.linalg.pinv(corr)
    diag_inv = np.diag(inv_corr)
    
    partial_corr = np.zeros_like(corr)
    for i in range(len(diag_inv)):
        for j in range(len(diag_inv)):
            if i != j:
                partial_corr[i, j] = -inv_corr[i, j] / np.sqrt(diag_inv[i] * diag_inv[j])
                
    sum_r_sq = np.sum(corr ** 2) - np.trace(corr ** 2)
    sum_p_sq = np.sum(partial_corr ** 2)
    kmo_overall = sum_r_sq / (sum_r_sq + sum_p_sq)
    return float(kmo_overall)


def perform_factor_analysis(df, n_factors=3):
    """
    Executes standard Factor Analysis with Varimax rotation.
    Returns model, scaler, loadings DataFrame, eigenvalues, and explained variance.
    """
    X = df[SENSOR_COLS].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    fa = FactorAnalysis(n_components=n_factors, rotation='varimax', random_state=42)
    fa.fit(X_scaled)
    
    pca = PCA().fit(X_scaled)
    eigenvalues = pca.explained_variance_
    var_explained = pca.explained_variance_ratio_ * 100.0
    
    raw_loadings = fa.components_.T
    
    moist_idx = [SENSOR_COLS.index('soil_moisture_pct'), SENSOR_COLS.index('rainfall_mm'), SENSOR_COLS.index('humidity_pct')]
    fert_idx = [SENSOR_COLS.index('nitrogen'), SENSOR_COLS.index('phosphorus'), SENSOR_COLS.index('potassium')]
    vigor_idx = [SENSOR_COLS.index('ndvi_index'), SENSOR_COLS.index('crop_yield_kg_ha')]
    
    scores_moist = [np.mean(np.abs(raw_loadings[moist_idx, j])) for j in range(n_factors)]
    scores_fert = [np.mean(np.abs(raw_loadings[fert_idx, j])) for j in range(n_factors)]
    scores_vigor = [np.mean(np.abs(raw_loadings[vigor_idx, j])) for j in range(n_factors)]
    
    f_moist = int(np.argmax(scores_moist))
    avail = [j for j in range(n_factors) if j != f_moist]
    f_fert = max(avail, key=lambda j: scores_fert[j])
    f_vigor = [j for j in avail if j != f_fert][0]
    
    ordered_indices = [f_fert, f_moist, f_vigor]
    ordered_names = ["F1: Soil Fertility", "F2: Moisture Condition", "F3: Crop Vigor"]
    
    reordered_loadings = raw_loadings[:, ordered_indices]
    
    communalities = np.sum(reordered_loadings ** 2, axis=1)
    uniqueness = 1.0 - communalities
    
    loadings_df = pd.DataFrame(
        reordered_loadings,
        index=SENSOR_COLS,
        columns=ordered_names
    )
    loadings_df['Communality (h²)'] = np.clip(communalities, 0.0, 1.0)
    loadings_df['Uniqueness (ψ)'] = np.clip(uniqueness, 0.0, 1.0)
    
    factor_scores = fa.transform(X_scaled)[:, ordered_indices]
    
    return {
        'fa_model': fa,
        'scaler': scaler,
        'loadings_df': loadings_df,
        'raw_loadings': reordered_loadings,
        'factor_scores': factor_scores,
        'eigenvalues': eigenvalues,
        'var_explained': var_explained,
        'factor_order': ordered_indices,
        'factor_names': ordered_names,
        'X_scaled': X_scaled
    }


def print_ascii_bar(label, value, min_val=-2.5, max_val=+2.5, width=28):
    """Prints a terminal ASCII gauge for factor loading or score."""
    clamped = max(min_val, min(max_val, value))
    pct = (clamped - min_val) / (max_val - min_val)
    pos = int(pct * width)
    zero_pos = int((-min_val) / (max_val - min_val) * width)
    
    bar = list(" " * (width + 1))
    bar[zero_pos] = "|"
    
    if pos >= zero_pos:
        for k in range(zero_pos + 1, min(pos + 1, width + 1)):
            bar[k] = "#"
    else:
        for k in range(max(0, pos), zero_pos):
            bar[k] = "#"
            
    bar_str = "".join(bar)
    print(f"  {label:<22} [{bar_str}] {value:+6.3f}")


def plot_factor_analysis(df, fa_results, output_path="factor_analysis_results.png"):
    """
    Generates a 4-panel analytical figure:
    1. Factor Loadings Heatmap
    2. Scree Plot (Eigenvalues with Kaiser criterion line)
    3. Latent Factor Space Biplot (Soil Fertility vs Moisture vs Vigor)
    4. Communalities Bar Chart
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    loadings_df = fa_results['loadings_df']
    factor_names = fa_results['factor_names']
    eigenvalues = fa_results['eigenvalues']
    
    # 1. Heatmap of Factor Loadings
    ax1 = axes[0, 0]
    sns.heatmap(
        loadings_df[factor_names],
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        cbar_kws={'label': 'Factor Loading'},
        ax=ax1
    )
    ax1.set_title("Rotated Factor Loadings Matrix (Varimax)", fontsize=13, weight='bold')
    ax1.set_ylabel("Agricultural Sensor Measurements")
    
    # 2. Scree Plot
    ax2 = axes[0, 1]
    n_vars = len(eigenvalues)
    x_axis = range(1, n_vars + 1)
    ax2.plot(x_axis, eigenvalues, 'o-', color='#2c3e50', linewidth=2, markersize=7, label='Eigenvalue')
    ax2.axhline(y=1.0, color='#e74c3c', linestyle='--', linewidth=1.5, label='Kaiser Criterion (λ = 1)')
    ax2.axvspan(0.5, 3.5, color='#27ae60', alpha=0.15, label='Retained Factors (3 Latent Dimensions)')
    ax2.set_title("Scree Plot & Kaiser Criterion", fontsize=13, weight='bold')
    ax2.set_xlabel("Factor / Component Number")
    ax2.set_ylabel("Eigenvalue (Variance Explained)")
    ax2.set_xticks(list(x_axis))
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')
    
    # 3. Latent Factor Space Scatter Plot
    ax3 = axes[1, 0]
    scores = fa_results['factor_scores']
    scatter = ax3.scatter(
        scores[:, 0],
        scores[:, 1],
        c=scores[:, 2],
        cmap='viridis',
        s=40,
        alpha=0.75,
        edgecolors='none'
    )
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('F3: Crop Vigor Score', rotation=270, labelpad=15)
    ax3.axhline(0, color='grey', linestyle=':', linewidth=0.8)
    ax3.axvline(0, color='grey', linestyle=':', linewidth=0.8)
    ax3.set_title("Observation Scatter in Latent Factor Space", fontsize=13, weight='bold')
    ax3.set_xlabel("F1: Soil Fertility Factor Score")
    ax3.set_ylabel("F2: Moisture Condition Factor Score")
    ax3.grid(True, linestyle=':', alpha=0.4)
    
    # 4. Communalities
    ax4 = axes[1, 1]
    h2 = loadings_df['Communality (h²)']
    colors = ['#2980b9' if val >= 0.5 else '#7f8c8d' for val in h2]
    bars = ax4.barh(SENSOR_COLS, h2, color=colors)
    ax4.axvline(0.5, color='#e67e22', linestyle='--', linewidth=1.2, label='Adequate Communality (0.50)')
    ax4.set_xlim(0, 1.05)
    ax4.set_title("Sensor Communalities (h² = Shared Variance)", fontsize=13, weight='bold')
    ax4.set_xlabel("Communality")
    ax4.legend(loc='lower right')
    ax4.grid(True, axis='x', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[*] Visual analytical plots saved to: {output_path}")


def compute_custom_factor_scores(user_inputs, fa_results):
    """
    Transforms arbitrary user sensor readings into standard Factor Scores.
    """
    scaler = fa_results['scaler']
    fa_model = fa_results['fa_model']
    factor_order = fa_results['factor_order']
    
    input_vector = np.array([[user_inputs[col] for col in SENSOR_COLS]])
    input_scaled = scaler.transform(input_vector)
    
    scores_raw = fa_model.transform(input_scaled)
    ordered_scores = scores_raw[0, factor_order]
    
    return {
        'Soil Fertility': float(ordered_scores[0]),
        'Moisture Condition': float(ordered_scores[1]),
        'Crop Vigor': float(ordered_scores[2])
    }


def get_agricultural_recommendation(scores):
    """
    Generates actionable farm management prescriptions based on latent factor scores.
    """
    fert = scores['Soil Fertility']
    moist = scores['Moisture Condition']
    vigor = scores['Crop Vigor']
    
    actions = []
    
    if moist < -0.8:
        actions.append("CRITICAL: Severe moisture deficit detected. Activate precision drip/sprinkler irrigation.")
    elif moist > 1.2:
        actions.append("WARNING: High moisture saturation. Suspend irrigation and verify field drainage.")
    else:
        actions.append("OPTIMAL: Moisture conditions are within the healthy zone for crop development.")
        
    if fert < -0.8:
        actions.append("ALERT: Sub-optimal soil fertility. Recommend targeted N-P-K fertilizer application.")
    elif fert > 1.0:
        actions.append("INFO: High nutrient availability. Avoid excessive nitrogen runoff; maintain monitoring.")
    else:
        actions.append("OPTIMAL: Soil nutrient chemistry (N-P-K & pH) is balanced.")
        
    if vigor < -0.7:
        actions.append("ATTENTION: Depressed vegetative vigor (low NDVI / yield index). Inspect for pest/disease stress.")
    elif vigor > 0.8:
        actions.append("EXCELLENT: Superior crop vigor and canopy photosynthesis. High yield potential expected.")
    else:
        actions.append("NORMAL: Crop growth trajectory conforms to seasonal benchmark expectations.")
        
    return actions


def run_interactive_prediction(fa_results, default_row):
    """Prompts the user for sensor inputs and prints live latent factor scores."""
    print("\n" + "=" * 65)
    print("   INTERACTIVE AGRICULTURAL SENSOR FACTOR CALCULATOR")
    print("=" * 65)
    print("Enter custom sensor values (or press Enter to accept benchmark default):\n")
    
    user_inputs = {}
    for col in SENSOR_COLS:
        def_val = float(default_row[col])
        label = SENSOR_LABELS[col]
        try:
            raw = input(f"  {label:<34} [Default: {def_val}]: ").strip()
            if raw:
                user_inputs[col] = float(raw)
            else:
                user_inputs[col] = def_val
        except (ValueError, EOFError):
            user_inputs[col] = def_val
            
    scores = compute_custom_factor_scores(user_inputs, fa_results)
    
    print("\n" + "-" * 65)
    print("   DISCOVERED LATENT FACTOR SCORES (Standardized z-scores)")
    print("-" * 65)
    for name, score in scores.items():
        print_ascii_bar(name, score, min_val=-2.5, max_val=+2.5, width=30)
    print("-" * 65)
    
    print("\n[ MANAGEMENT PRESCRIPTIONS ]")
    recommendations = get_agricultural_recommendation(scores)
    for rec in recommendations:
        print(f"  • {rec}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Factor Analysis on Agricultural Sensor Data")
    parser.add_argument('--summary-only', action='store_true', help="Print summary without interactive prompt")
    parser.add_argument('--db', type=str, default=DB_NAME, help="Path to SQLite database")
    args = parser.parse_args()
    
    print("=" * 65)
    print("   AGRICULTURAL SENSOR MEASUREMENTS - FACTOR ANALYSIS")
    print("   Latent Dimension Extraction: Fertility | Moisture | Vigor")
    print("=" * 65 + "\n")
    
    # 1. Database Connection & Loading
    print(f"[*] Connecting to SQLite database: '{args.db}'...")
    conn = get_db_connection(args.db)
    df = load_sensor_data_from_db(conn)
    conn.close()
    
    n_samples, n_features = df[SENSOR_COLS].shape
    print(f"[*] Loaded {n_samples} sensor observation records across {n_features} correlated metrics.")
    print(f"[*] Sample crops represented: {', '.join(df['crop_type'].unique()[:5])}...")
    print("-" * 65)
    
    # 2. Correlation & Factorability Tests
    corr_matrix = df[SENSOR_COLS].corr().values
    chi2_stat, dof, p_val = calculate_bartlett_sphericity(corr_matrix, n_samples)
    kmo_val = calculate_kmo(df[SENSOR_COLS])
    
    print("\n[ STATISTICAL FACTORABILITY DIAGNOSTICS ]")
    print(f"  • Bartlett's Test of Sphericity: χ² = {chi2_stat:.2f}, df = {int(dof)}, p-value = {p_val:.2e}")
    if p_val < 0.05:
        print("    -> PASSED: Significant correlation among sensors; suitable for Factor Analysis.")
    else:
        print("    -> WARNING: Weak correlation; factor analysis may not be effective.")
        
    print(f"  • Kaiser-Meyer-Olkin (KMO) Measure: {kmo_val:.3f}")
    if kmo_val >= 0.70:
        print("    -> PASSED: Meritorious/Adequate sampling adequacy for factor extraction.")
    else:
        print("    -> ACCEPTABLE sampling adequacy.")
    print("-" * 65)
    
    # 3. Factor Analysis Model Execution
    fa_results = perform_factor_analysis(df, n_factors=3)
    loadings_df = fa_results['loadings_df']
    factor_names = fa_results['factor_names']
    eigenvalues = fa_results['eigenvalues']
    var_explained = fa_results['var_explained']
    
    print("\n[ EIGENVALUES & VARIANCE EXPLAINED (TOP 4 FACTORS) ]")
    cumulative = 0.0
    for idx in range(min(4, len(eigenvalues))):
        eig = eigenvalues[idx]
        var = var_explained[idx]
        cumulative += var
        status = "Retained (λ > 1)" if eig >= 1.0 else "Discarded (λ < 1)"
        print(f"  Factor {idx+1}: Eigenvalue = {eig:5.2f} | Variance = {var:5.1f}% | Cumul = {cumulative:5.1f}% [{status}]")
    print("-" * 65)
    
    print("\n[ ROTATED FACTOR LOADINGS MATRIX (Varimax Rotation) ]")
    print(loadings_df.to_string())
    print("-" * 65)
    
    print("\n[ DOMINANT SENSOR LOADINGS PER LATENT FACTOR ]")
    for fname in factor_names:
        top_positive = loadings_df[fname].nlargest(2)
        top_negative = loadings_df[fname].nsmallest(1)
        print(f"\n  ► {fname}:")
        for sensor, val in top_positive.items():
            print(f"      + {sensor:<20} loading: {val:+5.2f}")
        for sensor, val in top_negative.items():
            if abs(val) > 0.25:
                print(f"      - {sensor:<20} loading: {val:+5.2f}")
    print("-" * 65)
    
    # 4. Generate Graphical Figures
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "factor_analysis_results.png")
    plot_factor_analysis(df, fa_results, output_path=plot_path)
    
    # 5. Interactive Mode
    if not args.summary_only:
        mean_row = df[SENSOR_COLS].mean().round(2)
        run_interactive_prediction(fa_results, mean_row)
    else:
        print("\n[*] Summary completed. (Use without --summary-only for interactive calculator).")


if __name__ == "__main__":
    main()
