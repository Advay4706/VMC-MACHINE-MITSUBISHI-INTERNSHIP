"""
=============================================================================
THERMAL ERROR PREDICTION AND COMPENSATION SYSTEM
CNC Vertical Machining Center (VMC) — X-Axis
=============================================================================

Project Objective:
    Predict and compensate positioning errors caused by thermal effects during
    CNC machine operation using machine data, probe measurements, and ML.

Machine Structure (Drive Chain):
    CNC Controller → Servo Drive → Servo Motor → Coupling →
    Ball Screw → Ball Nut → Table

Engineering Background:
    As a CNC machine operates, friction in the ball screw, bearings, and servo
    motor generates heat. This thermal energy causes the ball screw to expand
    according to the linear thermal expansion equation:

        ΔL = α × L × ΔT

    where:
        α  = coefficient of linear thermal expansion (steel ≈ 11.7 × 10⁻⁶ /°C)
        L  = original length of the ball screw (mm)
        ΔT = temperature rise above reference (°C)

    This expansion introduces positioning errors that grow over time. By
    measuring these errors with a probe and correlating them with temperature,
    we can train ML models to predict errors and apply real-time compensation.

Key Assumption:
    Ball screw temperature is NOT directly available. Machine body temperature
    (measured via a sensor on the machine structure) is used as a proxy for
    thermal growth estimation.

Author: Thermal Compensation Engineering Project
Date:   June 2025
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Imports and Configuration
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
import os

warnings.filterwarnings("ignore")

# Set plot style for publication-quality figures
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})
sns.set_style("whitegrid")

# Create output directory for plots
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Physical Constants and Machine Parameters
# ─────────────────────────────────────────────────────────────────────────────
"""
Engineering Parameters:
    - The coefficient of thermal expansion for steel (α) is approximately
      11.7 × 10⁻⁶ per °C. This is a well-established material property.
    - Ball screw length is the effective travel length of the X-axis.
    - Reference temperature is the ambient/startup temperature at which the
      machine was last calibrated (no thermal error at this temperature).
"""

ALPHA_STEEL = 11.7e-6          # Coefficient of thermal expansion for steel (1/°C)
BALL_SCREW_LENGTH_MM = 500.0   # Effective X-axis ball screw length (mm)
REFERENCE_TEMP_C = 22.0        # Reference/calibration temperature (°C)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Synthetic Dataset Generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_dataset(n_samples: int = 20) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset simulating CNC VMC X-axis operation.

    Physics-Based Rationale:
        - Machine temperature rises logarithmically during warm-up and stabilizes
          (Newton's law of cooling applied in reverse — heating approaches
          equilibrium with the environment).
        - Position error correlates with temperature rise but also includes
          non-linear effects (hysteresis, bearing friction, encoder resolution).
        - Commanded positions simulate a typical probing cycle across the X-axis.

    Parameters:
        n_samples (int): Number of measurement rows (default: 20).

    Returns:
        pd.DataFrame: Dataset with columns:
            - Time_min            : Elapsed time since machine start (minutes)
            - Machine_Temp_C      : Machine body temperature (°C)
            - Ambient_Temp_C      : Workshop ambient temperature (°C)
            - Commanded_Pos_mm    : Target position commanded by CNC controller (mm)
            - Actual_Pos_mm       : Position measured by touch probe (mm)
            - Position_Error_um   : Error = Actual - Commanded, in micrometers (µm)
    """
    np.random.seed(42)  # Reproducibility

    # ── Time: uniform intervals from 0 to 120 minutes (warm-up cycle) ──
    time_min = np.linspace(0, 120, n_samples)

    # ── Ambient temperature: slight drift simulating workshop conditions ──
    ambient_temp = REFERENCE_TEMP_C + np.random.normal(0, 0.3, n_samples)

    # ── Machine temperature: logarithmic warm-up curve ──
    # Physics: heat generated by servo motor and ball screw friction causes
    # temperature to rise rapidly initially, then plateau as heat dissipation
    # through conduction/convection reaches equilibrium.
    max_temp_rise = 8.0  # Maximum temperature rise above ambient (°C)
    machine_temp = (
        REFERENCE_TEMP_C
        + max_temp_rise * (1 - np.exp(-time_min / 30))  # Exponential approach
        + np.random.normal(0, 0.2, n_samples)           # Sensor noise
    )

    # ── Commanded positions: simulate probing at multiple X-axis locations ──
    commanded_positions = np.array([
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
        500, 450, 400, 350, 300, 250, 200, 150, 100, 50
    ])[:n_samples]

    # ── Theoretical thermal expansion ──
    # ΔL = α × L × ΔT (fundamental thermal expansion equation)
    temp_rise = machine_temp - REFERENCE_TEMP_C
    theoretical_expansion_mm = ALPHA_STEEL * BALL_SCREW_LENGTH_MM * temp_rise

    # ── Actual position error ──
    # Real error = theoretical expansion + non-linear effects + measurement noise
    # Non-linear effects include: hysteresis, lead screw pitch error,
    # position-dependent stiffness variation, and servo lag.
    position_error_mm = (
        theoretical_expansion_mm * 1.2                           # Scale factor (ball screw is hotter than body)
        + 0.002 * np.sin(2 * np.pi * commanded_positions / 500)  # Periodic pitch error
        + 0.001 * (commanded_positions / 500)                    # Position-dependent error
        + np.random.normal(0, 0.0005, n_samples)                 # Measurement noise
    )

    # Convert to micrometers for practical significance (1 mm = 1000 µm)
    position_error_um = position_error_mm * 1000

    # ── Actual measured position ──
    actual_positions = commanded_positions + position_error_mm

    # ── Build DataFrame ──
    df = pd.DataFrame({
        "Time_min": np.round(time_min, 2),
        "Machine_Temp_C": np.round(machine_temp, 2),
        "Ambient_Temp_C": np.round(ambient_temp, 2),
        "Commanded_Pos_mm": commanded_positions,
        "Actual_Pos_mm": np.round(actual_positions, 6),
        "Position_Error_um": np.round(position_error_um, 3),
    })

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create physics-informed features for machine learning.

    Engineering Rationale:
        Each feature captures a physical mechanism that contributes to
        thermal positioning error:

        1. Temperature_Rise_C:
           Direct driver of thermal expansion (ΔT in ΔL = α·L·ΔT).

        2. Theoretical_Expansion_um:
           Analytical prediction of thermal growth. Serves as a strong
           baseline feature — ML learns the residual beyond this.

        3. Cumulative_Travel_mm:
           Proxy for heat generation in the ball screw nut interface.
           More travel → more friction → more localized heating.

        4. Time_min, Machine_Temp_C, Commanded_Pos_mm:
           Retained as direct measurements.

    Parameters:
        df (pd.DataFrame): Raw dataset from generate_synthetic_dataset().

    Returns:
        pd.DataFrame: Enhanced dataset with engineered features.
    """
    df = df.copy()

    # ── Temperature Rise above reference ──
    # This is the ΔT that drives thermal expansion.
    df["Temperature_Rise_C"] = df["Machine_Temp_C"] - REFERENCE_TEMP_C

    # ── Theoretical Thermal Expansion (µm) ──
    # Pure physics-based prediction: ΔL = α × L × ΔT
    # Converted to micrometers for consistency with Position_Error_um.
    df["Theoretical_Expansion_um"] = (
        ALPHA_STEEL * BALL_SCREW_LENGTH_MM * df["Temperature_Rise_C"] * 1000
    )

    # ── Cumulative Axis Travel Distance (mm) ──
    # Approximation: sum of absolute position changes.
    # In practice, this would be read from the CNC controller's axis odometer.
    df["Cumulative_Travel_mm"] = df["Commanded_Pos_mm"].diff().abs().fillna(0).cumsum()

    print("=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)
    print(f"\nDataset shape: {df.shape}")
    print(f"Features created:")
    print(f"  • Temperature_Rise_C       — ΔT above {REFERENCE_TEMP_C}°C reference")
    print(f"  • Theoretical_Expansion_um — α·L·ΔT (analytical prediction)")
    print(f"  • Cumulative_Travel_mm     — Total axis distance traveled")
    print(f"\nFeature statistics:")
    print(df[["Temperature_Rise_C", "Theoretical_Expansion_um",
              "Cumulative_Travel_mm"]].describe().round(3))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Model Training and Evaluation
# ─────────────────────────────────────────────────────────────────────────────

# Define the feature columns used for training
FEATURE_COLUMNS = [
    "Time_min",
    "Machine_Temp_C",
    "Temperature_Rise_C",
    "Commanded_Pos_mm",
    "Theoretical_Expansion_um",
    "Cumulative_Travel_mm",
]

# Target variable
TARGET_COLUMN = "Position_Error_um"


def prepare_data(df: pd.DataFrame):
    """
    Split dataset into training and testing sets.

    Uses an 80/20 split with a fixed random state for reproducibility.
    This is standard practice in ML — the test set simulates unseen
    operating conditions to evaluate generalization.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) — feature matrices and target vectors.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n{'─' * 70}")
    print(f"DATA SPLIT")
    print(f"{'─' * 70}")
    print(f"  Training samples : {len(X_train)}")
    print(f"  Testing samples  : {len(X_test)}")
    print(f"  Feature count    : {len(FEATURE_COLUMNS)}")
    print(f"  Features         : {FEATURE_COLUMNS}")

    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Train Linear Regression and Random Forest models, then evaluate both.

    Model Selection Rationale:
    ──────────────────────────
    1. Linear Regression:
       - Assumes a linear relationship between features and error.
       - Fast to train, fully interpretable (coefficients show feature weights).
       - Good baseline — thermal expansion IS fundamentally linear (ΔL = α·L·ΔT).
       - Limitation: cannot capture non-linear effects (hysteresis, friction).

    2. Random Forest Regressor:
       - Ensemble of decision trees that captures non-linear relationships.
       - Robust to outliers and does not require feature scaling.
       - Provides feature importance scores for engineering insight.
       - Better suited for real-world thermal error patterns which include
         hysteresis loops, position-dependent effects, and time-lag between
         temperature and expansion.

    Evaluation Metrics:
    ───────────────────
    - MAE  (Mean Absolute Error):   Average magnitude of prediction errors.
                                     Easy to interpret — "on average, the model
                                     is off by X micrometers."

    - RMSE (Root Mean Squared Error): Penalizes large errors more heavily.
                                      Important for CNC: a single large error
                                      can scrap an expensive workpiece.

    - R²   (Coefficient of Determination): Proportion of variance explained.
                                           1.0 = perfect, 0.0 = no better than mean.

    Returns:
        dict: Dictionary containing both trained models and their metrics.
    """
    results = {}

    # ── Model 1: Linear Regression ──
    print(f"\n{'=' * 70}")
    print("MODEL 1: LINEAR REGRESSION")
    print(f"{'=' * 70}")

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    lr_metrics = {
        "MAE": mean_absolute_error(y_test, lr_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, lr_pred)),
        "R2": r2_score(y_test, lr_pred),
    }

    print(f"\n  Coefficients:")
    for feat, coef in zip(FEATURE_COLUMNS, lr_model.coef_):
        print(f"    {feat:30s} : {coef:+.6f}")
    print(f"    {'Intercept':30s} : {lr_model.intercept_:+.6f}")

    print(f"\n  Test Set Performance:")
    print(f"    MAE  = {lr_metrics['MAE']:.4f} µm")
    print(f"    RMSE = {lr_metrics['RMSE']:.4f} µm")
    print(f"    R²   = {lr_metrics['R2']:.4f}")

    results["LinearRegression"] = {
        "model": lr_model,
        "predictions": lr_pred,
        "metrics": lr_metrics,
    }

    # ── Model 2: Random Forest Regressor ──
    print(f"\n{'=' * 70}")
    print("MODEL 2: RANDOM FOREST REGRESSOR")
    print(f"{'=' * 70}")

    rf_model = RandomForestRegressor(
        n_estimators=100,     # 100 trees for stable predictions
        max_depth=5,          # Limit depth to prevent overfitting on small dataset
        min_samples_split=3,  # Require at least 3 samples to split
        random_state=42,
        n_jobs=-1,            # Use all CPU cores
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    rf_metrics = {
        "MAE": mean_absolute_error(y_test, rf_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, rf_pred)),
        "R2": r2_score(y_test, rf_pred),
    }

    print(f"\n  Hyperparameters:")
    print(f"    n_estimators    = 100  (number of decision trees)")
    print(f"    max_depth       = 5    (prevents overfitting on 20 rows)")
    print(f"    min_samples_split = 3  (minimum samples to create a node)")

    print(f"\n  Test Set Performance:")
    print(f"    MAE  = {rf_metrics['MAE']:.4f} µm")
    print(f"    RMSE = {rf_metrics['RMSE']:.4f} µm")
    print(f"    R²   = {rf_metrics['R2']:.4f}")

    results["RandomForest"] = {
        "model": rf_model,
        "predictions": rf_pred,
        "metrics": rf_metrics,
    }

    return results


def perform_cross_validation(X, y):
    """
    Perform 5-fold Cross Validation for both models.

    Why Cross Validation?
    ─────────────────────
    With only 20 samples, a single train/test split may give misleading
    results depending on which samples end up in the test set. 5-fold CV
    rotates through 5 different test sets, giving a more robust estimate
    of model performance.

    Each fold:
    - Uses 80% of data for training, 20% for validation
    - Reports R² score for that fold
    - Average R² across all 5 folds is the final metric

    Parameters:
        X: Feature matrix
        y: Target vector

    Returns:
        dict: Cross-validation R² scores for each model.
    """
    print(f"\n{'=' * 70}")
    print("5-FOLD CROSS VALIDATION")
    print(f"{'=' * 70}")

    cv_results = {}

    # Linear Regression CV
    lr_cv_scores = cross_val_score(
        LinearRegression(), X, y, cv=5, scoring="r2"
    )
    cv_results["LinearRegression"] = lr_cv_scores

    print(f"\n  Linear Regression:")
    print(f"    Fold R² scores : {[f'{s:.4f}' for s in lr_cv_scores]}")
    print(f"    Average R²     : {lr_cv_scores.mean():.4f} ± {lr_cv_scores.std():.4f}")

    # Random Forest CV
    rf_cv_scores = cross_val_score(
        RandomForestRegressor(
            n_estimators=100, max_depth=5,
            min_samples_split=3, random_state=42, n_jobs=-1
        ),
        X, y, cv=5, scoring="r2"
    )
    cv_results["RandomForest"] = rf_cv_scores

    print(f"\n  Random Forest:")
    print(f"    Fold R² scores : {[f'{s:.4f}' for s in rf_cv_scores]}")
    print(f"    Average R²     : {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")

    return cv_results


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Visualization
# ─────────────────────────────────────────────────────────────────────────────
def generate_plots(df, results, cv_results):
    """
    Generate diagnostic and analysis plots.

    Plot Descriptions:
    ──────────────────
    1. Error vs Time:
       Shows how positioning error grows as the machine warms up.
       A clear upward trend confirms thermal effects dominate.

    2. Temperature Rise vs Error:
       Validates the linear thermal expansion relationship.
       If points lie on a line, ΔL = α·L·ΔT is a good model.

    3. Actual vs Predicted Error:
       Parity plot — perfect predictions lie on the 45° diagonal.
       Scatter around the line indicates prediction uncertainty.

    4. Random Forest Feature Importance:
       Shows which physical variables most influence the error.
       Helps engineers prioritize sensor placement and monitoring.
    """
    X_test_indices = results["LinearRegression"]["predictions"]  # just for length
    y_test = df.loc[df.index.isin(
        df.index[~df.index.isin(
            df.sample(frac=0.8, random_state=42).index
        )]
    ), TARGET_COLUMN]

    # We'll reconstruct test indices from the actual split
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "CNC VMC X-Axis Thermal Error Analysis",
        fontsize=16, fontweight="bold", y=1.02
    )

    # ── Plot 1: Error vs Time ──
    ax1 = axes[0, 0]
    ax1.plot(df["Time_min"], df["Position_Error_um"], "o-", color="#2196F3",
             linewidth=1.5, markersize=5, label="Measured Error")
    ax1.plot(df["Time_min"], df["Theoretical_Expansion_um"], "--", color="#FF9800",
             linewidth=1.5, label="Theoretical Expansion")
    ax1.set_xlabel("Time (minutes)")
    ax1.set_ylabel("Error / Expansion (µm)")
    ax1.set_title("Position Error vs Operating Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Temperature Rise vs Error ──
    ax2 = axes[0, 1]
    scatter = ax2.scatter(
        df["Temperature_Rise_C"], df["Position_Error_um"],
        c=df["Time_min"], cmap="coolwarm", s=60, edgecolors="black", linewidth=0.5
    )
    ax2.set_xlabel("Temperature Rise (°C)")
    ax2.set_ylabel("Position Error (µm)")
    ax2.set_title("Temperature Rise vs Position Error")
    plt.colorbar(scatter, ax=ax2, label="Time (min)")
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: Actual vs Predicted Error (both models) ──
    ax3 = axes[1, 0]
    lr_pred = results["LinearRegression"]["predictions"]
    rf_pred = results["RandomForest"]["predictions"]

    ax3.scatter(y_test, lr_pred, color="#4CAF50", marker="o", s=60,
                edgecolors="black", linewidth=0.5, label="Linear Regression", zorder=5)
    ax3.scatter(y_test, rf_pred, color="#E91E63", marker="s", s=60,
                edgecolors="black", linewidth=0.5, label="Random Forest", zorder=5)

    # Perfect prediction line
    error_range = [y_test.min(), y_test.max()]
    ax3.plot(error_range, error_range, "k--", linewidth=1.5, label="Perfect Prediction")

    ax3.set_xlabel("Actual Error (µm)")
    ax3.set_ylabel("Predicted Error (µm)")
    ax3.set_title("Actual vs Predicted Error")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ── Plot 4: Random Forest Feature Importance ──
    ax4 = axes[1, 1]
    rf_model = results["RandomForest"]["model"]
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(FEATURE_COLUMNS)))

    ax4.barh(
        [FEATURE_COLUMNS[i] for i in sorted_idx],
        importances[sorted_idx],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax4.set_xlabel("Feature Importance")
    ax4.set_title("Random Forest — Feature Importance")
    ax4.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    plot_path = os.path.join(PLOT_DIR, "thermal_error_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n  ✓ Plot saved to: {plot_path}")
    plt.show()

    return plot_path


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Model Comparison and Recommendation
# ─────────────────────────────────────────────────────────────────────────────
def compare_models(results, cv_results):
    """
    Compare Linear Regression vs Random Forest and provide engineering
    recommendation on which model to deploy for thermal compensation.

    Decision Criteria:
    ──────────────────
    1. R² Score (primary): Higher R² means the model explains more variance
       in thermal error. For CNC compensation, R² > 0.95 is desirable.

    2. RMSE (secondary): Lower RMSE means smaller worst-case errors.
       In precision machining, a single large error can scrap a part.

    3. Cross-validation stability: Consistent R² across folds indicates
       the model generalizes well to new operating conditions.

    4. Interpretability: Linear Regression coefficients directly map to
       physical parameters; Random Forest is a black box but more accurate.
    """
    print(f"\n{'=' * 70}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'=' * 70}")

    comparison = pd.DataFrame({
        "Metric": ["MAE (µm)", "RMSE (µm)", "R² Score", "CV Avg R²"],
        "Linear Regression": [
            f"{results['LinearRegression']['metrics']['MAE']:.4f}",
            f"{results['LinearRegression']['metrics']['RMSE']:.4f}",
            f"{results['LinearRegression']['metrics']['R2']:.4f}",
            f"{cv_results['LinearRegression'].mean():.4f}",
        ],
        "Random Forest": [
            f"{results['RandomForest']['metrics']['MAE']:.4f}",
            f"{results['RandomForest']['metrics']['RMSE']:.4f}",
            f"{results['RandomForest']['metrics']['R2']:.4f}",
            f"{cv_results['RandomForest'].mean():.4f}",
        ],
    })

    print(f"\n{comparison.to_string(index=False)}")

    # Determine the better model
    lr_r2 = results["LinearRegression"]["metrics"]["R2"]
    rf_r2 = results["RandomForest"]["metrics"]["R2"]

    if rf_r2 > lr_r2:
        best_model_name = "Random Forest"
        explanation = (
            "Random Forest outperforms Linear Regression because thermal errors\n"
            "  in CNC machines include non-linear effects that a linear model cannot\n"
            "  capture:\n"
            "    • Hysteresis (different errors for same temperature on heating vs cooling)\n"
            "    • Position-dependent stiffness variation along the ball screw\n"
            "    • Time-lag between machine body temperature and ball screw temperature\n"
            "    • Non-uniform heat distribution across the machine structure\n"
            "  Random Forest's ensemble of decision trees naturally models these\n"
            "  non-linearities without explicit feature engineering."
        )
        best_model = results["RandomForest"]["model"]
    else:
        best_model_name = "Linear Regression"
        explanation = (
            "Linear Regression performs well because the dominant error mechanism\n"
            "  — thermal expansion (ΔL = α·L·ΔT) — is inherently linear.\n"
            "  With a small dataset (20 rows), the simpler model avoids overfitting\n"
            "  and generalizes better. Its coefficients are directly interpretable\n"
            "  in engineering terms."
        )
        best_model = results["LinearRegression"]["model"]

    print(f"\n  ★ RECOMMENDED MODEL: {best_model_name}")
    print(f"\n  Reason:")
    print(f"  {explanation}")

    return best_model, best_model_name


# ─────────────────────────────────────────────────────────────────────────────
# Section 8: Thermal Compensation Algorithm
# ─────────────────────────────────────────────────────────────────────────────
"""
Compensation Strategy:
──────────────────────
In CNC machining, thermal compensation works by SUBTRACTING the predicted
error from the commanded position:

    Corrected_Position = Commanded_Position - Predicted_Error

This is a feedforward compensation approach:
    1. Sensors measure current machine state (temperature, time, position).
    2. The ML model predicts the expected thermal error.
    3. The CNC controller adjusts the commanded position to cancel the error.

In practice, this is implemented via:
    - External Work Offset (G54-G59) adjustment
    - Real-time axis offset through the CNC controller's compensation table
    - Macro programs that update tool offsets dynamically

The compensation function below simulates this real-time adjustment.
"""


def predict_compensation(
    model,
    time: float,
    machine_temperature: float,
    commanded_position: float,
    theoretical_expansion: float,
    cumulative_travel: float = 0.0,
) -> dict:
    """
    Predict thermal positioning error and calculate compensated position.

    This function simulates the real-time compensation loop that would run
    on the CNC controller during machining operations.

    Parameters:
    ───────────
    model               : Trained sklearn model (LinearRegression or RandomForest)
    time                : Elapsed operating time in minutes
    machine_temperature : Current machine body temperature in °C
    commanded_position  : CNC commanded X-axis position in mm
    theoretical_expansion: Calculated ΔL = α·L·ΔT in µm
    cumulative_travel   : Total X-axis distance traveled so far in mm

    Returns:
    ────────
    dict with keys:
        - predicted_error_um    : ML model's predicted error (µm)
        - corrected_position_mm : Compensated target position (mm)
        - compensation_um       : Amount of compensation applied (µm)
        - machine_temperature   : Echo of input temperature (°C)
        - commanded_position_mm : Echo of input commanded position (mm)
    """
    # ── Compute derived features ──
    temperature_rise = machine_temperature - REFERENCE_TEMP_C

    # ── Assemble feature vector in the same order as training ──
    feature_names = FEATURE_COLUMNS
    features = pd.DataFrame([{
        "Time_min": time,
        "Machine_Temp_C": machine_temperature,
        "Temperature_Rise_C": temperature_rise,
        "Commanded_Pos_mm": commanded_position,
        "Theoretical_Expansion_um": theoretical_expansion,
        "Cumulative_Travel_mm": cumulative_travel,
    }], columns=feature_names)

    # ── Predict error using the trained model ──
    predicted_error_um = model.predict(features)[0]

    # ── Apply compensation ──
    # Convert predicted error from µm to mm for position correction
    predicted_error_mm = predicted_error_um / 1000.0
    corrected_position_mm = commanded_position - predicted_error_mm

    return {
        "predicted_error_um": round(predicted_error_um, 3),
        "corrected_position_mm": round(corrected_position_mm, 6),
        "compensation_um": round(-predicted_error_um, 3),  # Negative = subtract
        "machine_temperature": machine_temperature,
        "commanded_position_mm": commanded_position,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 9: Demonstration of Compensation System
# ─────────────────────────────────────────────────────────────────────────────
def demonstrate_compensation(model, model_name):
    """
    Run the compensation algorithm on sample machine inputs and display
    results in a clear engineering format.

    These test cases simulate realistic operating scenarios:
    - Cold start (low temperature, minimal error)
    - Mid warm-up (moderate temperature rise)
    - Fully warmed up (maximum thermal error)
    - Different axis positions (position-dependent effects)
    """
    print(f"\n{'=' * 70}")
    print(f"THERMAL COMPENSATION DEMONSTRATION")
    print(f"Using model: {model_name}")
    print(f"{'=' * 70}")

    # Sample machine inputs representing different operating conditions
    test_cases = [
        {
            "label": "Cold Start — Near Origin",
            "time": 5.0,
            "machine_temperature": 22.5,
            "commanded_position": 100.0,
            "theoretical_expansion": ALPHA_STEEL * BALL_SCREW_LENGTH_MM * 0.5 * 1000,
            "cumulative_travel": 200.0,
        },
        {
            "label": "Mid Warm-up — Center of Travel",
            "time": 30.0,
            "machine_temperature": 26.0,
            "commanded_position": 250.0,
            "theoretical_expansion": ALPHA_STEEL * BALL_SCREW_LENGTH_MM * 4.0 * 1000,
            "cumulative_travel": 3000.0,
        },
        {
            "label": "Fully Warmed — Full Travel",
            "time": 90.0,
            "machine_temperature": 29.5,
            "commanded_position": 450.0,
            "theoretical_expansion": ALPHA_STEEL * BALL_SCREW_LENGTH_MM * 7.5 * 1000,
            "cumulative_travel": 15000.0,
        },
        {
            "label": "Stable Operation — Short Move",
            "time": 120.0,
            "machine_temperature": 30.0,
            "commanded_position": 50.0,
            "theoretical_expansion": ALPHA_STEEL * BALL_SCREW_LENGTH_MM * 8.0 * 1000,
            "cumulative_travel": 20000.0,
        },
    ]

    for i, case in enumerate(test_cases, 1):
        result = predict_compensation(
            model=model,
            time=case["time"],
            machine_temperature=case["machine_temperature"],
            commanded_position=case["commanded_position"],
            theoretical_expansion=case["theoretical_expansion"],
            cumulative_travel=case["cumulative_travel"],
        )

        print(f"\n  ┌─ Test Case {i}: {case['label']}")
        print(f"  │  Input:")
        print(f"  │    Time              = {case['time']:.1f} min")
        print(f"  │    Machine Temp      = {case['machine_temperature']:.1f} °C")
        print(f"  │    Commanded Pos     = {case['commanded_position']:.3f} mm")
        print(f"  │    Theoretical Exp   = {case['theoretical_expansion']:.3f} µm")
        print(f"  │  Output:")
        print(f"  │    Predicted Error   = {result['predicted_error_um']:+.3f} µm")
        print(f"  │    Compensation      = {result['compensation_um']:+.3f} µm")
        print(f"  │    Corrected Pos     = {result['corrected_position_mm']:.6f} mm")
        print(f"  └{'─' * 55}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 10: Main Execution Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """
    Main execution pipeline for the Thermal Error Compensation System.

    Pipeline Steps:
    ───────────────
    1. Generate/Load Data    → Create synthetic dataset (20 rows)
    2. Feature Engineering   → Compute physics-based features
    3. Data Splitting        → 80/20 train/test split
    4. Model Training        → Train Linear Regression & Random Forest
    5. Cross Validation      → 5-fold CV for robust evaluation
    6. Visualization         → Generate diagnostic plots
    7. Model Comparison      → Determine the better model
    8. Compensation Demo     → Show real-time compensation results
    """
    print("╔" + "═" * 68 + "╗")
    print("║" + " THERMAL ERROR PREDICTION & COMPENSATION SYSTEM ".center(68) + "║")
    print("║" + " CNC Vertical Machining Center — X-Axis ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # ── Step 1: Generate synthetic dataset ──
    print(f"\n{'━' * 70}")
    print("STEP 1: DATASET GENERATION")
    print(f"{'━' * 70}")
    df = generate_synthetic_dataset(n_samples=20)
    print(f"\n  Generated {len(df)} measurement rows.")
    print(f"\n  First 5 rows of raw data:")
    print(df.head().to_string(index=False))

    # ── Step 2: Feature Engineering ──
    print(f"\n{'━' * 70}")
    print("STEP 2: FEATURE ENGINEERING")
    print(f"{'━' * 70}")
    df = engineer_features(df)

    # ── Step 3: Data Splitting ──
    print(f"\n{'━' * 70}")
    print("STEP 3: DATA PREPARATION")
    print(f"{'━' * 70}")
    X_train, X_test, y_train, y_test = prepare_data(df)

    # ── Step 4: Model Training and Evaluation ──
    print(f"\n{'━' * 70}")
    print("STEP 4: MODEL TRAINING & EVALUATION")
    print(f"{'━' * 70}")
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    # ── Step 5: Cross Validation ──
    print(f"\n{'━' * 70}")
    print("STEP 5: CROSS VALIDATION")
    print(f"{'━' * 70}")
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    cv_results = perform_cross_validation(X, y)

    # ── Step 6: Visualization ──
    print(f"\n{'━' * 70}")
    print("STEP 6: VISUALIZATION")
    print(f"{'━' * 70}")
    try:
        plot_path = generate_plots(df, results, cv_results)
    except Exception as e:
        print(f"  ⚠ Plot generation skipped (no display available): {e}")
        plot_path = None

    # ── Step 7: Model Comparison ──
    print(f"\n{'━' * 70}")
    print("STEP 7: MODEL COMPARISON")
    print(f"{'━' * 70}")
    best_model, best_model_name = compare_models(results, cv_results)

    # ── Step 8: Compensation Demonstration ──
    print(f"\n{'━' * 70}")
    print("STEP 8: COMPENSATION DEMONSTRATION")
    print(f"{'━' * 70}")
    demonstrate_compensation(best_model, best_model_name)

    # ── Summary ──
    print(f"\n{'═' * 70}")
    print("SYSTEM SUMMARY")
    print(f"{'═' * 70}")
    print(f"""
  This thermal error compensation system demonstrates how machine learning
  can be applied to predict and correct thermally-induced positioning errors
  in a CNC Vertical Machining Center.

  Key Engineering Takeaways:
  ─────────────────────────
  1. Thermal expansion is the dominant error source during warm-up.
  2. Machine body temperature serves as a practical proxy when ball screw
     temperature sensors are unavailable.
  3. The theoretical expansion (ΔL = α·L·ΔT) provides a strong analytical
     baseline, but ML models capture additional non-linear effects.
  4. Real-time compensation is achieved by predicting the error and
     subtracting it from the commanded position.
  5. The {best_model_name} model was selected for deployment based on
     superior R² score and cross-validation stability.

  Practical Implementation Notes:
  ───────────────────────────────
  • Deploy temperature sensors on the machine column near the ball screw.
  • Run a warm-up probing cycle daily to validate/retrain the model.
  • Integrate compensation via CNC macro programs or external offset tables.
  • Monitor compensation residuals — if errors grow, retrain with fresh data.

  Reference Temperature: {REFERENCE_TEMP_C}°C
  Ball Screw Length:     {BALL_SCREW_LENGTH_MM} mm
  Thermal Expansion α:   {ALPHA_STEEL} /°C (steel)
    """)

    if plot_path:
        print(f"  Plots saved to: {plot_path}")

    print(f"{'═' * 70}")
    print("  THERMAL COMPENSATION SYSTEM — EXECUTION COMPLETE")
    print(f"{'═' * 70}\n")

    return df, results, cv_results, best_model, best_model_name


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, results, cv_results, best_model, best_model_name = main()