"""
market_predict.py
====================
Seamlessly ingests the 22-row seasonal snapshot from market_harvester.py.
Calculates pure expected points and DNF probabilities without requiring
RoundNumber logic.
"""

import json
import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("market_predict")

DRIVERS_INPUT = Path("drivers_ml_ready.csv")
CONSTRUCTORS_INPUT = Path("constructors_ml_ready.csv")
OUTPUT_JSON = Path("market_predictions.json")

# Track Profile (Update week-to-week as needed)
TRACK_CALENDAR = {
    "Monaco Grand Prix": {"is_street": True, "circuit_dnf_rate": 0.18},
    "Melbourne Grand Prix": {"is_street": True, "circuit_dnf_rate": 0.12},
    "Silverstone Grand Prix": {"is_street": False, "circuit_dnf_rate": 0.08},
}
DEFAULT_TRACK = "Monaco Grand Prix"
IS_STREET = TRACK_CALENDAR[DEFAULT_TRACK]["is_street"]
CIRCUIT_DNF_RATE = TRACK_CALENDAR[DEFAULT_TRACK]["circuit_dnf_rate"]

def load_data():
    if not DRIVERS_INPUT.exists() or not CONSTRUCTORS_INPUT.exists():
        raise FileNotFoundError("Missing ML-ready CSV files. Run market_harvester.py first.")
    df_d = pd.read_csv(DRIVERS_INPUT).fillna(0)
    df_c = pd.read_csv(CONSTRUCTORS_INPUT).fillna(0)
    return df_d, df_c

def run_pipeline():
    log.info("Starting baseline prediction validation...")
    df_d, df_c = load_data()
    
    d_exp, d_dnf, d_base, d_costs = {}, {}, {}, {}
    
    # Safely get max historical DNFs for probability math
    max_d_dnf = max(df_d["DNFs"].max(), 1) if "DNFs" in df_d.columns else 1
    
    # 1. Evaluate Drivers
    for _, row in df_d.iterrows():
        name = str(row["Driver"]).strip()
        d_costs[name] = float(row["Cost"])
        
        # Pace multiplier based on raw speed
        delta_factor = (row.get("FP1_DeltaBest", 1.0) + row.get("FP2_DeltaBest", 1.0) + row.get("FP3_DeltaBest", 1.0)) / 3.0
        pace_multiplier = np.clip(1.3 - (delta_factor * 0.15), 0.5, 1.5)
        
        # Calculate Base Score using bounded progression
        avg_pts = row.get("AvgPoints", 10.0)
        progression = np.clip(row.get("FP_Progression", 0.0), -1, 1)
        score = (avg_pts * pace_multiplier) + (progression * 0.5)
        d_base[name] = max(1.0, float(score))
        
        # Calculate DNF Probability
        historical_dnfs = row.get("DNFs", 0)
        prob = (historical_dnfs / max_d_dnf * 0.40) + (CIRCUIT_DNF_RATE * 0.20)
        d_dnf[name] = float(np.clip(prob, 0.02, 0.85))
        
        # Final Expected Points
        d_exp[name] = float(max(0.0, (1.0 - d_dnf[name]) * d_base[name]))

    # 2. Evaluate Constructors
    c_exp, c_base, c_costs = {}, {}, {}
    for _, row in df_c.iterrows():
        c_name = str(row["Constructor"]).strip()
        c_costs[c_name] = float(row["Cost"])
        
        # Find drivers signed to this team
        team_drivers = df_d[df_d["Team"].str.strip().str.upper() == c_name.upper()]["Driver"].tolist()
        driver_sum = sum(d_exp.get(d, 0.0) for d in team_drivers)
        
        pitstop_eff = float(np.clip(1.0 + (row.get("FastestPitstops", 0) * 0.01), 0.9, 1.1))
        c_base[c_name] = float(driver_sum)
        c_exp[c_name] = float(max(0.0, driver_sum * 1.45 * pitstop_eff))

    # 3. Export structured JSON for lineup_optimizer & weekend_predictor
    payload = {
        "metadata": {
            "track_name": DEFAULT_TRACK, 
            "is_street": IS_STREET, 
            "circuit_dnf_rate": CIRCUIT_DNF_RATE
        },
        "drivers": {
            k: {
                "cost": d_costs[k], 
                "base_points": d_base[k], 
                "dnf_probability": d_dnf[k], 
                "expected_points": d_exp[k]
            } for k in d_costs
        },
        "constructors": {
            k: {
                "cost": c_costs[k], 
                "base_points": c_base[k], 
                "expected_points": c_exp[k]
            } for k in c_costs
        }
    }
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
        
    log.info("Predictions generated successfully.")
    return d_costs, d_exp, c_costs, c_exp

if __name__ == "__main__":
    driver_costs, driver_expected_points, constructor_costs, constructor_expected_points = run_pipeline()
