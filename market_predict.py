import json
import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("market_predict")

DRIVERS_INPUT      = Path("drivers_ml_ready.csv")
CONSTRUCTORS_INPUT = Path("constructors_ml_ready.csv")
OUTPUT_JSON        = Path("market_predictions.json")
RACE_METADATA_FILE = Path("race_metadata.json")

TRACK_REGISTRY = {
    "monaco":      (True,  0.18),
    "singapore":   (True,  0.16),
    "baku":        (True,  0.15),
    "azerbaij":    (True,  0.15),
    "las vegas":   (True,  0.13),
    "jeddah":      (True,  0.12),
    "saudi":       (True,  0.12),
    "melbourne":   (True,  0.12),
    "australia":   (True,  0.12),
    "miami":       (False, 0.09),
    "canada":      (False, 0.09),
    "silverstone": (False, 0.08),
    "britain":     (False, 0.08),
    "monza":       (False, 0.08),
    "italy":       (False, 0.08),
    "spa":         (False, 0.11),
    "belgium":     (False, 0.11),
    "suzuka":      (False, 0.07),
    "japan":       (False, 0.07),
    "bahrain":     (False, 0.08),
    "shanghai":    (False, 0.08),
    "china":       (False, 0.08),
    "imola":       (False, 0.09),
    "emilia":      (False, 0.09),
    "barcelona":   (False, 0.07),
    "spain":       (False, 0.07),
    "austria":     (False, 0.08),
    "zandvoort":   (False, 0.08),
    "netherlands": (False, 0.08),
    "hungary":     (False, 0.08),
    "austin":      (False, 0.08),
    "mexico":      (False, 0.09),
    "interlagos":  (False, 0.10),
    "brazil":      (False, 0.10),
    "abu dhabi":   (False, 0.07),
    "yas":         (False, 0.07),
    "qatar":       (False, 0.08),
}


def detect_current_track():
    if not RACE_METADATA_FILE.exists():
        log.warning("race_metadata.json not found — run market_harvester.py first. Defaulting to generic track.")
        return "Unknown Grand Prix", False, 0.10

    with open(RACE_METADATA_FILE) as f:
        meta = json.load(f)

    event_name = meta.get("event_name", "Unknown Grand Prix")
    name_lower = event_name.lower()

    for keyword, (is_street, dnf_rate) in TRACK_REGISTRY.items():
        if keyword in name_lower:
            log.info(f"Track auto-detected: '{event_name}'  street={is_street}  dnf_rate={dnf_rate}")
            return event_name, is_street, dnf_rate

    log.warning(f"Track '{event_name}' not matched in registry — using generic defaults.")
    return event_name, False, 0.10


DEFAULT_TRACK, IS_STREET, CIRCUIT_DNF_RATE = detect_current_track()


def load_data():
    if not DRIVERS_INPUT.exists() or not CONSTRUCTORS_INPUT.exists():
        raise FileNotFoundError("Missing ML-ready CSV files. Run market_harvester.py first.")
    df_d = pd.read_csv(DRIVERS_INPUT).fillna(0)
    df_c = pd.read_csv(CONSTRUCTORS_INPUT).fillna(0)
    return df_d, df_c


def compute_pace_multiplier(row):
    current_fp_available = bool(row.get("CurrentFP_Available", False))

    fp1 = float(row.get("FP1_DeltaBest", np.nan))
    fp2 = float(row.get("FP2_DeltaBest", np.nan))
    fp3 = float(row.get("FP3_DeltaBest", np.nan))
    gq  = float(row.get("Q_DeltaBest", np.nan))
    gr  = float(row.get("R_DeltaBest", np.nan))
    best_fp = float(row.get("BestFP_Delta", np.nan))

    if current_fp_available:
        # Heavily weight Race and Qualifying if they are already live for the weekend
        available = [(v, w) for v, w in [(gr, 5.0), (gq, 4.0), (fp3, 3.0), (fp2, 2.0), (fp1, 1.0)] if not np.isnan(v)]
        if available:
            total_w = sum(w for _, w in available)
            delta_factor = sum(v * w for v, w in available) / total_w
        elif not np.isnan(best_fp):
            delta_factor = best_fp
        else:
            delta_factor = 1.2
    else:
        vals = [v for v in [fp1, fp2, fp3, gq, gr] if not np.isnan(v)]
        delta_factor = float(np.mean(vals)) if vals else 1.2

    return float(np.clip(1.3 - (delta_factor * 0.15), 0.5, 1.5))


def run_pipeline():
    log.info(f"Starting predictions for: {DEFAULT_TRACK}  (street={IS_STREET}, dnf_rate={CIRCUIT_DNF_RATE})")
    df_d, df_c = load_data()

    d_exp, d_dnf, d_base, d_costs = {}, {}, {}, {}

    max_d_dnf = max(df_d["DNFs"].max(), 1) if "DNFs" in df_d.columns else 1

    for _, row in df_d.iterrows():
        name = str(row["Driver"]).strip()
        d_costs[name] = float(row["Cost"])

        pace_multiplier = compute_pace_multiplier(row)

        avg_pts    = float(row.get("AvgPoints", 10.0))
        progression = float(np.clip(row.get("FP_Progression", 0.0), -1, 1))
        score      = (avg_pts * pace_multiplier) + (progression * 0.5)
        d_base[name] = max(1.0, float(score))

        historical_dnfs = float(row.get("DNFs", 0))
        prob = (historical_dnfs / max_d_dnf * 0.40) + (CIRCUIT_DNF_RATE * 0.20)
        d_dnf[name] = float(np.clip(prob, 0.02, 0.85))

        d_exp[name] = float(max(0.0, (1.0 - d_dnf[name]) * d_base[name]))

    c_exp, c_base, c_costs = {}, {}, {}
    for _, row in df_c.iterrows():
        c_name = str(row["Constructor"]).strip()
        c_costs[c_name] = float(row["Cost"])

        team_drivers = df_d[df_d["Team"].str.strip().str.upper() == c_name.upper()]["Driver"].tolist()
        driver_sum = sum(d_exp.get(d, 0.0) for d in team_drivers)

        pitstop_eff = float(np.clip(1.0 + (float(row.get("FastestPitstops", 0)) * 0.01), 0.9, 1.1))
        c_base[c_name] = float(driver_sum)
        c_exp[c_name]  = float(max(0.0, driver_sum * 1.45 * pitstop_eff))

    payload = {
        "metadata": {
            "track_name":       DEFAULT_TRACK,
            "is_street":        IS_STREET,
            "circuit_dnf_rate": CIRCUIT_DNF_RATE
        },
        "drivers": {
            k: {
                "cost":             d_costs[k],
                "base_points":      d_base[k],
                "dnf_probability":  d_dnf[k],
                "expected_points":  d_exp[k]
            } for k in d_costs
        },
        "constructors": {
            k: {
                "cost":            c_costs[k],
                "base_points":     c_base[k],
                "expected_points": c_exp[k]
            } for k in c_costs
        }
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    log.info(f"Predictions written to {OUTPUT_JSON}")
    return d_costs, d_exp, c_costs, c_exp


if __name__ == "__main__":
    driver_costs, driver_expected_points, constructor_costs, constructor_expected_points = run_pipeline()
