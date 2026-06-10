"""
market_harvester.py
====================
Enriches F1 fantasy CSVs with historical 2026 data + live telemetry.
Weights the current race highest and aggregates everything into EXACTLY 22 rows
to prevent duplicates. Progression stats are clamped to prevent qualifying anomalies.
"""

import warnings
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import fastf1

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("market_harvester")

DRIVERS_CSV        = Path("drivers_fantasy.csv")
CONSTRUCTORS_CSV   = Path("constructors_fantasy.csv")
DRIVERS_OUT        = Path("drivers_ml_ready.csv")
CONSTRUCTORS_OUT   = Path("constructors_ml_ready.csv")
FASTF1_CACHE       = Path(".fastf1_cache")

if not FASTF1_CACHE.exists():
    FASTF1_CACHE.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(FASTF1_CACHE))

def get_driver_code_mapping():
    return {
        "VERSTAPPEN": "VER", "HAMILTON": "HAM", "RUSSELL": "RUS", 
        "LECLERC": "LEC", "NORRIS": "NOR", "PIASTRI": "PIA", 
        "ANTONELLI": "ANT", "SAINZ": "SAI", "ALBON": "ALB", 
        "GASLY": "GAS", "OCON": "OCO", "HULKENBERG": "HUL", 
        "BOTTAS": "BOT", "PEREZ": "PER", "LAWSON": "LAW", 
        "TSUNODA": "TSU", "STROLL": "STR", "ALONSO": "ALO", 
        "MAGNUSSEN": "MAG", "BEARMAN": "BEA", "COLAPINTO": "COL",
        "HADJAR": "HAD", "BORTOLETO": "BOR", "LINDBLAD": "LIN"
    }

def harvest_session_deltas(year, round_num, session_code):
    deltas = {}
    try:
        session = fastf1.get_session(year, round_num, session_code)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps
        if laps.empty: return deltas
            
        valid_laps = laps.pick_quicklaps() if not laps.pick_quicklaps().empty else laps
        fastest_overall = valid_laps['LapTime'].min()
        if pd.isna(fastest_overall): return deltas
            
        for driver in valid_laps['Driver'].unique():
            driver_laps = valid_laps[valid_laps['Driver'] == driver]
            if not driver_laps.empty:
                best_lap = driver_laps['LapTime'].min()
                if not pd.isna(best_lap):
                    deltas[driver] = (best_lap - fastest_overall).total_seconds()
    except Exception as e:
        log.debug(f"Session {session_code} not available: {e}")
    return deltas

def harvest_deg_slope(year, round_num):
    slopes = {}
    try:
        session = fastf1.get_session(year, round_num, 'FP2')
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps
        if laps.empty: return slopes
            
        for driver in laps['Driver'].unique():
            d_laps = laps[(laps['Driver'] == driver) & (laps['LapNumber'] > 2)].copy()
            if len(d_laps) > 4:
                d_laps['LapTimeSec'] = d_laps['LapTime'].dt.total_seconds()
                d_laps = d_laps.dropna(subset=['LapTimeSec'])
                if len(d_laps) > 4:
                    slope, _ = np.polyfit(d_laps['LapNumber'], d_laps['LapTimeSec'], 1)
                    slopes[driver] = max(0.0, float(slope))
    except Exception:
        pass
    return slopes

def main():
    log.info("========================================================")
    log.info(" RUNNING MULTI-ROUND TELEMETRY HARVESTER Engine")
    log.info("========================================================")

    if not DRIVERS_CSV.exists() or not CONSTRUCTORS_CSV.exists():
        log.error("Baseline CSV files missing!")
        return

    drivers_df = pd.read_csv(DRIVERS_CSV)
    constructors_df = pd.read_csv(CONSTRUCTORS_CSV)

    try:
        schedule = fastf1.get_event_schedule(2026)
        now = pd.Timestamp.now(tz='UTC')
        future_events = schedule[schedule['EventDate'] >= now]
        current_round = future_events.iloc[0]['RoundNumber'] if not future_events.empty else schedule['RoundNumber'].max()
    except Exception:
        current_round = 5
    
    log.info(f"Current active Round: {current_round}")

    driver_mapping = get_driver_code_mapping()
    telemetry_history = {code: {"FP1": [], "FP2": [], "FP3": [], "Slope": [], "Weights": []} for code in driver_mapping.values()}

    for r in range(1, current_round + 1):
        is_current = (r == current_round)
        weight = 1.0 if is_current else 0.2
        
        log.info(f"🔄 Scanning Round {r}/{current_round} (Weight: {weight})...")
        
        fp1_d = harvest_session_deltas(2026, r, "FP1")
        fp2_d = harvest_session_deltas(2026, r, "FP2")
        fp3_d = harvest_session_deltas(2026, r, "FP3")
        slope_d = harvest_deg_slope(2026, r)
        
        if is_current and not fp1_d and not fp2_d and not fp3_d:
            log.info("  ↳ Live weekend sessions not started yet. Utilizing historical.")
            continue

        for driver_code in telemetry_history.keys():
            f1, f2, f3, sl = fp1_d.get(driver_code, np.nan), fp2_d.get(driver_code, np.nan), fp3_d.get(driver_code, np.nan), slope_d.get(driver_code, np.nan)
            
            if not (np.isnan(f1) and np.isnan(f2) and np.isnan(f3)):
                telemetry_history[driver_code]["FP1"].append(f1)
                telemetry_history[driver_code]["FP2"].append(f2)
                telemetry_history[driver_code]["FP3"].append(f3)
                telemetry_history[driver_code]["Slope"].append(sl if not np.isnan(sl) else 0.01)
                telemetry_history[driver_code]["Weights"].append(weight)

    final_telemetry = {}
    for code, data in telemetry_history.items():
        w_array = np.array(data["Weights"])
        if len(w_array) > 0 and w_array.sum() > 0:
            fp1_avg = float(np.nansum(np.array(data["FP1"]) * w_array) / w_array.sum())
            fp2_avg = float(np.nansum(np.array(data["FP2"]) * w_array) / w_array.sum())
            fp3_avg = float(np.nansum(np.array(data["FP3"]) * w_array) / w_array.sum())
            slope_avg = float(np.nansum(np.array(data["Slope"]) * w_array) / w_array.sum())
        else:
            fp1_avg, fp2_avg, fp3_avg, slope_avg = 1.2, 1.2, 1.2, 0.01

        # FIX: Clamp the progression feature so it doesn't break Qualifying predictions
        raw_progression = float(fp1_avg - fp3_avg) if not np.isnan(fp1_avg - fp3_avg) else 0.0
        clamped_progression = np.clip(raw_progression * 0.1, -0.2, 0.2)

        final_telemetry[code] = {
            "FP1_DeltaBest": fp1_avg if not np.isnan(fp1_avg) else 1.0,
            "FP2_DeltaBest": fp2_avg if not np.isnan(fp2_avg) else 1.0,
            "FP3_DeltaBest": fp3_avg if not np.isnan(fp3_avg) else 1.0,
            "FP2_DegSlope": slope_avg if not np.isnan(slope_avg) else 0.01,
            "FP_Progression": float(clamped_progression)
        }

    enriched_drivers = []
    for _, row in drivers_df.iterrows():
        d_name = str(row["Driver"]).upper()
        d_code = driver_mapping.get(d_name, d_name[:3])
        
        tel = final_telemetry.get(d_code, {"FP1_DeltaBest": 1.0, "FP2_DeltaBest": 1.0, "FP3_DeltaBest": 1.0, "FP2_DegSlope": 0.01, "FP_Progression": 0.0})
        
        row_dict = row.to_dict()
        row_dict.update(tel)
        
        row_dict["Overtake_Efficiency"] = float(row.get("Overtakes", 0) / (row.get("TotalPoints", 1) + 1e-5))
        row_dict["Hype_Ratio"] = float(row.get("SelectionPct", 0) / (row.get("Cost", 1) + 1e-5))
        row_dict["DriverCode"] = d_code
        
        enriched_drivers.append(row_dict)

    drivers_out_df = pd.DataFrame(enriched_drivers).fillna(0)

    enriched_constructors = []
    for _, row in constructors_df.iterrows():
        c_name = row["Constructor"]
        team_drivers = drivers_out_df[drivers_out_df["Team"].str.upper() == str(c_name).upper()]
        
        row_dict = row.to_dict()
        if not team_drivers.empty:
            row_dict["FP1_DeltaBest"] = float(team_drivers["FP1_DeltaBest"].mean())
            row_dict["FP2_DeltaBest"] = float(team_drivers["FP2_DeltaBest"].mean())
            row_dict["FP3_DeltaBest"] = float(team_drivers["FP3_DeltaBest"].mean())
            row_dict["FP2_DegSlope"] = float(team_drivers["FP2_DegSlope"].mean())
            row_dict["FP_Progression"] = float(team_drivers["FP_Progression"].mean())
        else:
            row_dict["FP1_DeltaBest"], row_dict["FP2_DeltaBest"], row_dict["FP3_DeltaBest"] = 1.0, 1.0, 1.0
            row_dict["FP2_DegSlope"], row_dict["FP_Progression"] = 0.01, 0.0

        enriched_constructors.append(row_dict)

    constructors_out_df = pd.DataFrame(enriched_constructors).fillna(0)

    log.info(f"Writing outputs out: {DRIVERS_OUT} ({len(drivers_out_df)} rows) | {CONSTRUCTORS_OUT} ({len(constructors_out_df)} rows)")
    drivers_out_df.to_csv(DRIVERS_OUT, index=False)
    constructors_out_df.to_csv(CONSTRUCTORS_OUT, index=False)

if __name__ == "__main__":
    main()
