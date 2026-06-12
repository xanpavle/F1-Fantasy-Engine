import json
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
RACE_METADATA_OUT  = Path("race_metadata.json")
FASTF1_CACHE       = Path(".fastf1_cache")

HISTORICAL_WEIGHT  = 0.2
CURRENT_FP_WEIGHT  = 3.0

if not FASTF1_CACHE.exists():
    FASTF1_CACHE.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(FASTF1_CACHE))
fastf1.set_log_level('ERROR')  # Silences scary internal tracebacks for upcoming sessions


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
        if laps.empty:
            return deltas
        valid_laps = laps.pick_quicklaps() if not laps.pick_quicklaps().empty else laps
        fastest_overall = valid_laps['LapTime'].min()
        if pd.isna(fastest_overall):
            return deltas
        for driver in valid_laps['Driver'].unique():
            driver_laps = valid_laps[valid_laps['Driver'] == driver]
            if not driver_laps.empty:
                best_lap = driver_laps['LapTime'].min()
                if not pd.isna(best_lap):
                    deltas[driver] = (best_lap - fastest_overall).total_seconds()
    except Exception as e:
        log.debug(f"Session {session_code} round {round_num} not available: {e}")
    return deltas


def harvest_deg_slope(year, round_num, session_code="FP2"):
    slopes = {}
    try:
        session = fastf1.get_session(year, round_num, session_code)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps
        if laps.empty:
            return slopes
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


def get_current_round_and_event(schedule):
    now = pd.Timestamp.utcnow().tz_localize(None)
    event_dates = pd.to_datetime(schedule['EventDate']).dt.tz_localize(None)
    completed = schedule[event_dates < now]
    upcoming = schedule[event_dates >= now]

    if not upcoming.empty:
        next_event = upcoming.iloc[0]
        days_until = (pd.to_datetime(next_event['EventDate']).tz_localize(None) - now).days
        if days_until <= 4:
            return int(next_event['RoundNumber']), next_event
        if not completed.empty:
            last = completed.iloc[-1]
            return int(last['RoundNumber']), last
        return int(next_event['RoundNumber'], next_event)
    else:
        last = completed.iloc[-1]
        return int(last['RoundNumber']), last


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
        schedule = schedule[schedule['RoundNumber'] > 0].reset_index(drop=True)
        current_round, current_event = get_current_round_and_event(schedule)
        event_name = str(current_event.get('EventName', current_event.get('OfficialEventName', 'Unknown Grand Prix')))
    except Exception as e:
        log.warning(f"Schedule detection failed ({e}), defaulting to round 1.")
        current_round = 1
        event_name = "Unknown Grand Prix"

    log.info(f"Current active Round: {current_round} — {event_name}")

    race_metadata = {"round": current_round, "event_name": event_name}
    with open(RACE_METADATA_OUT, "w") as f:
        json.dump(race_metadata, f, indent=2)
    log.info(f"Race metadata saved → {RACE_METADATA_OUT}")

    driver_mapping = get_driver_code_mapping()
    telemetry_history = {
        code: {"FP1": [], "FP2": [], "FP3": [], "Q": [], "R": [], "Slope": [], "Weights": []}
        for code in driver_mapping.values()
    }

    current_fp_available = False

    for r in range(1, current_round + 1):
        is_current = (r == current_round)
        weight = CURRENT_FP_WEIGHT if is_current else HISTORICAL_WEIGHT

        log.info(f"Scanning Round {r}/{current_round} (Weight: {weight})...")

        fp1_d = harvest_session_deltas(2026, r, "FP1")
        fp2_d = harvest_session_deltas(2026, r, "FP2")
        fp3_d = harvest_session_deltas(2026, r, "FP3")
        q_d   = harvest_session_deltas(2026, r, "Q")
        r_d   = harvest_session_deltas(2026, r, "R")
        slope_d = harvest_deg_slope(2026, r, "FP2")

        if is_current:
            if fp1_d or fp2_d or fp3_d or q_d or r_d:
                current_fp_available = True
                available = [s for s, d in [("FP1", fp1_d), ("FP2", fp2_d), ("FP3", fp3_d), ("Q", q_d), ("R", r_d)] if d]
                log.info(f"  Current weekend sessions available: {', '.join(available)}")
            else:
                log.info("  Current weekend sessions not started yet — using historical only.")
                continue

        for driver_code in telemetry_history.keys():
            f1 = fp1_d.get(driver_code, np.nan)
            f2 = fp2_d.get(driver_code, np.nan)
            f3 = fp3_d.get(driver_code, np.nan)
            gq = q_d.get(driver_code, np.nan)
            gr = r_d.get(driver_code, np.nan)
            sl = slope_d.get(driver_code, np.nan)

            if not (np.isnan(f1) and np.isnan(f2) and np.isnan(f3) and np.isnan(gq) and np.isnan(gr)):
                telemetry_history[driver_code]["FP1"].append(f1)
                telemetry_history[driver_code]["FP2"].append(f2)
                telemetry_history[driver_code]["FP3"].append(f3)
                telemetry_history[driver_code]["Q"].append(gq)
                telemetry_history[driver_code]["R"].append(gr)
                telemetry_history[driver_code]["Slope"].append(sl if not np.isnan(sl) else 0.01)
                telemetry_history[driver_code]["Weights"].append(weight)

    final_telemetry = {}
    for code, data in telemetry_history.items():
        w_array = np.array(data["Weights"])
        if len(w_array) > 0 and w_array.sum() > 0:
            fp1_vals = np.array(data["FP1"], dtype=float)
            fp2_vals = np.array(data["FP2"], dtype=float)
            fp3_vals = np.array(data["FP3"], dtype=float)
            q_vals   = np.array(data["Q"], dtype=float)
            r_vals   = np.array(data["R"], dtype=float)
            sl_vals  = np.array(data["Slope"], dtype=float)

            def weighted_nanmean(vals, weights):
                mask = ~np.isnan(vals)
                if mask.sum() == 0:
                    return np.nan
                return float(np.nansum(vals[mask] * weights[mask]) / weights[mask].sum())

            fp1_avg   = weighted_nanmean(fp1_vals, w_array)
            fp2_avg   = weighted_nanmean(fp2_vals, w_array)
            fp3_avg   = weighted_nanmean(fp3_vals, w_array)
            q_avg     = weighted_nanmean(q_vals, w_array)
            r_avg     = weighted_nanmean(r_vals, w_array)
            slope_avg = weighted_nanmean(sl_vals,  w_array)
        else:
            fp1_avg, fp2_avg, fp3_avg, q_avg, r_avg, slope_avg = 1.2, 1.2, 1.2, 1.2, 1.2, 0.01

        available_fps = [v for v in [r_avg, q_avg, fp3_avg, fp2_avg, fp1_avg] if v is not None and not np.isnan(v)]
        best_fp = available_fps[0] if available_fps else 1.2

        if not np.isnan(fp1_avg) and not np.isnan(q_avg):
            raw_progression = float(fp1_avg - q_avg)
        elif not np.isnan(fp1_avg) and not np.isnan(fp3_avg):
            raw_progression = float(fp1_avg - fp3_avg)
        elif not np.isnan(fp1_avg) and not np.isnan(fp2_avg):
            raw_progression = float(fp1_avg - fp2_avg)
        else:
            raw_progression = 0.0
        clamped_progression = float(np.clip(raw_progression * 0.1, -0.2, 0.2))

        final_telemetry[code] = {
            "FP1_DeltaBest":  fp1_avg  if (fp1_avg  is not None and not np.isnan(fp1_avg))  else 1.0,
            "FP2_DeltaBest":  fp2_avg  if (fp2_avg  is not None and not np.isnan(fp2_avg))  else 1.0,
            "FP3_DeltaBest":  fp3_avg  if (fp3_avg  is not None and not np.isnan(fp3_avg))  else 1.0,
            "Q_DeltaBest":    q_avg    if (q_avg    is not None and not np.isnan(q_avg))    else 1.0,
            "R_DeltaBest":    r_avg    if (r_avg    is not None and not np.isnan(r_avg))    else 1.0,
            "BestFP_Delta":   best_fp,
            "FP2_DegSlope":   slope_avg if (slope_avg is not None and not np.isnan(slope_avg)) else 0.01,
            "FP_Progression": clamped_progression,
            "CurrentFP_Available": current_fp_available,
        }

    enriched_drivers = []
    for _, row in drivers_df.iterrows():
        d_name = str(row["Driver"]).upper()
        d_code = driver_mapping.get(d_name, d_name[:3])

        tel = final_telemetry.get(d_code, {
            "FP1_DeltaBest": 1.0, "FP2_DeltaBest": 1.0, "FP3_DeltaBest": 1.0,
            "Q_DeltaBest": 1.0, "R_DeltaBest": 1.0,
            "BestFP_Delta": 1.0, "FP2_DegSlope": 0.01, "FP_Progression": 0.0,
            "CurrentFP_Available": False
        })

        row_dict = row.to_dict()
        row_dict.update(tel)
        row_dict["Overtake_Efficiency"] = float(row.get("Overtakes", 0) / (row.get("TotalPoints", 1) + 1e-5))
        row_dict["Hype_Ratio"]          = float(row.get("SelectionPct", 0) / (row.get("Cost", 1) + 1e-5))
        row_dict["DriverCode"]          = d_code
        enriched_drivers.append(row_dict)

    drivers_out_df = pd.DataFrame(enriched_drivers).fillna(0)

    enriched_constructors = []
    for _, row in constructors_df.iterrows():
        c_name = row["Constructor"]
        team_drivers = drivers_out_df[drivers_out_df["Team"].str.upper() == str(c_name).upper()]

        row_dict = row.to_dict()
        if not team_drivers.empty:
            row_dict["FP1_DeltaBest"]  = float(team_drivers["FP1_DeltaBest"].mean())
            row_dict["FP2_DeltaBest"]  = float(team_drivers["FP2_DeltaBest"].mean())
            row_dict["FP3_DeltaBest"]  = float(team_drivers["FP3_DeltaBest"].mean())
            row_dict["Q_DeltaBest"]    = float(team_drivers["Q_DeltaBest"].mean())
            row_dict["R_DeltaBest"]    = float(team_drivers["R_DeltaBest"].mean())
            row_dict["BestFP_Delta"]   = float(team_drivers["BestFP_Delta"].mean())
            row_dict["FP2_DegSlope"]   = float(team_drivers["FP2_DegSlope"].mean())
            row_dict["FP_Progression"] = float(team_drivers["FP_Progression"].mean())
        else:
            row_dict["FP1_DeltaBest"]  = 1.0
            row_dict["FP2_DeltaBest"]  = 1.0
            row_dict["FP3_DeltaBest"]  = 1.0
            row_dict["Q_DeltaBest"]    = 1.0
            row_dict["R_DeltaBest"]    = 1.0
            row_dict["BestFP_Delta"]   = 1.0
            row_dict["FP2_DegSlope"]   = 0.01
            row_dict["FP_Progression"] = 0.0
        enriched_constructors.append(row_dict)

    constructors_out_df = pd.DataFrame(enriched_constructors).fillna(0)

    log.info(f"Writing: {DRIVERS_OUT} ({len(drivers_out_df)} rows) | {CONSTRUCTORS_OUT} ({len(constructors_out_df)} rows)")
    drivers_out_df.to_csv(DRIVERS_OUT, index=False)
    constructors_out_df.to_csv(CONSTRUCTORS_OUT, index=False)


if __name__ == "__main__":
    main()
