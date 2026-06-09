"""
market_harvester.py
====================
Enriches F1 fantasy CSVs with live FastF1 telemetry for the current 2026
race weekend, engineers ML-ready features, and writes two output files.

STRICTLY FREE PRACTICE ONLY — NO QUALIFYING OR RACE DATA REQUIRED.
Includes automatic fallback for mid-week execution.
"""

# ── Standard library ────────────────────────────────────────────────────────
import warnings
import logging
from pathlib import Path
from datetime import datetime

# ── Third-party ──────────────────────────────────────────────────────────────
import fastf1
import numpy as np
import pandas as pd
from scipy.stats import linregress

# ── Config ───────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)

# Flush any existing handlers injected by other libraries to prevent duplication
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Re-establish a clean, standard layout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("market_harvester")

# Paths
DRIVERS_CSV        = Path("drivers_fantasy.csv")
CONSTRUCTORS_CSV   = Path("constructors_fantasy.csv")
DRIVERS_OUT        = Path("drivers_ml_ready.csv")
CONSTRUCTORS_OUT   = Path("constructors_ml_ready.csv")
FASTF1_CACHE       = Path(".fastf1_cache")

# Enable FastF1 caching
if not FASTF1_CACHE.exists():
    FASTF1_CACHE.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(FASTF1_CACHE))


# ── Step 1: Secure Event Handling with Automatic Fallback ───────────────────
def get_current_or_last_available_event(year=2026):
    """
    Fetches the current event. If it's mid-week and telemetry isn't live yet,
    it automatically steps back to the most recently completed event.
    """
    try:
        event = fastf1.get_current_event()
        # Test if FP1 data is available yet for this event
        session = fastf1.get_session(year, event["EventName"], "FP1")
        session.load(telemetry=False, weather=False, messages=False)
        log.info(f"✅ Active event detected: {event['EventName']}")
        return event
    except Exception:
        log.warning("⚠️ Current weekend sessions are not yet available on API (Mid-week run).")
        log.info("🔄 Scanning schedule backward for the latest completed race weekend...")
        
        schedule = fastf1.get_event_schedule(year)
        # Filter for official Grands Prix that have passed
        past_events = schedule[
            (schedule["EventFormat"] != "testing") & 
            (schedule["EventDate"] < datetime.now())
        ]
        
        if not past_events.empty:
            last_event = past_events.iloc[-1]
            log.info(f"📋 Falling back to historical data from Round {last_event['RoundNumber']}: {last_event['EventName']}")
            return last_event
        else:
            raise RuntimeError("Could not find any available race weekends for this season.")


# ── Step 2: Load Input Data ──────────────────────────────────────────────────
def load_fantasy_csvs():
    if not DRIVERS_CSV.exists() or not CONSTRUCTORS_CSV.exists():
        raise FileNotFoundError("Missing baseline drivers_fantasy.csv or constructors_fantasy.csv inputs.")
    
    drivers_df = pd.read_csv(DRIVERS_CSV)
    constructors_df = pd.read_csv(CONSTRUCTORS_CSV)
    
    # Normalize naming styles
    drivers_df["Driver"] = drivers_df["Driver"].str.strip().str.upper()
    constructors_df["Constructor"] = constructors_df["Constructor"].str.strip()
    return drivers_df, constructors_df


# ── Step 3: Build Driver Code Lookup Map ──────────────────────────────────────
def build_driver_lookup(year, event_name):
    """Maps 3-letter abbreviations (e.g. 'HAM') to full uppercase names using FP1."""
    log.info("Building driver name translation map...")
    try:
        session = fastf1.get_session(year, event_name, "FP1")
        session.load(telemetry=False, weather=False, messages=False)
        
        lookup = {}
        for _, row in session.results.iterrows():
            code = str(row["Abbreviation"]).strip().upper()
            full_name = f"{row['LastName']}".strip().upper()
            if code and full_name:
                lookup[code] = full_name
        return lookup
    except Exception as e:
        log.error(f"Failed to compile driver lookup mapping: {e}")
        return {}


# ── Step 4: Scrape FP1, FP2, and FP3 Telemetry ───────────────────────────────
def calculate_fp2_deg_slope(session):
    """Calculates tire degradation slope based on consecutive pacing stint runs."""
    deg_data = []
    try:
        laps = session.laps.pick_quicklaps()
        if laps.empty:
            return {}
            
        for driver in laps["Driver"].unique():
            driver_laps = laps[laps["Driver"] == driver].copy()
            if len(driver_laps) < 4:
                continue
                
            # Filter to long run stints
            driver_laps["LapTimeSeconds"] = driver_laps["LapTime"].dt.total_seconds()
            driver_laps["StintLapNumber"] = driver_laps.groupby("Stint").cumcount() + 1
            
            # Linear regression of lap times over stint progression
            slope, _, _, _, _ = linregress(driver_laps["StintLapNumber"], driver_laps["LapTimeSeconds"])
            if not np.isnan(slope):
                deg_data.append({"DriverCode": driver, "FP2_DegSlope": slope})
    except Exception as e:
        log.debug(f"Degradation slope step skipped/failed: {e}")
        
    df_deg = pd.DataFrame(deg_data)
    return df_deg.set_index("DriverCode")["FP2_DegSlope"].to_dict() if not df_deg.empty else {}


def build_telemetry_table(year, event_name):
    """Gathers median lap metrics, delta gaps, and degradation scores across practices."""
    master_telemetry = {}
    
    for session_code in ["FP1", "FP2", "FP3"]:
        log.info(f"Processing session telemetry details for: {session_code}...")
        try:
            session = fastf1.get_session(year, event_name, session_code)
            session.load(telemetry=True, weather=False, messages=False)
            
            laps = session.laps.pick_quicklaps()
            if laps.empty:
                continue
                
            # Session baseline minimum lap time
            best_session_time = laps["LapTime"].min().total_seconds()
            
            # Compute degradation metrics if we are currently looking at FP2
            deg_map = calculate_fp2_deg_slope(session) if session_code == "FP2" else {}
            
            for driver in laps["Driver"].unique():
                d_laps = laps[laps["Driver"] == driver]
                if d_laps.empty:
                    continue
                    
                med_lap = d_laps["LapTime"].median().total_seconds()
                best_lap = d_laps["LapTime"].min().total_seconds()
                delta_best = best_lap - best_session_time
                
                if driver not in master_telemetry:
                    master_telemetry[driver] = {"DriverCode": driver}
                    
                master_telemetry[driver][f"{session_code}_MedianLap"] = med_lap
                master_telemetry[driver][f"{session_code}_DeltaBest"] = delta_best
                
                if session_code == "FP2":
                    master_telemetry[driver]["FP2_DegSlope"] = deg_map.get(driver, np.nan)
                    
        except Exception as e:
            log.warning(f"⚠️ Could not fully process {session_code}: {e}. Filling with column medians.")
            
    # Convert map payload directly into a DataFrame
    telemetry_df = pd.DataFrame(master_telemetry.values())
    
    # Fill gaps safely with default missing flag formats if any practices missed data
    expected_cols = ["FP1_MedianLap", "FP1_DeltaBest", "FP2_MedianLap", "FP2_DeltaBest", "FP3_MedianLap", "FP3_DeltaBest", "FP2_DegSlope"]
    for col in expected_cols:
        if col not in telemetry_df.columns:
            telemetry_df[col] = np.nan
            
    return telemetry_df


# ── Step 5: Merge FastF1 Scrapes into Fantasy Arrays ─────────────────────────
def find_driver_match(lookup_name, driver_lookup, fantasy_name):
    """Robust lookup helper to align abbreviated codes to real name formats."""
    f_clean = str(fantasy_name).upper().strip()
    for code, full_name in driver_lookup.items():
        if code == lookup_name or full_name in f_clean or f_clean in full_name:
            return code
    return None


def merge_drivers(drivers_df, telemetry, driver_lookup):
    log.info("Blending scraped telemetry metrics into driver records...")
    
    # Map the accurate matching lookup code onto the original dataset row indices
    drivers_df["DriverCode"] = drivers_df["Driver"].apply(
        lambda name: next((c for c, f in driver_lookup.items() if f in name or name in f), None)
    )
    
    # Fallback merge matching using raw dictionary mapping strings
    merged = pd.merge(drivers_df, telemetry, on="DriverCode", how="left")
    
    # Ensure any missing fields are filled with column medians to prevent XGBoost training failure
    for col in ["FP1_DeltaBest", "FP2_DeltaBest", "FP3_DeltaBest", "FP2_DegSlope"]:
        if col in merged.columns and merged[col].isna().any():
            merged[col] = merged[col].fillna(merged[col].median())
            
    return merged


def merge_constructors(constructors_df, drivers_enriched):
    log.info("Compiling constructor records from grouped team telemetry attributes...")
    
    # Simple normalization mapping array lookup
    team_mapping = {
        "RACING BULLS": "RACINGBULLS",
        "SAUBER": "KICK SAUBER",
        "HAAS F1 TEAM": "HAAS",
        "RED BULL RACING": "REDBULL"
    }
    
    drivers_enriched["CleanTeam"] = drivers_enriched["Team"].str.upper().str.strip().replace(team_mapping)
    constructors_df["CleanTeam"] = constructors_df["Constructor"].str.upper().str.strip().replace(team_mapping)
    
    # Aggregate dynamic average statistics for team compositions
    team_stats = drivers_enriched.groupby("CleanTeam").agg({
        "FP2_DeltaBest": "mean",
        "FP2_DegSlope": "mean",
        "Overtakes": "sum"
    }).rename(columns={
        "FP2_DeltaBest": "Team_Avg_FP2_Delta",
        "FP2_DegSlope": "Team_Avg_FP2_Slope"
    })
    
    merged = pd.merge(constructors_df, team_stats, on="CleanTeam", how="left").drop(columns=["CleanTeam"])
    return merged


# ── Step 6: Feature Engineering Pipeline ──────────────────────────────────────
def engineer_features(drivers, constructors):
    log.info("Executing mathematical feature engineering processes...")
    
    # Driver Metrics
    drivers["Overtake_Efficiency"] = drivers["Overtakes"] / (drivers["AvgPoints"] + 1e-5)
    drivers["Hype_Ratio"] = drivers["SelectionPct"] / (drivers["Cost"] + 1e-5)
    drivers["FP_Progression"] = drivers["FP1_DeltaBest"] - drivers["FP3_DeltaBest"]
    
    # Constructor Metrics
    constructors["Hype_Ratio"] = constructors["SelectionPct"] / (constructors["Cost"] + 1e-5)
    
    return drivers, constructors


def save_outputs(drivers, constructors):
    log.info(f"Writing outputs out to standard dataset targets: {DRIVERS_OUT} | {CONSTRUCTORS_OUT}")
    drivers.to_csv(DRIVERS_OUT, index=False)
    constructors.to_csv(CONSTRUCTORS_OUT, index=False)


# ── Main Runtime Execution ───────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 60)
    log.info(" RUNNING PRE-QUALIFYING TELEMETRY EXTRACTION ENGINE")
    log.info("=" * 60)

    # 1. Load Data
    drivers_baseline, constructors_baseline = load_fantasy_csvs()

    # 2. Get Safe Target Race Weekend (With Auto-Fallback)
    current_event = get_current_or_last_available_event(year=2026)
    target_race = current_event["EventName"]

    # 3. Build Mapping Context
    lookup_map = build_driver_lookup(2026, target_race)

    # 4. Fetch Pure FP Telemetry
    telemetry_table = build_telemetry_table(2026, target_race)

    # 5. Pipeline Merging Processing Steps
    drivers_enriched = merge_drivers(drivers_baseline, telemetry_table, lookup_map)
    constructors_enriched = merge_constructors(constructors_baseline, drivers_enriched)

    # 6. Feature Engineering
    drivers_final, constructors_final = engineer_features(drivers_enriched, constructors_enriched)

    # 7. Save to ML-Ready Files
    save_outputs(drivers_final, constructors_final)

    log.info("=" * 60)
    log.info("  Done. Pure Free Practice ML-ready files updated.")
    log.info("=" * 60)
