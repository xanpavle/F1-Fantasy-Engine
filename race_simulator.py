"""
race_simulator.py  —  F1 Monte Carlo Race Simulator (10,000 Iterations)
═══════════════════════════════════════════════════════════════════════════════
Simulates a Grand Prix lap-by-lap using stochastic probability matrices.
Takes predicted grid positions, DNF probabilities, and overtake efficiencies
to output true percentage-based outcomes for the weekend.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")

# ── Configuration ──────────────────────────────────────────────────────────────
N_SIMULATIONS = 10000
TOTAL_LAPS = 70  # Can dynamically adjust based on track
LAP1_SC_RISK_MULTIPLIER = 2.5
LAP1_DNF_RISK_MULTIPLIER = 3.0

# Used to convert statistical average position difference into seconds on track
GAP_MULTIPLIER_SECONDS = 4.25 

PREDICTIONS_JSON = Path("market_predictions.json")
DRIVERS_CSV = Path("drivers_ml_ready.csv")

def load_simulation_data():
    """Loads baseline data to seed the starting grid and driver stats."""
    if not DRIVERS_CSV.exists() or not PREDICTIONS_JSON.exists():
        raise FileNotFoundError("Missing required files. Run market_predict.py first.")

    with open(PREDICTIONS_JSON, "r") as fh:
        predict_data = json.load(fh)

    df_drivers = pd.read_csv(DRIVERS_CSV)
    
    # Extract metadata
    meta = predict_data.get("metadata", {})
    track_name = meta.get("track_name", "Unknown Track")
    circuit_dnf = meta.get("circuit_dnf_rate", 0.12)
    
    # Build Driver Profiles
    drivers_json = predict_data.get("drivers", {})
    
    driver_stats = []
    for d_name, d_data in drivers_json.items():
        csv_row = df_drivers[df_drivers["Driver"].str.strip() == d_name]
        
        overtake_eff = csv_row["Overtake_Efficiency"].values[0] if not csv_row.empty and "Overtake_Efficiency" in csv_row.columns else 0.5
        pace_score = d_data.get("base_points", 1.0)
        
        driver_stats.append({
            "driver": d_name,
            "pace": pace_score,
            "dnf_prob_per_race": d_data.get("dnf_probability", 0.05),
            "overtake_eff": overtake_eff
        })
        
    # Sort by pace to simulate Starting Grid (Qualifying Order)
    driver_stats.sort(key=lambda x: x["pace"], reverse=True)
    
    return track_name, circuit_dnf, driver_stats

def run_monte_carlo(track_name, circuit_dnf, drivers):
    """Executes the lap-by-lap simulation N_SIMULATIONS times."""
    
    num_drivers = len(drivers)
    driver_names = [d["driver"] for d in drivers]
    
    # Pre-calculate lap-by-lap probabilities
    base_lap_dnf = np.array([1 - (1 - d["dnf_prob_per_race"])**(1/TOTAL_LAPS) for d in drivers])
    pace_array = np.array([d["pace"] for d in drivers])
    overtake_array = np.array([d["overtake_eff"] for d in drivers])
    
    # Circuit specific constants
    lap_sc_prob = circuit_dnf / TOTAL_LAPS 
    track_overtake_difficulty = 0.8 if "Monaco" in track_name or "Singapore" in track_name else 0.4
    
    # Result Trackers
    finish_positions = {name: [] for name in driver_names}
    sc_in_first_10 = 0
    
    logging.info(f"Starting {N_SIMULATIONS:,} Monte Carlo lap-by-lap simulations for {track_name}...")
    start_time = time.time()

    for sim in range(N_SIMULATIONS):
        current_grid = np.arange(num_drivers)
        active_status = np.ones(num_drivers, dtype=bool)
        sim_sc_first_10 = False

        for lap in range(1, TOTAL_LAPS + 1):
            # 1. SC Roll
            sc_chance = lap_sc_prob * (LAP1_SC_RISK_MULTIPLIER if lap == 1 else 1.0)
            if np.random.rand() < sc_chance:
                if lap <= 10:
                    sim_sc_first_10 = True
                continue 
                
            # 2. DNF Rolls
            dnf_multiplier = LAP1_DNF_RISK_MULTIPLIER if lap == 1 else 1.0
            dnf_rolls = np.random.rand(num_drivers)
            crashes = (dnf_rolls < (base_lap_dnf * dnf_multiplier)) & active_status
            active_status[crashes] = False
            
            # 3. Overtake Phase
            if lap > 1: 
                for pos in range(num_drivers - 1, 0, -1):
                    car_behind_idx = current_grid[pos]
                    car_ahead_idx = current_grid[pos - 1]
                    
                    if not active_status[car_behind_idx] or not active_status[car_ahead_idx]:
                        continue
                        
                    pace_delta = pace_array[car_behind_idx] - pace_array[car_ahead_idx]
                    eff = overtake_array[car_behind_idx]
                    
                    ovt_prob = max(0.0, (pace_delta * 0.05) + (eff * 0.1) - (track_overtake_difficulty * 0.15))
                    
                    if np.random.rand() < ovt_prob:
                        current_grid[pos], current_grid[pos - 1] = current_grid[pos - 1], current_grid[pos]

        # Record classification
        active_cars = [current_grid[i] for i in range(num_drivers) if active_status[current_grid[i]]]
        dnf_cars = [current_grid[i] for i in range(num_drivers) if not active_status[current_grid[i]]]
        
        final_classification = active_cars + dnf_cars
        
        for final_pos, driver_idx in enumerate(final_classification):
            name = driver_names[driver_idx]
            finish_positions[name].append(final_pos + 1)
            
        if sim_sc_first_10:
            sc_in_first_10 += 1

    elapsed = time.time() - start_time
    logging.info(f"Simulations complete in {elapsed:.2f} seconds.")
    
    return finish_positions, sc_in_first_10

def format_output(finish_positions, sc_in_first_10, drivers):
    """Calculates probabilities and prints the two requested tables."""
    
    print("\n" + "═" * 80)
    print(f"  MONTE CARLO RACE SIMULATION RESULTS ({N_SIMULATIONS:,} Iterations)")
    print("═" * 80)
    
    # Process Results
    results_summary = []
    for d in drivers:
        name = d["driver"]
        positions = np.array(finish_positions[name])
        
        avg_pos = np.mean(positions)
        win_pct = np.mean(positions == 1) * 100
        podium_pct = np.mean(positions <= 3) * 100
        top10_pct = np.mean(positions <= 10) * 100
        dnf_pct = np.mean(positions > 15) * 100 # Proxy for DNF zone
        
        results_summary.append({
            "name": name,
            "avg_pos": avg_pos,
            "win": win_pct,
            "podium": podium_pct,
            "top10": top10_pct,
            "dnf": dnf_pct
        })
        
    # Sort by Average Finishing Position (best expected outcome)
    results_summary.sort(key=lambda x: x["avg_pos"])

    # ───────────────────────────────────────────────────────────────────────────
    # TABLE 1: PROBABILITY MATRIX
    # ───────────────────────────────────────────────────────────────────────────
    print("\n  📊 TABLE 1: OUTCOME PROBABILITIES (ALL 22 DRIVERS)")
    print("  " + "─" * 76)
    print(f"  {'Rank':<4} | {'Driver':<14} | {'Avg Pos':>7} | {'Win %':>8} | {'Podium %':>8} | {'Top 10 %':>8} | {'DNF %':>8}")
    print("  " + "─" * 76)
    
    for i, row in enumerate(results_summary):
        print(f"  {'P'+str(i+1):<4} | {row['name']:<14} | {row['avg_pos']:>7.2f} | "
              f"{row['win']:>7.1f}% | {row['podium']:>7.1f}% | {row['top10']:>7.1f}% | {row['dnf']:>7.1f}%")


    # ───────────────────────────────────────────────────────────────────────────
    # TABLE 2: EXPECTED RACE CLASSIFICATION & TIME GAPS
    # ───────────────────────────────────────────────────────────────────────────
    print("\n\n  ⏱️  TABLE 2: EXPECTED FINAL CLASSIFICATION & TIME GAPS")
    print("  (Time gaps calculated directly from the statistical simulation spread)")
    print("  " + "─" * 76)
    print(f"  {'Pos':<4} | {'Driver':<14} | {'Avg Pos':>7} | {'Interval':>12} | {'Gap to Leader':>15}")
    print("  " + "─" * 76)
    
    leader_avg_pos = results_summary[0]["avg_pos"]
    
    for i, row in enumerate(results_summary):
        pos = i + 1
        avg_pos = row["avg_pos"]
        
        if pos == 1:
            interval_str = "Leader"
            gap_leader_str = "-"
        else:
            prev_avg_pos = results_summary[i-1]["avg_pos"]
            
            # Convert Avg Pos difference into a track time gap
            interval_sec = (avg_pos - prev_avg_pos) * GAP_MULTIPLIER_SECONDS
            leader_gap_sec = (avg_pos - leader_avg_pos) * GAP_MULTIPLIER_SECONDS
            
            # If DNF probability is dominant, label them as DNF
            if row["dnf"] > 50.0:
                interval_str = "DNF"
                gap_leader_str = "DNF"
            else:
                interval_str = f"+{interval_sec:.3f}s"
                gap_leader_str = f"+{leader_gap_sec:.3f}s"
                
        print(f"  {'P'+str(pos):<4} | {row['name']:<14} | {avg_pos:>7.2f} | {interval_str:>12} | {gap_leader_str:>15}")

    # ───────────────────────────────────────────────────────────────────────────
    # VARIANT QUESTION SUMMARY
    # ───────────────────────────────────────────────────────────────────────────
    print("\n\n  🚨 EVENT PROBABILITIES")
    print("  " + "─" * 76)
    sc_prob = (sc_in_first_10 / N_SIMULATIONS) * 100
    print(f"  Chance of a Safety Car/VSC between Laps 1-10: {sc_prob:.1f}%")
    print("\n" + "═" * 80 + "\n")

if __name__ == "__main__":
    try:
        t_name, c_dnf, driver_grid = load_simulation_data()
        finishes, sc_early = run_monte_carlo(t_name, c_dnf, driver_grid)
        format_output(finishes, sc_early, driver_grid)
    except Exception as e:
        logging.error(f"Simulation failed: {e}")
