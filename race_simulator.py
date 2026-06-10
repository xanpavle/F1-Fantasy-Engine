"""
race_simulator.py  —  F1 Monte Carlo Race Simulator (10,000 Iterations)
═══════════════════════════════════════════════════════════════════════════════
v4 — The Pro-Strategist & Environmental Expansion Engine
  Includes all 10 V3 Elite Optimizations + 5 Level 4 Pro Upgrades:
  1. Dynamic Tyre Degradation               11. Dynamic Weather Engine
  2. Non-Linear Traffic/DRS Dampening       12. Tactical Team Orders Logic
  3. Aggression-Based Incident Spikes       13. Track Evolution (Rubbering-In)
  4. Track Positioning DNF Scaling          14. "Clutch" Momentum Streak
  5. Constructor Pit-Stop & Blunders        15. Mechanical Reliability Wear & Tear
  6. Safety Car Grid Compact Engine
  7. Fuel-Weight Burn-Off Acceleration
  8. Real-Time Overtake Efficiency Battle
  9. Blue Flag Backmarker Eraser
  10. Post-Crash Survival Promotion
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Core Configuration ────────────────────────────────────────────────────────
N_SIMULATIONS            = 10000
TOTAL_LAPS               = 78
LAP1_SC_RISK_MULTIPLIER  = 2.5
LAP1_DNF_RISK_MULTIPLIER = 3.0
GAP_MULTIPLIER_SECONDS   = 4.25

PREDICTIONS_JSON = Path("market_predictions.json")
DRIVERS_CSV      = Path("drivers_ml_ready.csv")

# ── V2 Chaos Module Configuration ──────────────────────────────────────────────
MECHANICAL_DNF_FRACTION        = 0.40
INCIDENT_DNF_FRACTION          = 0.60
FRONT_GRID_CUTOFF              = 5
FRONT_GRID_INCIDENT_MULTIPLIER = 1.35
SURVIVAL_LAP_THRESHOLD         = 20
HIGH_DNF_RISK_THRESHOLD        = 0.28
SURVIVAL_BONUS_ADVANCE         = 2

# ── V3 Elite Physics & Strategy Constants ──────────────────────────────────────
FUEL_BURN_PER_LAP           = 0.06  # Seconds gained per lap via fuel burn
DIRTY_AIR_PENALTY           = 1.15  # Pace multiplier when stuck in traffic
TRAFFIC_GAP_THRESHOLD       = 1.0   # Seconds gap to trigger dirty air
BLUE_FLAG_THRESHOLD         = 80.0  # Gap size (seconds) to trigger lapping
BLUE_FLAG_PENALTY           = 3.0   # Seconds added to lapped car
SC_COMPACT_GAP              = 0.2   # Time gap between cars post-Safety Car
PIT_WINDOW_START            = 20    # Lap when pit window opens
PIT_STOP_BASE_TIME          = 20.0  # Base seconds lost in pit lane
BOTCHED_PIT_PENALTY         = 8.0   # Seconds added on pit blunder
AGGRESSION_SPIKE_MULTIPLIER = 2.0   # DNF multiplier if stuck in traffic > 5 laps


# ── load_simulation_data ──────────────────────────────────────────────────────
def load_simulation_data():
    """Loads baseline data and maps dynamic variables from V3 & V4 features."""
    if not DRIVERS_CSV.exists() or not PREDICTIONS_JSON.exists():
        raise FileNotFoundError("Missing required files. Run market_predict.py first.")

    with open(PREDICTIONS_JSON, "r") as fh:
        predict_data = json.load(fh)

    df_drivers = pd.read_csv(DRIVERS_CSV)

    meta        = predict_data.get("metadata", {})
    track_name  = meta.get("track_name", "Unknown Track")
    circuit_dnf = meta.get("circuit_dnf_rate", 0.12)

    drivers_json = predict_data.get("drivers", {})

    driver_stats = []
    for d_name, d_data in drivers_json.items():
        csv_row = df_drivers[df_drivers["Driver"].str.strip() == d_name]

        overtake_eff = csv_row["Overtake_Efficiency"].values[0] if not csv_row.empty and "Overtake_Efficiency" in csv_row.columns else 0.5
        deg_slope    = csv_row["FP2_DegSlope"].values[0] if not csv_row.empty and "FP2_DegSlope" in csv_row.columns else 0.015
        hype_ratio   = csv_row["Hype_Ratio"].values[0] if not csv_row.empty and "Hype_Ratio" in csv_row.columns else 0.2
        pit_speed    = csv_row["FastestPitstops"].values[0] if not csv_row.empty and "FastestPitstops" in csv_row.columns else 2.5
        
        # Level 4 Hook: Extract Team info for Team Orders
        team         = csv_row["Team"].values[0] if not csv_row.empty and "Team" in csv_row.columns else f"Team_{d_name[:3].upper()}"

        pace_score = d_data.get("base_points", 1.0)

        driver_stats.append({
            "driver":            d_name,
            "pace":              pace_score,
            "dnf_prob_per_race": d_data.get("dnf_probability", 0.05),
            "overtake_eff":      overtake_eff,
            "deg_slope":         deg_slope,
            "hype_ratio":        hype_ratio,
            "pit_speed":         pit_speed,
            "team":              team
        })

    driver_stats.sort(key=lambda x: x["pace"], reverse=True)
    return track_name, circuit_dnf, driver_stats


# ── run_monte_carlo (V4 Pro-Strategist Loop) ──────────────────────────────────
def run_monte_carlo(track_name, circuit_dnf, drivers):
    num_drivers  = len(drivers)
    driver_names = [d["driver"] for d in drivers]
    team_array   = [d["team"] for d in drivers]

    base_lap_dnf = np.array([1.0 - (1.0 - d["dnf_prob_per_race"]) ** (1.0 / TOTAL_LAPS) for d in drivers])
    base_mech_dnf     = base_lap_dnf * MECHANICAL_DNF_FRACTION
    base_incident_dnf = base_lap_dnf * INCIDENT_DNF_FRACTION

    # Convert pace score to a base lap time (lower is faster)
    pace_scores       = np.array([d["pace"] for d in drivers])
    base_lap_times    = 90.0 - pace_scores 
    
    overtake_array    = np.array([d["overtake_eff"] for d in drivers])
    deg_slope_array   = np.array([d["deg_slope"] for d in drivers])
    hype_array        = np.array([d["hype_ratio"] for d in drivers])
    pit_speed_array   = np.array([d["pit_speed"] for d in drivers])
    high_dnf_mask     = np.array([d["dnf_prob_per_race"] > HIGH_DNF_RISK_THRESHOLD for d in drivers])

    lap_sc_prob = circuit_dnf / TOTAL_LAPS
    is_street = "Monaco" in track_name or "Singapore" in track_name or "Baku" in track_name
    track_overtake_scaler = 0.15 if is_street else 0.85

    # Level 4 Upgrade 1: Scaled Rain Probability from Circuit Attrition Metadata
    rain_probability = min(0.35, circuit_dnf * 1.5)

    finish_positions = {name: [] for name in driver_names}
    sc_in_first_10   = 0

    logging.info(f"Starting {N_SIMULATIONS:,} Level 4 Pro-Strategist Simulations for {track_name}...")
    start_time = time.time()

    for sim in range(N_SIMULATIONS):
        # Tracking cumulative race time in seconds
        cumulative_time = np.arange(num_drivers) * 0.5 
        active_status   = np.ones(num_drivers, dtype=bool) 
        
        stuck_laps             = np.zeros(num_drivers)
        pit_completed          = np.zeros(num_drivers, dtype=bool)
        survival_bonus_applied = np.zeros(num_drivers, dtype=bool)
        sim_sc_first_10        = False

        # Level 4 Track/Driver States
        is_raining            = np.random.rand() < rain_probability
        consecutive_fast_laps = np.zeros(num_drivers)
        confidence_boost_laps = np.zeros(num_drivers)

        for lap in range(1, TOTAL_LAPS + 1):
            
            active_idxs = [i for i in range(num_drivers) if active_status[i]]
            active_idxs.sort(key=lambda x: cumulative_time[x])

            # ── Idea 6: Safety Car Grid Compact Engine ────────────────────────
            sc_chance = lap_sc_prob * (LAP1_SC_RISK_MULTIPLIER if lap == 1 else 1.0)
            if is_raining: sc_chance *= 1.5  # Rain elevates track hazards
            
            if np.random.rand() < sc_chance:
                if lap <= 10: sim_sc_first_10 = True
                for k in range(1, len(active_idxs)):
                    cumulative_time[active_idxs[k]] = cumulative_time[active_idxs[k-1]] + SC_COMPACT_GAP
                continue 

            # ── Idea 1 & 7: Tyre Degradation & Fuel Burn Physics ──────────────
            current_lap_times = base_lap_times + (lap * deg_slope_array) - (lap * FUEL_BURN_PER_LAP)

            # ── Level 4 Upgrade 3: Track Rubbering-In (Evolution) ─────────────
            global_grip_gain = 1.0 - (0.0005 * lap)
            current_lap_times *= global_grip_gain

            # ── Level 4 Upgrade 1: Rain Engine Pace Variance ──────────────────
            if is_raining:
                current_lap_times += np.random.normal(0, 0.45, num_drivers)  # Variance triples
            else:
                current_lap_times += np.random.normal(0, 0.15, num_drivers)  # Standard stochastic variance

            # ── Level 4 Upgrade 4: "Clutch" Momentum Streak ───────────────────
            for idx in active_idxs:
                if current_lap_times[idx] < base_lap_times[idx]:
                    consecutive_fast_laps[idx] += 1
                else:
                    consecutive_fast_laps[idx] = 0

                if consecutive_fast_laps[idx] >= 3:
                    confidence_boost_laps[idx] = 5

                if confidence_boost_laps[idx] > 0:
                    current_lap_times[idx] *= 0.95  # 5% confidence pace surge
                    confidence_boost_laps[idx] -= 1

            # ── Idea 2 & 9: Traffic Dampening and Blue Flags ──────────────────
            for k in range(1, len(active_idxs)):
                ahead_idx  = active_idxs[k-1]
                behind_idx = active_idxs[k]
                gap = cumulative_time[behind_idx] - cumulative_time[ahead_idx]

                if gap > BLUE_FLAG_THRESHOLD: # Backmarker being lapped
                    current_lap_times[behind_idx] += BLUE_FLAG_PENALTY
                    continue

                if gap < TRAFFIC_GAP_THRESHOLD:
                    current_lap_times[behind_idx] *= DIRTY_AIR_PENALTY
                    stuck_laps[behind_idx] += 1
                else:
                    stuck_laps[behind_idx] = 0

            # Calculate proposed state
            proposed_time = cumulative_time.copy()
            for idx in active_idxs:
                proposed_time[idx] += current_lap_times[idx]

            # ── Idea 5 & Level 4 Upgrade 1: Dynamic Pit Strategy shifts ───────
            effective_pit_start = 1 if is_raining else PIT_WINDOW_START
            pit_chance          = 0.35 if is_raining else 0.15

            if lap >= effective_pit_start:
                for idx in active_idxs:
                    if not pit_completed[idx] and np.random.rand() < pit_chance: 
                        pit_time = PIT_STOP_BASE_TIME + np.random.normal(0, pit_speed_array[idx])
                        if is_raining: pit_time += 2.0  # Handling switch overhead
                        if np.random.rand() < (0.10 / pit_speed_array[idx]): # Botch check
                            pit_time += BOTCHED_PIT_PENALTY
                        proposed_time[idx] += pit_time
                        pit_completed[idx] = True

            # ── Idea 8: Real-Time Overtake Check ──────────────────────────────
            new_order = sorted(active_idxs, key=lambda x: proposed_time[x])
            for k in range(1, len(new_order)):
                car_now_ahead  = new_order[k-1]
                car_now_behind = new_order[k]

                # Check if they swapped physical time ranking
                if active_idxs.index(car_now_ahead) > active_idxs.index(car_now_behind): 
                    eff_atk = overtake_array[car_now_ahead]
                    eff_def = overtake_array[car_now_behind]
                    prob = (eff_atk / (eff_atk + eff_def + 0.001)) * track_overtake_scaler

                    if np.random.rand() > prob:
                        # Defense holds! Lock behind car to bumper
                        proposed_time[car_now_ahead] = proposed_time[car_now_behind] + 0.1

            # ── Level 4 Upgrade 2: Tactical Team Orders Logic ─────────────────
            final_ordered_idxs = sorted(active_idxs, key=lambda x: proposed_time[x])
            for k in range(1, len(final_ordered_idxs)):
                car_ahead  = final_ordered_idxs[k-1]
                car_behind = final_ordered_idxs[k]

                if team_array[car_ahead] == team_array[car_behind]:
                    # Car behind is executing significantly faster baseline pace
                    if pace_scores[car_behind] > pace_scores[car_ahead]:
                        # Wingman complies unless they have an aggressive hype ratio
                        if hype_array[car_ahead] <= 0.4:
                            tmp = proposed_time[car_ahead]
                            proposed_time[car_ahead] = proposed_time[car_behind]
                            proposed_time[car_behind] = tmp

            cumulative_time = proposed_time

            # ── V2 Survival Bonus mapped to Time ──────────────────────────────
            if lap == SURVIVAL_LAP_THRESHOLD:
                for idx in active_idxs:
                    if high_dnf_mask[idx] and not survival_bonus_applied[idx]:
                        cumulative_time[idx] -= (SURVIVAL_BONUS_ADVANCE * GAP_MULTIPLIER_SECONDS)
                        survival_bonus_applied[idx] = True

            # ── Idea 3 & 4: Dynamic DNF Multipliers ───────────────────────────
            dnf_multiplier = LAP1_DNF_RISK_MULTIPLIER if lap == 1 else 1.0
            if is_raining: dnf_multiplier *= 3.0  # Environmental threat multiplier
            
            lap_incident_modifier = np.ones(num_drivers)

            for grid_pos, idx in enumerate(active_idxs):
                if grid_pos < FRONT_GRID_CUTOFF:
                    lap_incident_modifier[idx] *= FRONT_GRID_INCIDENT_MULTIPLIER
                
                if lap == 1:
                    if grid_pos < 3: lap_incident_modifier[idx] *= 0.5
                    elif 9 <= grid_pos <= 15: lap_incident_modifier[idx] *= 1.5
                
                if stuck_laps[idx] > 5:
                    lap_incident_modifier[idx] *= (1.0 + hype_array[idx]) * AGGRESSION_SPIKE_MULTIPLIER

            # ── Incident Roll & Post-Crash Execution ──────────────────────────
            # Level 4 Upgrade 5: Reliability Wear & Tear factor scales up over laps
            reliability_factor = lap / TOTAL_LAPS

            mech_rolls  = np.random.rand(num_drivers)
            mech_dnfs   = (mech_rolls < (base_mech_dnf * dnf_multiplier * reliability_factor)) & active_status
            active_status[mech_dnfs] = False

            incident_rolls = np.random.rand(num_drivers)
            incident_dnfs  = (incident_rolls < (base_incident_dnf * dnf_multiplier * lap_incident_modifier)) & active_status
            active_status[incident_dnfs] = False

        # ── End of Race Classification ────────────────────────────────────────
        active_finishers = [i for i in range(num_drivers) if active_status[i]]
        active_finishers.sort(key=lambda x: cumulative_time[x])
        dnf_finishers = [i for i in range(num_drivers) if not active_status[i]]

        final_classification = active_finishers + dnf_finishers
        for final_pos, driver_idx in enumerate(final_classification):
            finish_positions[driver_names[driver_idx]].append(final_pos + 1)

        if sim_sc_first_10:
            sc_in_first_10 += 1

    elapsed = time.time() - start_time
    logging.info(f"Simulations complete in {elapsed:.2f} seconds.")

    return finish_positions, sc_in_first_10


# ── format_output ─────────────────────────────────────────────────────────────
def format_output(finish_positions, sc_in_first_10, drivers):
    print("\n" + "═" * 80)
    print(f"  MONTE CARLO RACE SIMULATION RESULTS ({N_SIMULATIONS:,} Iterations)")
    print("  ENGINE: Level 4 Pro-Strategist Custom Build")
    print("═" * 80)

    results_summary = []
    for d in drivers:
        name      = d["driver"]
        positions = np.array(finish_positions[name])

        avg_pos   = np.mean(positions)
        win_pct   = np.mean(positions == 1)  * 100
        podium_pct = np.mean(positions <= 3) * 100
        top10_pct  = np.mean(positions <= 10) * 100
        dnf_pct    = np.mean(positions > 15)  * 100  

        results_summary.append({
            "name":    name,
            "avg_pos": avg_pos,
            "win":     win_pct,
            "podium":  podium_pct,
            "top10":   top10_pct,
            "dnf":     dnf_pct,
        })

    results_summary.sort(key=lambda x: x["avg_pos"])

    print("\n  📊 TABLE 1: OUTCOME PROBABILITIES (ALL 22 DRIVERS)")
    print("  " + "─" * 76)
    print(f"  {'Rank':<4} | {'Driver':<14} | {'Avg Pos':>7} | {'Win %':>8} | {'Podium %':>8} | {'Top 10 %':>8} | {'DNF %':>8}")
    print("  " + "─" * 76)

    for i, row in enumerate(results_summary):
        print(f"  {'P'+str(i+1):<4} | {row['name']:<14} | {row['avg_pos']:>7.2f} | {row['win']:>7.1f}% | {row['podium']:>7.1f}% | {row['top10']:>7.1f}% | {row['dnf']:>7.1f}%")

    print("\n\n  ⏱️  TABLE 2: EXPECTED FINAL CLASSIFICATION & TIME GAPS")
    print("  " + "─" * 76)
    print(f"  {'Pos':<4} | {'Driver':<14} | {'Avg Pos':>7} | {'Interval':>12} | {'Gap to Leader':>15}")
    print("  " + "─" * 76)

    leader_avg_pos = results_summary[0]["avg_pos"]
    for i, row in enumerate(results_summary):
        pos = i + 1
        if pos == 1:
            interval_str, gap_leader_str = "Leader", "-"
        else:
            prev_avg_pos   = results_summary[i - 1]["avg_pos"]
            interval_sec   = (row["avg_pos"] - prev_avg_pos) * GAP_MULTIPLIER_SECONDS
            leader_gap_sec = (row["avg_pos"] - leader_avg_pos) * GAP_MULTIPLIER_SECONDS

            if row["dnf"] > 50.0:
                interval_str, gap_leader_str = "DNF", "DNF"
            else:
                interval_str, gap_leader_str = f"+{interval_sec:.3f}s", f"+{leader_gap_sec:.3f}s"

        print(f"  {'P'+str(pos):<4} | {row['name']:<14} | {row['avg_pos']:>7.2f} | {interval_str:>12} | {gap_leader_str:>15}")

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

