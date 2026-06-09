"""
weekend_predictor.py  —  F1 Prediction Game Solver  (2026 Season)
═══════════════════════════════════════════════════════════════════════════════
Strictly FP-Telemetry Driven Prediction Engine.

Requires (run in order before this script):
  1. market_harvester.py   → writes drivers_ml_ready.csv + constructors_ml_ready.csv
  2. market_predict.py     → writes market_predictions.json

Answers all known F1 Prediction Game question variants:

  CORE (always present):
    Q1  — Full race top-10 finishers
    Q2  — Pole position driver
    Q3  — All possible driver 1v1 qualifying matchups (full matrix)
    Q4  — Team qualifying segment (Q1/Q2/Q3) projections — all teams, all outcomes
    Q5  — Which team sets the fastest DHL pitstop
    Q6  — Will there be a Safety Car or VSC during the GP
    Q7  — Midfield 4-way head-to-head (any combination)
    Q8  — Driver position bracket (Podium / 4th–10th / 11th–22nd / NC)
    Q9  — Who sets the fastest lap of the race
    Q10 — How many classified finishers

  VARIANT (appear some weekends):
    V1  — How many times will the red flag be used
    V2  — Will all 22 cars complete the first lap
    V3  — How many teams score points in the Sprint
    V4  — Which driver finishes highest in the Sprint
    V5  — Who leads the Drivers' Championship after this race
    V6  — Where does [driver] finish in the Sprint
    V7  — Will the Safety Car/VSC be required in the first 10 laps
    V8  — Who leads the Constructors' Championship after this race
    V9  — How many teams will get a driver into Q3
    V10 — Which team scores the most points over the weekend
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PREDICTIONS_JSON  = Path("market_predictions.json")
DRIVERS_CSV       = Path("drivers_ml_ready.csv")
CONSTRUCTORS_CSV  = Path("constructors_ml_ready.csv")

# ── Sprint weekends in 2026 (race rounds) ─────────────────────────────────────
SPRINT_ROUNDS = {2, 6, 9, 18, 20, 22}   # adjust as the calendar firms up

# ── Points system (standard + fastest lap bonus) ──────────────────────────────
POINTS_TABLE = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}
SPRINT_POINTS = {1:8, 2:7, 3:6, 4:5, 5:4, 6:3, 7:2, 8:1}

# ── Safety car / red flag / lap-1 thresholds ─────────────────────────────────
SC_THRESHOLD        = 0.075   # circuit_dnf_rate ≥ this → SC/VSC expected
SC_EARLY_THRESHOLD  = 0.085   # for "first 10 laps" variant
RF_THRESHOLD        = 0.100   # red flag expected
LAP1_INCIDENT_LIMIT = 0.090   # above this → not all 22 will complete lap 1


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_data():
    """
    Loads telemetry dataframes and the market predictions json payload defensively.
    Maps keys to all case-sensitive and abbreviation variants (including DNF_Prob) 
    used across the downstream solver modules.
    """
    if not DRIVERS_CSV.exists() or not CONSTRUCTORS_CSV.exists():
        raise FileNotFoundError("Missing ML-ready CSV files. Run market_harvester.py first.")
    if not PREDICTIONS_JSON.exists():
        raise FileNotFoundError("Missing market_predictions.json. Run market_predict.py first.")

    # Load telemetry structures
    df_drivers = pd.read_csv(DRIVERS_CSV)
    df_constructors = pd.read_csv(CONSTRUCTORS_CSV)

    # Load JSON model payload
    with open(PREDICTIONS_JSON, "r") as fh:
        predict_data = json.load(fh)

    # Extract metadata safely
    meta = predict_data.get("metadata", {})
    
    # Forward-compatibility patch for main() keys
    meta["track_name"] = meta.get("track_name", "Monaco Grand Prix")
    meta["race_round"] = meta.get("race_round", 8)
    meta["is_sprint"] = meta.get("is_sprint", False)
    meta["circuit_dnf_rate"] = meta.get("circuit_dnf_rate", 0.12)

    track_name = meta["track_name"]

    # Defensive parsing dictionaries
    drivers_json = predict_data.get("drivers", {})
    constructors_json = predict_data.get("constructors", {})

    dnf_prob_map = {k: v.get("dnf_probability", 0.05) for k, v in drivers_json.items()}
    exp_points_map = {k: v.get("expected_points", 0.0) for k, v in drivers_json.items()}
    base_points_map = {k: v.get("base_points", 0.0) for k, v in drivers_json.items()}

    # Map variables back to columns using lowercase formats
    df_drivers["dnf_probability"] = df_drivers["Driver"].map(dnf_prob_map).fillna(0.05)
    df_drivers["expected_points"] = df_drivers["Driver"].map(exp_points_map).fillna(0.0)
    df_drivers["base_points"] = df_drivers["Driver"].map(base_points_map).fillna(0.0)

    # Core CamelCase alignments and abbreviation mappings for ALL solver files (Q1 - Q10)
    df_drivers["ExpectedPoints"] = df_drivers["expected_points"]
    df_drivers["BasePoints"] = df_drivers["base_points"]
    df_drivers["DNFProbability"] = df_drivers["dnf_probability"]
    df_drivers["DNF_Prob"] = df_drivers["dnf_probability"]  

    # Map constructors safely
    c_exp_map = {k: v.get("expected_points", 0.0) for k, v in constructors_json.items()}
    df_constructors["expected_points"] = df_constructors["Constructor"].map(c_exp_map).fillna(0.0)
    df_constructors["ExpectedPoints"] = df_constructors["expected_points"]

    # --- BUG FIX: Derive Constructor DNF_Prob ---
    # Convert historical team DNFs into a 0.0 - 1.0 probability multiplier for Q5
    if "DNFs" in df_constructors.columns:
        max_dnfs = df_constructors["DNFs"].max()
        max_dnfs = max_dnfs if max_dnfs > 0 else 1
        df_constructors["DNF_Prob"] = df_constructors["DNFs"] / max_dnfs
    else:
        df_constructors["DNF_Prob"] = 0.05

    print(f"[INFO] Successfully loaded telemetry and model data for: {track_name}")
    return meta, df_drivers, df_constructors, predict_data

# ═══════════════════════════════════════════════════════════════════════════════
# FORMATTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

W = 72

def header(title: str, char: str = "═") -> None:
    print(char * W)
    print(f"  {title}")
    print(char * W)

def sub_header(title: str) -> None:
    print(f"\n  ── {title} {'─' * (W - len(title) - 6)}")

def divider(char: str = "─") -> None:
    print("  " + char * (W - 2))

def confidence_bar(score: float, max_score: float, width: int = 12) -> str:
    """ASCII progress bar for relative confidence."""
    filled = int(round((score / max_score) * width)) if max_score > 0 else 0
    return "█" * filled + "░" * (width - filled)

def medal(rank: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"P{rank:<2}")

def confidence_label(score: float, ranked_scores: list) -> str:
    """HIGH / MED / LOW relative to peer group."""
    if not ranked_scores:
        return "MED"
    top33  = np.percentile(ranked_scores, 67)
    bot33  = np.percentile(ranked_scores, 33)
    if score >= top33:   return "HIGH"
    elif score >= bot33: return "MED "
    return "LOW "


# ═══════════════════════════════════════════════════════════════════════════════
# Q1 — PREDICTED RACE TOP 10 FINISHERS
# ═══════════════════════════════════════════════════════════════════════════════

def q1_race_top10(drivers: pd.DataFrame, circuit_dnf: float) -> None:
    """
    Ranking signal: ExpectedPoints (already DNF-adjusted from market_predict.py).
    FP_Progression is used as a tie-breaker / recency weighting (+bonus).
    """
    df = drivers.copy()
    df["RaceScore"] = (
        df["ExpectedPoints"] * 0.70
      + df["AvgPoints"]       * 0.20
      + df["FP_Progression"].fillna(0).clip(-10, 10) * 0.5   # recency momentum
    )
    df = df.sort_values("RaceScore", ascending=False).reset_index(drop=True)
    top10 = df.head(10)

    all_scores = df["RaceScore"].tolist()

    header("Q1  ·  PREDICTED GRAND PRIX TOP 10 FINISHERS")
    print(f"\n  {'Pos':<5} {'Driver':<16} {'Team':<14} {'Exp.Pts':>8}  {'Confidence':>14}  {'Signal'}")
    divider()
    for i, row in top10.iterrows():
        pos   = i + 1
        bar   = confidence_bar(row["RaceScore"], all_scores[0])
        conf  = confidence_label(row["RaceScore"], all_scores)
        print(
            f"  {medal(pos):<5} {row['Driver']:<16} {row['Team']:<14} "
            f"{row['ExpectedPoints']:>7.2f}p  {bar}  {conf}"
        )
    divider()

    # Podium call with brief rationale
    p1, p2, p3 = df.iloc[0], df.iloc[1], df.iloc[2]
    print(f"\n  PREDICTED PODIUM:")
    for pos, drv in enumerate([p1, p2, p3], 1):
        gap_note = ""
        if drv["FP3_DeltaBest"] < 3.0:
            gap_note = "  ← very strong qualifying pace"
        elif drv["FP2_DegSlope"] < 0:
            gap_note = "  ← improving tyre deg trend"
        print(f"    {medal(pos)} {drv['Driver']:<16} ({drv['Team']}){gap_note}")

    # DNF watch-outs
    high_dnf = df[(df["DNF_Prob"] > 0.15) & (df.index < 10)]
    if not high_dnf.empty:
        print(f"\n  ⚠  DNF WATCH in top 10: " +
              ", ".join(f"{r['Driver']} ({r['DNF_Prob']*100:.0f}%)" for _, r in high_dnf.iterrows()))
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Q2 — POLE POSITION PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def q2_pole_position(drivers: pd.DataFrame) -> None:
    """
    FP3_DeltaBest = gap to session best on a representative single lap.
    Lowest delta = closest to outright pace = best pole candidate.
    FP_Progression adds recency momentum signal (negative = improving).
    """
    df = drivers.copy()
    df["PoleScore"] = df["FP3_DeltaBest"] - df["FP_Progression"].fillna(0).clip(-5, 5) * 0.3
    df = df.sort_values("PoleScore", ascending=True).reset_index(drop=True)

    all_deltas = df["PoleScore"].tolist()

    header("Q2  ·  POLE POSITION PREDICTION")
    print(f"\n  {'Pos':<5} {'Driver':<16} {'Team':<14} {'FP3 Δ Best':>10} {'FP Trend':>10}  {'Conf'}")
    divider()

    for i in range(min(8, len(df))):
        row  = df.iloc[i]
        conf = confidence_label(-row["PoleScore"], [-s for s in all_deltas])
        trend_symbol = "↑ improving" if row["FP_Progression"] < -1 else (
                       "↓ dropping"  if row["FP_Progression"] >  1 else "→ stable  ")
        print(
            f"  {'P'+str(i+1):<5} {row['Driver']:<16} {row['Team']:<14} "
            f"{row['FP3_DeltaBest']:>9.4f}s {trend_symbol:>10}  {conf}"
        )

    divider()
    pole = df.iloc[0]
    print(f"\n  ✦ PREDICTED POLE:  {pole['Driver']}  ({pole['Team']})")
    print(f"    FP3 gap to best: {pole['FP3_DeltaBest']:.4f}s")
    if pole["FP_Progression"] < -1:
        print(f"    Trend: Improving across practice — confidence is HIGH")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Q3 — ALL 1V1 QUALIFYING MATCHUPS
# ═══════════════════════════════════════════════════════════════════════════════

def q3_qualifying_1v1s(drivers: pd.DataFrame) -> None:
    """
    Predict head-to-head qualifying outcome for every possible driver pairing.
    Uses FP3_DeltaBest as the primary signal (lower = faster).
    """
    df = drivers[["Driver", "Team", "FP3_DeltaBest", "FP_Progression"]].copy()
    df["PoleScore"] = (
        df["FP3_DeltaBest"]
      - df["FP_Progression"].fillna(0).clip(-5, 5) * 0.3
    )
    df = df.sort_values("PoleScore", ascending=True).reset_index(drop=True)

    # Build ranking lookup
    rank_map  = {row["Driver"]: i for i, row in df.iterrows()}
    score_map = {row["Driver"]: row["PoleScore"] for _, row in df.iterrows()}

    driver_list = df["Driver"].tolist()
    all_pairs   = list(combinations(driver_list, 2))

    header("Q3  ·  HEAD-TO-HEAD QUALIFYING MATCHUPS  (ALL PAIRS)")
    print(f"\n  Qualifying rank reference (fastest → slowest):")
    divider()
    for i, row in df.iterrows():
        bar = confidence_bar(1 / (row["PoleScore"] + 0.001), 1 / (df["PoleScore"].min() + 0.001))
        print(f"    {'Q'+str(i+1):<5} {row['Driver']:<16} {row['Team']:<14}  Δ{row['FP3_DeltaBest']:.4f}s  {bar}")

    print(f"\n  Full 1v1 prediction matrix  ({len(all_pairs)} matchups):")
    divider()
    print(f"  {'Driver A':<16} vs  {'Driver B':<16}  →  {'Predicted Winner':<16}  Margin  Conf")
    divider()

    for d1, d2 in all_pairs:
        r1, r2 = rank_map[d1], rank_map[d2]
        s1, s2 = score_map[d1], score_map[d2]
        winner  = d1 if r1 < r2 else d2
        loser   = d2 if r1 < r2 else d1
        margin  = abs(s1 - s2)
        rank_gap = abs(r1 - r2)

        if rank_gap >= 8:   conf = "HIGH"
        elif rank_gap >= 4: conf = "MED "
        else:               conf = "LOW "

        print(
            f"  {d1:<16} vs  {d2:<16}  →  {winner:<16} "
            f" Δ{margin:.3f}s  {conf}"
        )

    print(f"\n  TIP: For the game's random 2-driver Q3 question — find both names\n"
          f"       in the table above and pick the predicted winner directly.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Q4 — TEAM QUALIFYING SEGMENT PROJECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def q4_team_qualifying_segments(drivers: pd.DataFrame) -> None:
    df = drivers[["Driver", "Team", "FP3_DeltaBest", "FP_Progression"]].copy()
    df["PoleScore"] = (
        df["FP3_DeltaBest"]
      - df["FP_Progression"].fillna(0).clip(-5, 5) * 0.3
    )
    df = df.sort_values("PoleScore", ascending=True).reset_index(drop=True)

    Q3_CUT = 10
    Q2_CUT = 15

    def segment(pos):  
        if pos <= Q3_CUT: return "Q3"
        if pos <= Q2_CUT: return "Q2"
        return "Q1 EXIT"

    df["QSegment"] = [segment(i + 1) for i in range(len(df))]
    df["QPos"]     = range(1, len(df) + 1)

    team_results = df.groupby("Team").apply(
        lambda g: {
            "drivers": list(zip(g["Driver"], g["QPos"], g["QSegment"])),
            "both_in_Q3":    (g["QSegment"] == "Q3").sum() == 2,
            "one_in_Q3":     (g["QSegment"] == "Q3").sum() == 1,
            "zero_in_Q3":    (g["QSegment"] == "Q3").sum() == 0,
            "q1_exit_count": (g["QSegment"] == "Q1 EXIT").sum(),
            "q2_exit_count": (g["QSegment"] == "Q2").sum(),
        }
    , include_groups=False)

    header("Q4  ·  TEAM QUALIFYING SEGMENT PROJECTIONS")
    print(f"\n  Legend:  ✅ Both cars  |  〽 One car  |  ❌ No cars  per segment\n")
    print(f"  {'Team':<15}  {'Driver 1':<14} (QPos)  {'Driver 2':<14} (QPos)  Q3?   Q2   Q1exit")
    divider()

    for team in sorted(team_results.index):
        res = team_results[team]
        drv = sorted(res["drivers"], key=lambda x: x[1])  
        d1  = f"{drv[0][0]:<14}  Q{drv[0][1]:<3}" if len(drv) > 0 else "—"
        d2  = f"{drv[1][0]:<14}  Q{drv[1][1]:<3}" if len(drv) > 1 else "—"

        q3_icon  = "✅" if res["both_in_Q3"] else ("〽" if res["one_in_Q3"] else "❌")
        q2_icon  = f"{res['q2_exit_count']} car(s)" if res["q2_exit_count"] else "  —   "
        q1_icon  = f"{res['q1_exit_count']} car(s)" if res["q1_exit_count"] else "  —   "

        print(f"  {team:<15}  {d1}  {d2}  {q3_icon}    {q2_icon}  {q1_icon}")

    divider()
    print(f"\n  ANSWER VARIANTS:")

    q3_teams = [t for t, r in team_results.items() if r["both_in_Q3"] or r["one_in_Q3"]]
    q3_both  = [t for t, r in team_results.items() if r["both_in_Q3"]]
    q1_exits = [(t, r["q1_exit_count"]) for t, r in team_results.items() if r["q1_exit_count"] > 0]
    q2_exits = [(t, r["q2_exit_count"]) for t, r in team_results.items() if r["q2_exit_count"] > 0]

    print(f"  • Teams with BOTH cars in Q3 ({len(q3_both)}): {', '.join(q3_both) or 'None'}")
    print(f"  • Teams with at least ONE car in Q3 ({len(q3_teams)}): {', '.join(q3_teams) or 'None'}")

    if q2_exits:
        print(f"  • Teams exiting in Q2:")
        for t, cnt in q2_exits:
            team_drv = [d for d in team_results[t]["drivers"] if d[2] == "Q2"]
            print(f"      {t}: " + ", ".join(f"{d[0]} (P{d[1]})" for d in team_drv))

    if q1_exits:
        print(f"  • Teams exiting in Q1:")
        for t, cnt in q1_exits:
            team_drv = [d for d in team_results[t]["drivers"] if d[2] == "Q1 EXIT"]
            print(f"      {t}: " + ", ".join(f"{d[0]} (P{d[1]})" for d in team_drv))
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Q5 — FASTEST DHL PITSTOP TEAM
# ═══════════════════════════════════════════════════════════════════════════════

def q5_fastest_pitstop(constructors: pd.DataFrame) -> None:
    df = constructors.copy()

    max_fp  = df["FastestPitstops"].max() or 1
    max_ppm = df["PPM"].max() or 1

    df["PitScore"] = (
        (df["FastestPitstops"] / max_fp)  * 0.60
      + (1 - df["DNF_Prob"])              * 0.25
      + (df["PPM"] / max_ppm)             * 0.15
    )
    df = df.sort_values("PitScore", ascending=False).reset_index(drop=True)

    header("Q5  ·  FASTEST DHL PITSTOP TEAM FORECAST")
    print(f"\n  {'Rank':<6} {'Team':<16} {'Fast Stops':>10} {'DNF%':>6} {'PPM':>6}  {'Score':>7}  {'Bar'}")
    divider()

    all_scores = df["PitScore"].tolist()
    for i, row in df.iterrows():
        bar = confidence_bar(row["PitScore"], all_scores[0])
        print(
            f"  {'P'+str(i+1):<6} {row['Constructor']:<16} {row['FastestPitstops']:>10} "
            f"{row['DNF_Prob']*100:>5.1f}%  {row['PPM']:>5.2f}  {row['PitScore']:>7.4f}  {bar}"
        )

    divider()
    winner = df.iloc[0]
    print(f"\n  ✦ PREDICTED FASTEST PITSTOP:  {winner['Constructor']}")
    print(f"    Season fastest stops: {int(winner['FastestPitstops'])}  |  DNF risk: {winner['DNF_Prob']*100:.1f}%\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Q6 — SAFETY CAR / VSC DURING THE GP
# ═══════════════════════════════════════════════════════════════════════════════

def q6_safety_car(circuit_dnf: float, drivers: pd.DataFrame) -> None:
    avg_driver_dnf_rate = drivers["DNF_Prob"].mean()
    combined_sc_risk = (circuit_dnf * 0.60) + (avg_driver_dnf_rate * 0.40)
    sc_probability = min(combined_sc_risk / 0.15, 1.0)  

    header("Q6  ·  SAFETY CAR / VSC APPEARANCE FORECAST")
    print(f"\n  Circuit baseline DNF rate : {circuit_dnf * 100:.1f}%")
    print(f"  Driver field DNF exposure : {avg_driver_dnf_rate * 100:.1f}% (season avg)")
    print(f"  Combined SC risk index    : {combined_sc_risk * 100:.1f}%")

    bar = confidence_bar(sc_probability, 1.0, 20)
    print(f"\n  SC/VSC probability: [{bar}]  {sc_probability * 100:.0f}%\n")

    if circuit_dnf >= SC_THRESHOLD:
        verdict = "YES — Safety Car or VSC is LIKELY"
        note = "High-risk circuit profile. Expect at least 1 SC/VSC intervention."
    elif circuit_dnf >= 0.065:
        verdict = "LIKELY — lean YES"
        note = "Moderate circuit risk. SC possible; VSC more probable than nothing."
    else:
        verdict = "NO — SC/VSC unlikely"
        note = "Low-risk circuit. Clean race possible but never certain in F1."

    print(f"  ✦ PREDICTED OUTCOME:  {verdict}")
    print(f"    Note: {note}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Q7 — MIDFIELD 4-WAY HEAD-TO-HEAD
# ═══════════════════════════════════════════════════════════════════════════════

def q7_midfield_4way(drivers: pd.DataFrame, midfield_drivers: list[str] | None = None) -> None:
    df = drivers.copy()
    df["RaceScore"] = (
        df["ExpectedPoints"] * 0.70
      + df["AvgPoints"]       * 0.20
      + df["FP_Progression"].fillna(0).clip(-10, 10) * 0.5
    )

    if midfield_drivers:
        subset = df[df["Driver"].str.upper().isin([m.upper() for m in midfield_drivers])]
    else:
        sorted_all = df.sort_values("RaceScore", ascending=False).reset_index(drop=True)
        subset = sorted_all.iloc[6:14].head(4)

    subset = subset.sort_values("RaceScore", ascending=False).reset_index(drop=True)

    header("Q7  ·  MIDFIELD HEAD-TO-HEAD RACE RANKING")
    print(f"\n  Drivers evaluated: {', '.join(subset['Driver'].tolist())}\n")
    print(f"  {'Rank':<6} {'Driver':<16} {'Team':<14} {'Exp.Pts':>8}  {'Signal'}")
    divider()

    all_scores = df["RaceScore"].tolist()
    for i, row in subset.iterrows():
        conf = confidence_label(row["RaceScore"], all_scores)
        bar  = confidence_bar(row["RaceScore"], max(all_scores))
        print(
            f"  {'P'+str(subset.index.get_loc(i)+1):<6} {row['Driver']:<16} {row['Team']:<14} "
            f"{row['ExpectedPoints']:>7.2f}p  {bar}  {conf}"
        )

    divider()
    winner = subset.iloc[0]
    print(f"\n  ✦ PREDICTED TO FINISH HIGHEST:  {winner['Driver']}  ({winner['Team']})\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Q8 — DRIVER POSITION BRACKET
# ═══════════════════════════════════════════════════════════════════════════════

def q8_driver_bracket(drivers: pd.DataFrame, target_drivers: list[str] | None = None) -> None:
    df = drivers.copy()
    df["RaceScore"] = (
        df["ExpectedPoints"] * 0.70
      + df["AvgPoints"]       * 0.20
      + df["FP_Progression"].fillna(0).clip(-10, 10) * 0.5
    )
    df = df.sort_values("RaceScore", ascending=False).reset_index(drop=True)
    df["PredictedPos"] = range(1, len(df) + 1)

    if target_drivers:
        targets = df[df["Driver"].str.upper().isin([t.upper() for t in target_drivers])]
    else:
        targets = df

    header("Q8  ·  DRIVER POSITION BRACKET PREDICTIONS")
    print(f"\n  {'Driver':<16} {'Team':<14} {'Pred Pos':>9}  {'Bracket':<22}  DNF%  Exp.Pts")
    divider()

    for _, row in targets.iterrows():
        pos = int(row["PredictedPos"])
        dnf = row["DNF_Prob"]

        if dnf > 0.35:
            bracket = "NOT CLASSIFIED ⚠"
        elif pos <= 3:
            bracket = "PODIUM 🏆"
        elif pos <= 10:
            bracket = "4th – 10th 📍"
        else:
            bracket = "11th – 22nd"

        print(
            f"  {row['Driver']:<16} {row['Team']:<14} {'P'+str(pos):>9}  "
            f"{bracket:<22}  {dnf*100:>3.0f}%  {row['ExpectedPoints']:>6.2f}p"
        )
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Q9 — FASTEST LAP OF THE RACE
# ═══════════════════════════════════════════════════════════════════════════════

def q9_fastest_lap(drivers: pd.DataFrame) -> None:
    df = drivers.copy()

    max_fl = df["FastestLaps"].max() or 1
    min_delta = df["FP3_DeltaBest"].min()

    df["PaceScore"]  = (df["FP3_DeltaBest"] - min_delta).max() - (df["FP3_DeltaBest"] - min_delta)
    df["DegScore"]   = -df["FP2_DegSlope"].fillna(df["FP2_DegSlope"].median())
    df["FinishScore"] = 1 - df["DNF_Prob"]
    df["FLHistory"]   = df["FastestLaps"] / max_fl

    for col in ["PaceScore", "DegScore"]:
        rng = df[col].max() - df[col].min()
        df[col] = (df[col] - df[col].min()) / rng if rng > 0 else 0.5

    df["FLScore"] = (
        df["PaceScore"]    * 0.45
      + df["DegScore"]     * 0.25
      + df["FinishScore"]  * 0.20
      + df["FLHistory"]    * 0.10
    )
    df = df.sort_values("FLScore", ascending=False).reset_index(drop=True)

    header("Q9  ·  FASTEST LAP OF THE RACE")
    print(f"\n  {'Rank':<6} {'Driver':<16} {'Team':<14} {'FP3Δ':>8} {'DegSlope':>9} {'FL Hist':>7}  {'Score':>7}  {'Bar'}")
    divider()

    all_scores = df["FLScore"].tolist()
    for i in range(min(8, len(df))):
        row = df.iloc[i]
        bar = confidence_bar(row["FLScore"], all_scores[0])
        deg_raw = row["FP2_DegSlope"]
        if pd.isna(deg_raw): deg_raw = 0
        print(
            f"  {'P'+str(i+1):<6} {row['Driver']:<16} {row['Team']:<14} "
            f"{row['FP3_DeltaBest']:>7.4f}s  {deg_raw:>+8.3f}  {int(row['FastestLaps']):>7}  "
            f"{row['FLScore']:>7.4f}  {bar}"
        )

    divider()
    winner = df.iloc[0]
    print(f"\n  ✦ PREDICTED FASTEST LAP:  {winner['Driver']}  ({winner['Team']})")
    print(f"    FP3 pace: Δ{winner['FP3_DeltaBest']:.4f}s  |  Tyre deg slope: {winner['FP2_DegSlope']:+.3f}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Q10 — CLASSIFIED FINISHERS COUNT
# ═══════════════════════════════════════════════════════════════════════════════

def q10_classified_finishers(drivers: pd.DataFrame, circuit_dnf: float) -> None:
    circuit_expected_dnfs  = circuit_dnf * 22
    driver_expected_dnfs   = drivers["DNF_Prob"].sum()
    blended_expected_dnfs  = (circuit_expected_dnfs * 0.40) + (driver_expected_dnfs * 0.60)
    blended_expected_dnfs  = min(blended_expected_dnfs, 10)  

    classified = max(12, int(round(22 - blended_expected_dnfs)))

    p_under_15  = min(1.0, blended_expected_dnfs / 8.0)
    p_15_to_18  = max(0.0, 1.0 - p_under_15) * 0.60
    p_19_to_22  = max(0.0, 1.0 - p_under_15) * 0.40

    header("Q10  ·  CLASSIFIED FINISHERS COUNT")
    print(f"\n  Circuit DNF baseline rate : {circuit_dnf * 100:.1f}%")
    print(f"  Expected DNFs (circuit)   : {circuit_expected_dnfs:.2f} cars")
    print(f"  Expected DNFs (drivers)   : {driver_expected_dnfs:.2f} cars")
    print(f"  Blended expected DNFs     : {blended_expected_dnfs:.2f} cars")
    print()

    print(f"  {'Bracket':<22}  {'Probability':>12}  {'Bar'}")
    divider()
    print(f"  {'≤ 14 finishers':<22}  {p_under_15 * 100:>11.0f}%  {confidence_bar(p_under_15, 1.0)}")
    print(f"  {'15 – 18 finishers':<22}  {p_15_to_18 * 100:>11.0f}%  {confidence_bar(p_15_to_18, 1.0)}")
    print(f"  {'19 – 22 finishers':<22}  {p_19_to_22 * 100:>11.0f}%  {confidence_bar(p_19_to_22, 1.0)}")

    divider()
    print(f"\n  ✦ PREDICTED CLASSIFIED FINISHERS:  {classified} / 22 cars\n")


# ═══════════════════════════════════════════════════════════════════════════════
# VARIANT QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def v1_red_flag_count(circuit_dnf: float) -> None:
    header("V1  ·  RED FLAG COUNT FORECAST", "─")
    if circuit_dnf >= RF_THRESHOLD:
        verdict = "1 or more times"
        risk = "HIGH"
    elif circuit_dnf >= 0.085:
        verdict = "Possible — lean towards 1"
        risk = "MED"
    else:
        verdict = "0 times"
        risk = "LOW"
    print(f"\n  Circuit DNF rate: {circuit_dnf * 100:.1f}%  |  Red Flag risk: {risk}")
    print(f"  ✦ PREDICTED:  {verdict}\n")


def v2_all_22_complete_lap1(circuit_dnf: float, drivers: pd.DataFrame) -> None:
    header("V2  ·  WILL ALL 22 CARS COMPLETE LAP 1?", "─")
    high_risk_drivers = (drivers["DNF_Prob"] > 0.20).sum()
    lap1_risk = (circuit_dnf * 0.60) + (high_risk_drivers / 22) * 0.40
    verdict = "NO" if lap1_risk > (LAP1_INCIDENT_LIMIT / 0.15) * 0.4 else "YES"
    print(f"\n  Lap 1 incident risk index : {lap1_risk:.3f}")
    print(f"  High-risk DNF drivers in field: {high_risk_drivers}")
    print(f"  ✦ PREDICTED:  {verdict} — {'all 22 unlikely to survive lap 1' if verdict == 'NO' else 'field should complete lap 1 cleanly'}\n")


def v3_sprint_teams_scoring(constructors: pd.DataFrame) -> None:
    header("V3  ·  HOW MANY TEAMS SCORE IN THE SPRINT?", "─")
    df = constructors.sort_values("ExpectedPoints", ascending=False).reset_index(drop=True)
    print(f"\n  Sprint points go to top 8 drivers (P1–P8).\n"
          f"  Projected top 6 constructors likely to score:")
    for i, row in df.head(6).iterrows():
        print(f"    P{i+1}. {row['Constructor']}")
    likely_teams = min(6, len(df[df["ExpectedPoints"] > df["ExpectedPoints"].median()]))
    print(f"\n  ✦ PREDICTED TEAMS SCORING:  {likely_teams} teams\n")


def v4_sprint_highest_driver(drivers: pd.DataFrame) -> None:
    header("V4  ·  WHICH DRIVER FINISHES HIGHEST IN THE SPRINT?", "─")
    df = drivers.copy()
    df["SprintScore"] = (
        df["FP3_DeltaBest"]
      - df["FP_Progression"].fillna(0).clip(-5, 5) * 0.3
    )
    df = df.sort_values("SprintScore", ascending=True).reset_index(drop=True)
    winner = df.iloc[0]
    print(f"\n  Sprint outcome mirrors qualifying pace closely.")
    print(f"  ✦ PREDICTED SPRINT WINNER:  {winner['Driver']}  ({winner['Team']})")
    print(f"    Top 5: {' → '.join(df['Driver'].head(5).tolist())}\n")


def v5_championship_leader_drivers(drivers: pd.DataFrame, predict_data: dict, race_pts: int = 25) -> None:
    header("V5  ·  DRIVERS' CHAMPIONSHIP LEADER AFTER THIS RACE", "─")
    df = drivers[["Driver", "Team", "TotalPoints", "ExpectedPoints"]].copy()
    df["ProjectedTotal"] = df["TotalPoints"] + df["ExpectedPoints"] * (race_pts / df["ExpectedPoints"].max())
    df = df.sort_values("ProjectedTotal", ascending=False).reset_index(drop=True)
    print(f"\n  {'Rank':<6} {'Driver':<16} {'Current Pts':>12} {'Projected':>10}")
    divider()
    for i, row in df.head(6).iterrows():
        print(f"  {'P'+str(i+1):<6} {row['Driver']:<16} {row['TotalPoints']:>12.0f} {row['ProjectedTotal']:>9.1f}")
    leader = df.iloc[0]
    print(f"\n  ✦ PREDICTED CHAMPIONSHIP LEADER:  {leader['Driver']}  ({leader['Team']})\n")


def v6_driver_sprint_bracket(drivers: pd.DataFrame, driver_name: str) -> None:
    header(f"V6  ·  WHERE DOES {driver_name.upper()} FINISH IN THE SPRINT?", "─")
    df = drivers.copy()
    df["SprintScore"] = (
        df["FP3_DeltaBest"]
      - df["FP_Progression"].fillna(0).clip(-5, 5) * 0.3
    )
    df = df.sort_values("SprintScore", ascending=True).reset_index(drop=True)
    df["SprintPos"] = range(1, len(df) + 1)

    match = df[df["Driver"].str.upper() == driver_name.upper()]
    if match.empty:
        print(f"  [WARN] Driver '{driver_name}' not found in data.\n")
        return

    row = match.iloc[0]
    pos = int(row["SprintPos"])
    if   pos == 1:   bracket = "Sprint Win 🏆"
    elif pos <= 3:   bracket = "Sprint Podium 🥉"
    elif pos <= 8:   bracket = "Sprint Points (P4–P8)"
    else:            bracket = "Outside Sprint Points (P9+)"

    print(f"\n  {driver_name.upper()} predicted sprint position: P{pos}")
    print(f"  ✦ PREDICTED BRACKET:  {bracket}\n")


def v7_sc_first_10_laps(circuit_dnf: float) -> None:
    header("V7  ·  SAFETY CAR / VSC IN FIRST 10 LAPS?", "─")
    early_risk = circuit_dnf * 1.15   
    verdict = "YES" if early_risk >= SC_EARLY_THRESHOLD else "NO"
    print(f"\n  Circuit DNF rate (early multiplier): {early_risk * 100:.1f}%")
    print(f"  ✦ PREDICTED:  {verdict} — SC/VSC in first 10 laps is "
          f"{'likely' if verdict == 'YES' else 'unlikely'}\n")


def v8_championship_leader_constructors(constructors: pd.DataFrame) -> None:
    header("V8  ·  CONSTRUCTORS' CHAMPIONSHIP LEADER AFTER THIS RACE", "─")
    df = constructors[["Constructor", "TotalPoints", "ExpectedPoints"]].copy()
    max_exp = df["ExpectedPoints"].max() or 1
    df["ProjectedTotal"] = df["TotalPoints"] + df["ExpectedPoints"] * (43 / max_exp)  
    df = df.sort_values("ProjectedTotal", ascending=False).reset_index(drop=True)
    print(f"\n  {'Rank':<6} {'Constructor':<18} {'Current Pts':>12} {'Projected':>10}")
    divider()
    for i, row in df.head(6).iterrows():
        print(f"  {'P'+str(i+1):<6} {row['Constructor']:<18} {row['TotalPoints']:>12.0f} {row['ProjectedTotal']:>9.1f}")
    leader = df.iloc[0]
    print(f"\n  ✦ PREDICTED CONSTRUCTORS' LEADER:  {leader['Constructor']}\n")


def v9_teams_in_q3(drivers: pd.DataFrame) -> None:
    header("V9  ·  HOW MANY TEAMS WILL GET A DRIVER INTO Q3?", "─")
    df = drivers.copy()
    df["PoleScore"] = (
        df["FP3_DeltaBest"]
      - df["FP_Progression"].fillna(0).clip(-5, 5) * 0.3
    )
    df = df.sort_values("PoleScore", ascending=True).reset_index(drop=True)
    q3_drivers = df.head(10)
    teams_in_q3 = q3_drivers["Team"].nunique()

    print(f"\n  Predicted Q3 drivers: {', '.join(q3_drivers['Driver'].tolist())}")
    print(f"  Teams represented   : {', '.join(q3_drivers['Team'].unique().tolist())}")
    print(f"\n  ✦ PREDICTED TEAMS IN Q3:  {teams_in_q3} teams\n")


def v10_most_points_over_weekend(constructors: pd.DataFrame, meta: dict) -> None:
    header("V10  ·  WHICH TEAM SCORES MOST POINTS OVER THE WEEKEND?", "─")
    is_sprint = meta.get("race_round") in SPRINT_ROUNDS
    df = constructors.copy()
    sprint_bonus = 0.15 if is_sprint else 0.0

    df["WeekendScore"] = df["ExpectedPoints"] * (1 + sprint_bonus)
    df = df.sort_values("WeekendScore", ascending=False).reset_index(drop=True)

    sprint_note = "  (Sprint weekend — bonus points factored in)" if is_sprint else ""
    print(f"\n  {'Rank':<6} {'Constructor':<18} {'Weekend Score':>14}{sprint_note}")
    divider()
    for i, row in df.head(6).iterrows():
        bar = confidence_bar(row["WeekendScore"], df["WeekendScore"].max())
        print(f"  {'P'+str(i+1):<6} {row['Constructor']:<18} {row['WeekendScore']:>13.2f}  {bar}")

    divider()
    winner = df.iloc[0]
    print(f"\n  ✦ PREDICTED MOST POINTS:  {winner['Constructor']}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    meta, drivers, constructors, predict_data = load_all_data()

    track_name   = meta["track_name"]
    circuit_dnf  = meta["circuit_dnf_rate"]
    race_round   = meta["race_round"]
    is_sprint    = race_round in SPRINT_ROUNDS

    # ── Banner ─────────────────────────────────────────────────────────────────
    print("\n" + "█" * W)
    print(f"  F1 PREDICTION GAME SOLVER  —  {track_name.upper()}")
    print(f"  Race Round: {race_round}  |  Circuit DNF Rate: {circuit_dnf*100:.1f}%  |"
          f"  Sprint Weekend: {'YES' if is_sprint else 'NO'}")
    print("█" * W + "\n")

    # ── CORE QUESTIONS ─────────────────────────────────────────────────────────
    q1_race_top10(drivers, circuit_dnf)
    q2_pole_position(drivers)
    q3_qualifying_1v1s(drivers)
    q4_team_qualifying_segments(drivers)
    q5_fastest_pitstop(constructors)
    q6_safety_car(circuit_dnf, drivers)
    q7_midfield_4way(drivers, midfield_drivers=None)   
    q8_driver_bracket(drivers, target_drivers=None)    
    q9_fastest_lap(drivers)
    q10_classified_finishers(drivers, circuit_dnf)

    # ── VARIANT QUESTIONS ──────────────────────────────────────────────────────
    print("\n" + "█" * W)
    print(f"  VARIANT QUESTION ANSWERS  —  {track_name.upper()}")
    print("█" * W + "\n")

    v1_red_flag_count(circuit_dnf)
    v2_all_22_complete_lap1(circuit_dnf, drivers)

    if is_sprint:
        v3_sprint_teams_scoring(constructors)
        v4_sprint_highest_driver(drivers)
        v6_driver_sprint_bracket(drivers, driver_name="NORRIS")

    v5_championship_leader_drivers(drivers, predict_data)
    v7_sc_first_10_laps(circuit_dnf)
    v8_championship_leader_constructors(constructors)
    v9_teams_in_q3(drivers)
    v10_most_points_over_weekend(constructors, meta)

    # ── Footer ─────────────────────────────────────────────────────────────────
    print("█" * W)
    print(f"  END OF PREDICTIONS  —  {track_name.upper()}")
    print(f"  All signals derived from FP1/FP2/FP3 telemetry only.")
    print(f"  Run market_harvester.py → market_predict.py → weekend_predictor.py")
    print("█" * W + "\n")


if __name__ == "__main__":
    main()
