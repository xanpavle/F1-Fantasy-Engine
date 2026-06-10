
# 🏎️ F1 Fantasy Engine 

A fully automated Formula 1 fantasy toolkit powered by live FastF1 telemetry, 
machine learning predictions, linear programming roster optimization, and 
Monte Carlo race simulation — all derived strictly from Free Practice data 
before qualifying locks in.

Built for the 2026 F1 season. Answers every known Prediction Game question 
and optimizes your fantasy lineup to the mathematical maximum within budget.
## Features

### 🔄 Live Telemetry Harvesting (`market_harvester.py`)
- Pulls FP1, FP2, FP3 session data via FastF1 with automatic caching
- Auto-fallback to last completed event when run mid-week
- Calculates tyre degradation slope via linear regression on stint data
- Engineers ML-ready features: `FP_Progression`, `Overtake_Efficiency`, `Hype_Ratio`

### 🧠 Predictive Scoring Engine (`market_predict.py`)
- DNF probability model blending driver history, constructor history, and circuit base rate
- Track archetype system (street vs. permanent circuit) with configurable DNF rates
- Constructor points derived from aggregated driver predictions with pitstop efficiency multiplier
- Outputs structured `market_predictions.json` and exports Python-level dicts for the optimizer

### 🏆 Lineup Optimizer (`lineup_optimizer.py`)
- Linear programming via PuLP (CBC solver) — finds the provably optimal roster
- Enforces all official fantasy constraints: 5 drivers, 2 constructors, $100M cap, 1 Turbo chip
- NaN/Inf sanitization layer prevents solver failures on corrupted telemetry
- Clean formatted terminal output with cost breakdown and remaining budget

### 🎲 Monte Carlo Race Simulator (`race_simulator.py`)
- 10,000 lap-by-lap race simulations with stochastic DNF and overtake rolls
- Lap 1 heightened risk multipliers (3× DNF, 2.5× Safety Car)
- Track-specific overtake difficulty (Monaco/Singapore vs. open circuits)
- Outputs win %, podium %, top-10 %, DNF % per driver, plus expected time gaps

### 📋 Prediction Game Solver (`weekend_predictor.py`)
- Answers all 10 core F1 Prediction Game questions (Q1–Q10)
- Covers 10 variant questions (V1–V10) including sprint weekends
- Confidence bars and HIGH/MED/LOW labels on all predictions
- Championship standings projection after the race
## Installation

**Requirements:** Python 3.10 or higher

### 1. Clone the repository

```bash
git clone https://github.com/xanpavle/f1-fantasy-engine.git
cd f1-fantasy-suite
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install fastf1 pandas numpy scipy pulp
```

### 4. Add your fantasy CSV files

Place your league-specific CSVs in the project root:
- `drivers_fantasy.csv` — driver roster with Cost, AvgPoints, DNFs, etc.
- `constructors_fantasy.csv` — constructor roster with matching columns

### 5. Execution Paths — Run Order by Goal

Depending on what you want, follow the matching path below.
**Always start with Steps 1 and 2** — every other script depends on them.

---

#### 🔁 Required First (every time, every path)

```bash
python market_harvester.py   # Step 1 — fetch live FP telemetry
python market_predict.py     # Step 2 — generate predictions + costs
```

---

#### 🏆 Path A — Optimise Your Fantasy Lineup

*Best lineup within $100M, with Turbo chip assigned automatically.*

```bash
python market_harvester.py
python market_predict.py
python lineup_optimizer.py
```

---

#### 📋 Path B — Answer the F1 Prediction Game Questions

*All Q1–Q10 core questions + V1–V10 variants (pole, top-10, pitstop, SC, etc.)*

```bash
python market_harvester.py
python market_predict.py
python weekend_predictor.py
```

---

#### 🎲 Path C — Run the Monte Carlo Race Simulation

*10,000 simulated races → win %, podium %, DNF % per driver + time gaps.*

```bash
python market_harvester.py
python market_predict.py
python race_simulator.py
```

---

#### 🚀 Path D — Full Suite (everything at once)

*Run all tools back to back for the complete picture.*

```bash
python market_harvester.py
python market_predict.py
python lineup_optimizer.py
python weekend_predictor.py
python race_simulator.py
```

---

> **Tip:** `market_harvester.py` caches session data locally in `.fastf1_cache/`.  
> You only need to re-run it if the race weekend changes or new FP data becomes available.  
> Steps 3–5 can be re-run as many times as you like without re-fetching telemetry.
## Optimizations

### FastF1 Caching
All session data is cached locally in `.fastf1_cache/` after the first fetch. 
Subsequent runs of `market_harvester.py` for the same event are near-instant. 
Do not delete this folder between pipeline runs on the same weekend.

### Solver Performance
The PuLP CBC solver runs silently (`msg=False`) and solves the integer program 
in under a second for the current pool size (22 drivers, 11 constructors). 
No performance tuning is needed at this scale.

### Monte Carlo Speed
10,000 simulations complete in approximately 5–15 seconds depending on hardware. 
NumPy vectorized per-lap DNF rolls replace Python loops where possible. 
Reduce `N_SIMULATIONS` to 1,000 for rapid testing; increase to 50,000 for 
tighter probability confidence intervals.

### Known Limitations
- The scoring model is an analytical heuristic, not a trained ML model. 
  Replacing `train_predictive_engine()` with a real XGBoost/LightGBM model 
  trained on historical seasons would significantly improve accuracy.
- Qualifying pace is approximated from FP3 delta times. Actual qualifying 
  simulation is not implemented.
- Constructor team-name normalization (e.g. "RACING BULLS" → "RACINGBULLS") 
  may need updating each season as team names change.
  ## FAQ

**Q: Do I need an F1 API key or paid subscription to use this?**  
A: No. All telemetry is fetched for free via FastF1, which uses the official 
F1 timing API under the hood. No credentials are required.

---

**Q: When should I run the pipeline each race weekend?**  
A: Run after FP2 completes (Saturday morning ideally after FP3). 
Running mid-week triggers the auto-fallback to the previous race's data, 
which is still useful for testing but won't reflect the current circuit.

---

**Q: The harvester fails with a session not available error. What do I do?**  
A: This is expected mid-week. The script auto-detects this and falls back 
to the most recent completed event. If it fails completely, check your 
internet connection and that FastF1 cache isn't corrupted (delete `.fastf1_cache/` and retry).

---

**Q: My lineup optimizer returns an infeasible/no optimal solution error.**  
A: This means the budget constraints can't be satisfied with the current 
driver pool. Check that your `drivers_fantasy.csv` has enough low-cost 
drivers to form a valid 5+2 roster under $100M. Also verify no cost values 
are zero or NaN.

---

**Q: The predicted points look wrong or very low for some drivers.**  
A: Check for NaN values in your input CSVs, especially the FP delta columns. 
The imputation layer in `market_predict.py` fills missing values with team 
or global medians — if an entire team's telemetry is missing, those drivers 
will score near the median baseline. Re-running `market_harvester.py` usually fixes this.

---

**Q: Can I use this for historical backtesting?**  
A: Yes. Change `year` in `market_harvester.py` and set `DEFAULT_TRACK` in 
`market_predict.py` to any completed race. FastF1 caches historical sessions 
just like live ones. Compare `expected_points` output against real race results.

---

**Q: How accurate are the predictions?**  
A: The model is purely FP-telemetry driven with no qualifying data, so 
accuracy varies. It is strongest at predicting relative pace order and 
worst at one-off events (reliability failures, weather, strategy calls). 
Think of it as a strong informed prior, not a guarantee.

---

**Q: Can I change the fantasy budget cap or roster size?**  
A: Yes. In `lineup_optimizer.py`, edit `BUDGET_CAP`, `NUM_DRIVERS`, 
and `NUM_CONSTRUCTORS` at the top of the file. The LP model adapts automatically.

---

**Q: What's the Turbo Driver (2X chip)?**  
A: One driver in your lineup can score double points. The optimizer 
automatically assigns the chip to whichever driver maximizes total 
expected points. It's constrained to only be applied to a driver 
already in your selected 5.

---

**Q: Will this work for the 2027 season?**  
A: The main things to update each season are: `SPRINT_ROUNDS` in 
`weekend_predictor.py`, team name normalization in `market_harvester.py`, 
and `TRACK_CALENDAR` entries in `market_predict.py`. The core pipeline 
is season-agnostic.
## Authors

- [@xanpavle](https://www.github.com/xanpavle)


## Badges

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![FastF1](https://img.shields.io/badge/FastF1-Live%20Telemetry-red)
![PuLP](https://img.shields.io/badge/PuLP-LP%20Optimizer-orange)
![Status](https://img.shields.io/badge/Season-2026%20Active-brightgreen)
![Monte Carlo](https://img.shields.io/badge/Simulations-10%2C000%20iterations-purple)

## Acknowledgements

 - [FastF1](https://github.com/theOehrly/Fast-F1) — the open-source library 
  that makes live F1 telemetry accessible in Python. This entire project 
  would not exist without it.
- [PuLP](https://github.com/coin-or/pulp) — linear programming solver used 
  to find the provably optimal fantasy lineup under budget constraints.
- [NumPy](https://numpy.org/) & [pandas](https://pandas.pydata.org/) — the 
  backbone of every data pipeline in this project.
- [SciPy](https://scipy.org/) — used for tyre degradation slope regression 
  in market_harvester.py.
- The F1 Fantasy community for surfacing edge-case scoring rules that shaped 
  the constraint model.
  

## API Reference

This project does not expose an HTTP API. Instead, each module exports 
Python-level dictionaries consumed by the next step in the pipeline.

### `market_predict.py` — Module Exports

These are imported directly by `lineup_optimizer.py`:

| Export | Type | Description |
|---|---|---|
| `driver_costs` | `dict[str, float]` | Driver name → cost in $M |
| `driver_expected_points` | `dict[str, float]` | Driver name → predicted fantasy points |
| `constructor_costs` | `dict[str, float]` | Constructor name → cost in $M |
| `constructor_expected_points` | `dict[str, float]` | Constructor name → predicted fantasy points |

### `market_predictions.json` — Output Schema

```json
{
  "metadata": {
    "track_name": "Monaco Grand Prix",
    "is_street": true,
    "circuit_dnf_rate": 0.18,
    "formula": "ExpectedPoints = (1 - DNF_Probability) × Base_Predicted_Points"
  },
  "drivers": {
    "DriverName": {
      "cost": 24.5,
      "base_points": 52.1,
      "dnf_probability": 0.14,
      "expected_points": 44.8
    }
  },
  "constructors": {
    "ConstructorName": {
      "cost": 26.0,
      "base_points": 88.6,
      "expected_points": 96.3
    }
  }
}
```

### Key Constants (configurable per file)

| Constant | File | Default | Description |
|---|---|---|---|
| `BUDGET_CAP` | lineup_optimizer.py | `100.0` | Fantasy budget in $M |
| `N_SIMULATIONS` | race_simulator.py | `10000` | Monte Carlo iterations |
| `TOTAL_LAPS` | race_simulator.py | `70` | Simulated race length |
| `DEFAULT_TRACK` | market_predict.py | `"Monaco Grand Prix"` | Active track config |
| `W_DRIVER_DNF` | market_predict.py | `0.40` | DNF weight: driver history |
| `W_TEAM_DNF` | market_predict.py | `0.40` | DNF weight: constructor history |
| `W_CIRCUIT` | market_predict.py | `0.20` | DNF weight: circuit base rate |



## Prediction Model Formula
### ExpectedPoints = (1 - DNF_Probability) × Base_Predicted_Points

Where:
- **Base_Predicted_Points** = `AvgPoints × pace_multiplier + (FP_Progression × 0.5)`
- **pace_multiplier** = `clip(1.3 - (avg_delta × 0.15), 0.5, 1.5)`
- **DNF_Probability** = `(driver_dnf × 0.40) + (team_dnf × 0.40) + (circuit_rate × 0.20)`

### Constructor Points Formula
ConstructorExpectedPoints = Σ(both_drivers_ExpPts) × 1.45 × pitstop_efficiency

Where `pitstop_efficiency = clip(1.0 + FastestPitstops × 0.01, 0.9, 1.1)`

### Track Archetypes

| Track | Street Circuit | Base DNF Rate |
|---|---|---|
| Monaco Grand Prix | ✅ | 18% |
| Melbourne Grand Prix | ✅ | 12% |
| Silverstone Grand Prix | ❌ | 8% |
| Spa Grand Prix | ❌ | 9% |

Add entries to `TRACK_CALENDAR` in `market_predict.py` to expand coverage.

### Fantasy Lineup Constraints

| Rule | Value |
|---|---|
| Budget cap | $100.0M |
| Drivers | Exactly 5 |
| Constructors | Exactly 2 |
| Turbo Driver (2X chip) | Exactly 1 (must be in selected 5) |

### Sprint Weekend Rounds (2026)

Rounds: 2, 6, 9, 18, 20, 22 — update `SPRINT_ROUNDS` in `weekend_predictor.py` 
as the calendar firms up.

