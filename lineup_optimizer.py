"""
lineup_optimizer.py
────────────────────────────────────────────────────────────────────────────────
Fantasy F1 Lineup Optimizer
Uses PuLP (linear programming) to select the maximum-scoring roster under
the official game constraints:
  • Budget cap  : $100.0 M
  • Drivers     : exactly 5
  • Constructors: exactly 2

Data source: Reads live predictions from market_predictions.json or runs market_predict.py
────────────────────────────────────────────────────────────────────────────────
"""

import json
import math
import sys
from pathlib import Path

try:
    import pulp
except ImportError:
    sys.exit(
        "[ERROR] PuLP is not installed.\n"
        "        Run:  pip install pulp\n"
        "        then retry."
    )

# ── 1. LOAD LIVE DATA FROM PREDICTIONS PATH ──────────────────────────────────

JSON_INPUT = Path("market_predictions.json")

driver_costs = {}
driver_expected_points = {}
constructor_costs = {}
constructor_expected_points = {}

try:
    # Strategy A: Read the freshly exported JSON file if it exists
    if JSON_INPUT.exists():
        with open(JSON_INPUT, "r") as f:
            payload = json.load(f)
        
        for name, stats in payload["drivers"].items():
            driver_costs[name] = stats["cost"]
            driver_expected_points[name] = stats["expected_points"]
            
        for name, stats in payload["constructors"].items():
            constructor_costs[name] = stats["cost"]
            constructor_expected_points[name] = stats["expected_points"]
            
        print(f"[INFO] Successfully loaded live data from {JSON_INPUT}\n")
        
    else:
        # Strategy B: Dynamic fallback - import pipeline and generate variables on the fly
        print("[INFO] market_predictions.json not found. Triggering market_predict pipeline...")
        from market_predict import run_pipeline
        driver_costs, driver_expected_points, constructor_costs, constructor_expected_points = run_pipeline()
        print("[INFO] Live data generated and loaded successfully.\n")

except Exception as e:
    # Strategy C: Hardcoded fallback if everything else fails
    print(f"[WARN] Error loading live predictions ({e}). Running with historical demo data.\n")

    driver_costs = {
        "M. Verstappen": 30.0, "L. Norris": 27.0, "C. Leclerc": 24.5, "C. Sainz": 22.0,
        "G. Russell": 21.5, "L. Hamilton": 20.0, "F. Alonso": 17.5, "S. Perez": 16.0,
        "O. Piastri": 18.5, "L. Stroll": 10.0, "E. Ocon": 9.5, "P. Gasly": 9.0,
        "V. Bottas": 8.5, "N. Hulkenberg": 9.0, "Y. Tsunoda": 8.0, "A. Albon": 8.5
    }
    driver_expected_points = {
        "M. Verstappen": 62.4, "L. Norris": 54.8, "C. Leclerc": 50.1, "C. Sainz": 47.3,
        "G. Russell": 45.0, "L. Hamilton": 43.5, "F. Alonso": 38.9, "S. Perez": 35.2,
        "O. Piastri": 40.6, "L. Stroll": 20.1, "E. Ocon": 18.4, "P. Gasly": 17.9,
        "V. Bottas": 15.3, "N. Hulkenberg": 19.8, "Y. Tsunoda": 16.7, "A. Albon": 17.2
    }
    constructor_costs = {
        "Red Bull": 30.5, "Ferrari": 26.0, "Mercedes": 23.5, "McLaren": 22.0, "Aston Martin": 14.5
    }
    constructor_expected_points = {
        "Red Bull": 88.6, "Ferrari": 74.3, "Mercedes": 70.1, "McLaren": 68.4, "Aston Martin": 46.2
    }


# ── 2. SANITIZE & VALIDATE DATA INTEGRITY ─────────────────────────────────────

def _sanitize_and_validate(costs: dict, points: dict, label: str) -> tuple[dict, dict]:
    cleaned_costs = {}
    cleaned_points = {}

    missing = set(costs) ^ set(points)
    if missing:
        sys.exit(f"[ERROR] {label} key mismatch between costs and points:\n        {missing}")

    for k, v in costs.items():
        val = float(v)
        cleaned_costs[k] = 0.0 if (math.isnan(val) or math.isinf(val)) else val

    for k, v in points.items():
        val = float(v)
        cleaned_points[k] = 0.0 if (math.isnan(val) or math.isinf(val)) else val

    return cleaned_costs, cleaned_points

driver_costs, driver_expected_points = _sanitize_and_validate(driver_costs, driver_expected_points, "Drivers")
constructor_costs, constructor_expected_points = _sanitize_and_validate(constructor_costs, constructor_expected_points, "Constructors")

drivers = list(driver_costs.keys())
constructors = list(constructor_costs.keys())

# ── 3. GAME CONSTRAINTS ───────────────────────────────────────────────────────

BUDGET_CAP = 100.0  
NUM_DRIVERS = 5
NUM_CONSTRUCTORS = 2

# ── 4. BUILD THE LINEAR PROGRAMMING MODEL ────────────────────────────────────

model = pulp.LpProblem("Fantasy_F1_Lineup_Optimizer", pulp.LpMaximize)

# Safe naming convention clean up for PuLP tokens
def sanitize_var_name(name):
    return name.replace(' ', '_').replace('.', '').replace('-', '_').replace('(', '').replace(')', '')

driver_vars = {d: pulp.LpVariable(f"driver_{sanitize_var_name(d)}", cat="Binary") for d in drivers}
chip_vars = {d: pulp.LpVariable(f"chip_{sanitize_var_name(d)}", cat="Binary") for d in drivers}
constructor_vars = {c: pulp.LpVariable(f"constructor_{sanitize_var_name(c)}", cat="Binary") for c in constructors}

# Object function: Maximize Expected Points
model += (
    pulp.lpSum(driver_expected_points[d] * driver_vars[d] for d in drivers)
    + pulp.lpSum(driver_expected_points[d] * chip_vars[d] for d in drivers)
    + pulp.lpSum(constructor_expected_points[c] * constructor_vars[c] for c in constructors)
), "Total_Expected_Points"

# Constraints
model += (pulp.lpSum(driver_costs[d] * driver_vars[d] for d in drivers) + 
          pulp.lpSum(constructor_costs[c] * constructor_vars[c] for c in constructors) <= BUDGET_CAP), "Budget_Cap"

model += (pulp.lpSum(driver_vars[d] for d in drivers) == NUM_DRIVERS), "Exactly_5_Drivers"
model += (pulp.lpSum(constructor_vars[c] for c in constructors) == NUM_CONSTRUCTORS), "Exactly_2_Constructors"

for d in drivers:
    model += (chip_vars[d] <= driver_vars[d]), f"Chip_Constraint_{sanitize_var_name(d)}"
model += (pulp.lpSum(chip_vars[d] for d in drivers) == 1), "Exactly_One_Chip"

# ── 5. SOLVE ──────────────────────────────────────────────────────────────────

solver = pulp.PULP_CBC_CMD(msg=False)
status = model.solve(solver)

if pulp.LpStatus[model.status] != "Optimal":
    sys.exit(f"[ERROR] Solver could not find an optimal solution. Status: {pulp.LpStatus[model.status]}")

# ── 6. EXTRACT RESULTS ────────────────────────────────────────────────────────

selected_drivers = [d for d in drivers if pulp.value(driver_vars[d]) > 0.5]
selected_constructors = [c for c in constructors if pulp.value(constructor_vars[c]) > 0.5]
chip_driver = next((d for d in drivers if pulp.value(chip_vars[d]) > 0.5), None)

total_cost = sum(driver_costs[d] for d in selected_drivers) + sum(constructor_costs[c] for c in selected_constructors)
total_points = (
    sum(driver_expected_points[d] for d in selected_drivers)
    + (driver_expected_points[chip_driver] if chip_driver else 0.0)
    + sum(constructor_expected_points[c] for c in selected_constructors)
)

selected_drivers.sort(key=lambda d: driver_expected_points[d], reverse=True)
selected_constructors.sort(key=lambda c: constructor_expected_points[c], reverse=True)

# ── 7. PRINT RESULTS ──────────────────────────────────────────────────────────

W = 58  
def divider(char="─"): print(char * W)

divider("═")
print(f"{'FANTASY F1 – OPTIMISED RACE WEEKEND LINEUP':^{W}}")
divider("═")

print(f"\n  DRIVERS")
divider()
print(f"  {'#':<4} {'Name':<22} {'Cost ($M)':>10} {'Exp. Pts':>10}")
divider()

for rank, d in enumerate(selected_drivers, 1):
    name_display = f"{d} (2X)" if d == chip_driver else d
    print(f"  {rank:<4} {name_display:<22} {'${:>6.1f}M'.format(driver_costs[d]):>10} {driver_expected_points[d]:>9.1f}p")

driver_subtotal_cost = sum(driver_costs[d] for d in selected_drivers)
driver_subtotal_points = sum(driver_expected_points[d] for d in selected_drivers) + (driver_expected_points[chip_driver] if chip_driver else 0.0)

divider()
print(f"  {'Driver Subtotals':<26}{'${:>6.1f}M'.format(driver_subtotal_cost):>10} {driver_subtotal_points:>9.1f}p")

print(f"\n  CONSTRUCTORS")
divider()
print(f"  {'#':<4} {'Name':<22} {'Cost ($M)':>10} {'Exp. Pts':>10}")
divider()

for rank, c in enumerate(selected_constructors, 1):
    print(f"  {rank:<4} {c:<22} {'${:>6.1f}M'.format(constructor_costs[c]):>10} {constructor_expected_points[c]:>9.1f}p")

con_subtotal_cost = sum(constructor_costs[c] for c in selected_constructors)
con_subtotal_points = sum(constructor_expected_points[c] for c in selected_constructors)

divider()
print(f"  {'Constructor Subtotals':<26}{'${:>6.1f}M'.format(con_subtotal_cost):>10} {con_subtotal_points:>9.1f}p")

remaining_budget = BUDGET_CAP - total_cost
print()
divider("═")
print(f"  {'TOTAL LINEUP COST':<26}{'${:>6.1f}M'.format(total_cost):>10}")
print(f"  {'REMAINING BUDGET':<26}{'${:>6.1f}M'.format(remaining_budget):>10}")
print(f"  {'TOTAL PROJECTED POINTS':<26}{total_points:>10.1f}p")
divider("═")
print(f"\n  Budget used: ${total_cost:.1f}M / ${BUDGET_CAP:.1f}M  ({(total_cost / BUDGET_CAP) * 100:.1f}% of cap)\n")
