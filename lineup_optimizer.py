"""
lineup_optimizer.py
────────────────────────────────────────────────────────────────────────────────
Fantasy F1 Lineup Optimizer
Uses PuLP (linear programming) to select the maximum-scoring roster under
the official game constraints:
  • Budget cap  : $100.0 M
  • Drivers     : exactly 5
  • Constructors: exactly 2

Data source: market_predict.py  (driver_costs, driver_expected_points,
                                  constructor_costs, constructor_expected_points)
────────────────────────────────────────────────────────────────────────────────
"""

import math
import sys

try:
    import pulp
except ImportError:
    sys.exit(
        "[ERROR] PuLP is not installed.\n"
        "        Run:  pip install pulp\n"
        "        then retry."
    )

# ── 1. LOAD DATA FROM market_predict.py ──────────────────────────────────────

try:
    from market_predict import (
        constructor_costs,
        constructor_expected_points,
        driver_costs,
        driver_expected_points,
    )

    print("[INFO] Data loaded successfully from market_predict.py\n")

except ImportError:
    # ── FALLBACK DEMO DATA ────────────────────────────────────────────────────
    # Mirrors the dictionary schema that market_predict.py is expected to export.
    print(
        "[WARN] market_predict.py not found – running with built-in demo data.\n"
    )

    driver_costs: dict[str, float] = {
        "M. Verstappen": 30.0,
        "L. Norris": 27.0,
        "C. Leclerc": 24.5,
        "C. Sainz": 22.0,
        "G. Russell": 21.5,
        "L. Hamilton": 20.0,
        "F. Alonso": 17.5,
        "S. Perez": 16.0,
        "O. Piastri": 18.5,
        "L. Stroll": 10.0,
        "E. Ocon": 9.5,
        "P. Gasly": 9.0,
        "V. Bottas": 8.5,
        "N. Hulkenberg": 9.0,
        "Y. Tsunoda": 8.0,
        "A. Albon": 8.5,
        "Z. Guanyu": 7.5,
        "K. Magnussen": 7.0,
        "L. Sargeant": 6.5,
        "N. De Vries": 6.0,
    }

    driver_expected_points: dict[str, float] = {
        "M. Verstappen": 62.4,
        "L. Norris": 54.8,
        "C. Leclerc": 50.1,
        "C. Sainz": 47.3,
        "G. Russell": 45.0,
        "L. Hamilton": 43.5,
        "F. Alonso": 38.9,
        "S. Perez": 35.2,
        "O. Piastri": 40.6,
        "L. Stroll": 20.1,
        "E. Ocon": 18.4,
        "P. Gasly": 17.9,
        "V. Bottas": 15.3,
        "N. Hulkenberg": 19.8,
        "Y. Tsunoda": 16.7,
        "A. Albon": 17.2,
        "Z. Guanyu": 12.0,
        "K. Magnussen": 11.5,
        "L. Sargeant": 9.8,
        "N. De Vries": 8.6,
    }

    constructor_costs: dict[str, float] = {
        "Red Bull": 30.5,
        "Ferrari": 26.0,
        "Mercedes": 23.5,
        "McLaren": 22.0,
        "Aston Martin": 14.5,
        "Alpine": 10.5,
        "Williams": 9.5,
        "AlphaTauri": 8.5,
        "Alfa Romeo": 8.0,
        "Haas": 7.5,
    }

    constructor_expected_points: dict[str, float] = {
        "Red Bull": 88.6,
        "Ferrari": 74.3,
        "Mercedes": 70.1,
        "McLaren": 68.4,
        "Aston Martin": 46.2,
        "Alpine": 28.5,
        "Williams": 24.8,
        "AlphaTauri": 22.1,
        "Alfa Romeo": 20.3,
        "Haas": 17.4,
    }

# ── 2. SANITIZE & VALIDATE DATA INTEGRITY ─────────────────────────────────────


def _sanitize_and_validate(costs: dict, points: dict, label: str) -> tuple[dict, dict]:
    """Scans and transforms any NaN or Inf telemetry leaks into clean numeric baselines."""
    cleaned_costs = {}
    cleaned_points = {}

    # Handle key mismatches first
    missing = set(costs) ^ set(points)
    if missing:
        sys.exit(
            f"[ERROR] {label} key mismatch between costs and expected_points:\n"
            f"        {missing}"
        )

    # Sanitize costs dictionary values
    for k, v in costs.items():
        try:
            val = float(v)
            if math.isnan(val) or math.isinf(val):
                print(f"[WARN] Cleaned NaN/Inf detected in {label} Cost for '{k}'. Set to 0.0.")
                cleaned_costs[k] = 0.0
            else:
                cleaned_costs[k] = val
        except (ValueError, TypeError):
            print(f"[WARN] Cleaned invalid numeric format in {label} Cost for '{k}'. Set to 0.0.")
            cleaned_costs[k] = 0.0

    # Sanitize expected points dictionary values
    for k, v in points.items():
        try:
            val = float(v)
            if math.isnan(val) or math.isinf(val):
                print(f"[WARN] Cleaned NaN/Inf detected in {label} Expected Points for '{k}'. Set to 0.0.")
                cleaned_points[k] = 0.0
            else:
                cleaned_points[k] = val
        except (ValueError, TypeError):
            print(f"[WARN] Cleaned invalid numeric format in {label} Expected Points for '{k}'. Set to 0.0.")
            cleaned_points[k] = 0.0

    return cleaned_costs, cleaned_points


# Execute telemetry data cleansing
driver_costs, driver_expected_points = _sanitize_and_validate(
    driver_costs, driver_expected_points, "Drivers"
)
constructor_costs, constructor_expected_points = _sanitize_and_validate(
    constructor_costs, constructor_expected_points, "Constructors"
)

drivers = list(driver_costs.keys())
constructors = list(constructor_costs.keys())

# ── 3. GAME CONSTRAINTS ───────────────────────────────────────────────────────

BUDGET_CAP = 100.0  # $M
NUM_DRIVERS = 5
NUM_CONSTRUCTORS = 2

# ── 4. BUILD THE LINEAR PROGRAMMING MODEL ────────────────────────────────────

model = pulp.LpProblem("Fantasy_F1_Lineup_Optimizer", pulp.LpMaximize)

# Binary decision variables: 1 = selected, 0 = not selected
driver_vars = {
    d: pulp.LpVariable(
        f"driver_{d.replace(' ', '_').replace('.', '')}", cat="Binary"
    )
    for d in drivers
}
chip_vars = {
    d: pulp.LpVariable(
        f"chip_{d.replace(' ', '_').replace('.', '')}", cat="Binary"
    )
    for d in drivers
}
constructor_vars = {
    c: pulp.LpVariable(f"constructor_{c.replace(' ', '_')}", cat="Binary")
    for c in constructors
}

# ── 4a. OBJECTIVE FUNCTION – maximise total expected points ───────────────────
model += (
    pulp.lpSum(driver_expected_points[d] * driver_vars[d] for d in drivers)
    + pulp.lpSum(driver_expected_points[d] * chip_vars[d] for d in drivers)
    + pulp.lpSum(
        constructor_expected_points[c] * constructor_vars[c]
        for c in constructors
    )
), "Total_Expected_Points"

# ── 4b. CONSTRAINT 1 – budget cap ────────────────────────────────────────────
model += (
    pulp.lpSum(driver_costs[d] * driver_vars[d] for d in drivers)
    + pulp.lpSum(constructor_costs[c] * constructor_vars[c] for c in constructors)
    <= BUDGET_CAP
), "Budget_Cap"

# ── 4c. CONSTRAINT 2 – exactly 5 drivers ─────────────────────────────────────
model += (
    pulp.lpSum(driver_vars[d] for d in drivers) == NUM_DRIVERS
), "Exactly_5_Drivers"

# ── 4d. CONSTRAINT 3 – exactly 2 constructors ────────────────────────────────
model += (
    pulp.lpSum(constructor_vars[c] for c in constructors) == NUM_CONSTRUCTORS
), "Exactly_2_Constructors"

# ── 4e. CONSTRAINT 4 – chip driver constraints ────────────────────────────────
for d in drivers:
    model += (chip_vars[d] <= driver_vars[d]), f"Chip_Constraint_{d.replace(' ', '_').replace('.', '')}"

model += (pulp.lpSum(chip_vars[d] for d in drivers) == 1), "Exactly_One_Chip"

# ── 5. SOLVE ──────────────────────────────────────────────────────────────────

solver = pulp.PULP_CBC_CMD(msg=False)  # suppress solver verbose output
status = model.solve(solver)

if pulp.LpStatus[model.status] != "Optimal":
    sys.exit(
        f"[ERROR] Solver could not find an optimal solution.\n"
        f"        Status: {pulp.LpStatus[model.status]}\n"
        "        Check that the budget cap and pool size are feasible."
    )

# ── 6. EXTRACT RESULTS ────────────────────────────────────────────────────────

selected_drivers = [d for d in drivers if pulp.value(driver_vars[d]) > 0.5]
selected_constructors = [
    c for c in constructors if pulp.value(constructor_vars[c]) > 0.5
]
chip_driver = next((d for d in drivers if pulp.value(chip_vars[d]) > 0.5), None)

total_cost = sum(driver_costs[d] for d in selected_drivers) + sum(
    constructor_costs[c] for c in selected_constructors
)
total_points = (
    sum(driver_expected_points[d] for d in selected_drivers)
    + (driver_expected_points[chip_driver] if chip_driver else 0.0)
    + sum(constructor_expected_points[c] for c in selected_constructors)
)

# Sort each group by expected points descending for cleaner display
selected_drivers.sort(key=lambda d: driver_expected_points[d], reverse=True)
selected_constructors.sort(
    key=lambda c: constructor_expected_points[c], reverse=True
)

# ── 7. PRINT RESULTS ──────────────────────────────────────────────────────────

W = 58  # terminal width


def divider(char="─"):
    print(char * W)


divider("═")
print(f"{'FANTASY F1 – OPTIMISED RACE WEEKEND LINEUP':^{W}}")
divider("═")

# ── Drivers ───────────────────────────────────────────────────────────────────
print(f"\n  {'DRIVERS':}")
divider()
print(f"  {'#':<4} {'Name':<22} {'Cost ($M)':>10} {'Exp. Pts':>10}")
divider()

for rank, d in enumerate(selected_drivers, 1):
    name_display = f"{d} (2X)" if d == chip_driver else d
    print(
        f"  {rank:<4} {name_display:<22} "
        f"{'${:>6.1f}M'.format(driver_costs[d]):>10} "
        f"{driver_expected_points[d]:>9.1f}p"
    )

driver_subtotal_cost = sum(driver_costs[d] for d in selected_drivers)
driver_subtotal_points = sum(driver_expected_points[d] for d in selected_drivers)
if chip_driver:
    driver_subtotal_points += driver_expected_points[chip_driver]

divider()
print(
    f"  {'Driver Subtotals':<26}"
    f"{'${:>6.1f}M'.format(driver_subtotal_cost):>10} "
    f"{driver_subtotal_points:>9.1f}p"
)

# ── Constructors ──────────────────────────────────────────────────────────────
print(f"\n  {'CONSTRUCTORS':}")
divider()
print(f"  {'#':<4} {'Name':<22} {'Cost ($M)':>10} {'Exp. Pts':>10}")
divider()

for rank, c in enumerate(selected_constructors, 1):
    print(
        f"  {rank:<4} {c:<22} "
        f"{'${:>6.1f}M'.format(constructor_costs[c]):>10} "
        f"{constructor_expected_points[c]:>9.1f}p"
    )

con_subtotal_cost = sum(constructor_costs[c] for c in selected_constructors)
con_subtotal_points = sum(
    constructor_expected_points[c] for c in selected_constructors
)

divider()
print(
    f"  {'Constructor Subtotals':<26}"
    f"{'${:>6.1f}M'.format(con_subtotal_cost):>10} "
    f"{con_subtotal_points:>9.1f}p"
)

# ── Grand totals ──────────────────────────────────────────────────────────────
remaining_budget = BUDGET_CAP - total_cost

print()
divider("═")
print(f"  {'TOTAL LINEUP COST':<26}" f"{'${:>6.1f}M'.format(total_cost):>10}")
print(
    f"  {'REMAINING BUDGET':<26}"
    f"{'${:>6.1f}M'.format(remaining_budget):>10}"
)
print(f"  {'TOTAL PROJECTED POINTS':<26}" f"{total_points:>10.1f}p")
divider("═")
print(
    f"\n  Budget used: ${total_cost:.1f}M / ${BUDGET_CAP:.1f}M  "
    f"({(total_cost / BUDGET_CAP) * 100:.1f}% of cap)\n"
)
