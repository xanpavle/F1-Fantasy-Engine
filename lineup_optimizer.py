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

JSON_INPUT = Path("market_predictions.json")

driver_costs = {}
driver_expected_points = {}
constructor_costs = {}
constructor_expected_points = {}

# A helper dictionary to map uppercase raw dataset keys to clean display names
NAME_DISPLAY_MAP = {
    "VERSTAPPEN": "M. Verstappen", "NORRIS": "L. Norris", "LECLERC": "C. Leclerc",
    "SAINZ": "C. Sainz", "RUSSELL": "G. Russell", "HAMILTON": "L. Hamilton",
    "ALONSO": "F. Alonso", "PEREZ": "S. Perez", "PIASTRI": "O. Piastri",
    "STROLL": "L. Stroll", "OCON": "E. Ocon", "GASLY": "P. Gasly",
    "BOTTAS": "V. Bottas", "HULKENBERG": "N. Hulkenberg", "TSUNODA": "Y. Tsunoda",
    "ALBON": "A. Albon", "ANTONELLI": "K. Antonelli", "LAWSON": "L. Lawson",
    "MAGNUSSEN": "K. Magnussen", "ZHOU": "G. Zhou", "SARGEANT": "L. Sargeant",
    "RICCIARDO": "D. Ricciardo", "BEARMAN": "O. Bearman", "COLA PINTO": "F. Colapinto"
}

try:
    if JSON_INPUT.exists():
        with open(JSON_INPUT, "r") as f:
            payload = json.load(f)
        
        # Load live drivers (Normalize keys to uppercase to match market_predictions.json output)
        for raw_name, stats in payload["drivers"].items():
            name_key = str(raw_name).strip().upper()
            display_name = NAME_DISPLAY_MAP.get(name_key, raw_name.title())
            driver_costs[display_name] = stats["cost"]
            driver_expected_points[display_name] = stats["expected_points"]
            
        # Load live constructors (Ensure clean casing string matches)
        for raw_name, stats in payload["constructors"].items():
            c_name = str(raw_name).strip().title().replace("Redbull", "Red Bull").replace("Racingbulls", "Racing Bulls")
            constructor_costs[c_name] = stats["cost"]
            constructor_expected_points[c_name] = stats["expected_points"]
            
        print(f"[INFO] Successfully loaded live data from {JSON_INPUT}\n")
        
    else:
        print("[INFO] market_predictions.json not found. Triggering market_predict pipeline...")
        from market_predict import run_pipeline
        raw_dc, raw_dep, raw_cc, raw_cep = run_pipeline()
        
        # Normalize keys generated live from pipeline execution
        for raw_name, cost in raw_dc.items():
            name_key = str(raw_name).strip().upper()
            display_name = NAME_DISPLAY_MAP.get(name_key, raw_name.title())
            driver_costs[display_name] = cost
            driver_expected_points[display_name] = raw_dep[raw_name]
            
        for raw_name, cost in raw_cc.items():
            c_name = str(raw_name).strip().title().replace("Redbull", "Red Bull").replace("Racingbulls", "Racing Bulls")
            constructor_costs[c_name] = cost
            constructor_expected_points[c_name] = raw_cep[raw_name]
            
        print("[INFO] Live data generated and loaded successfully.\n")

except Exception as e:
    print(f"[WARN] Error loading live predictions ({e}). Running with historical demo data.\n")
    # Backup static fallback data
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

BUDGET_CAP = 100.0  
NUM_DRIVERS = 5
NUM_CONSTRUCTORS = 2

model = pulp.LpProblem("Fantasy_F1_Lineup_Optimizer", pulp.LpMaximize)

def sanitize_var_name(name):
    return name.replace(' ', '_').replace('.', '').replace('-', '_').replace('(', '').replace(')', '')

driver_vars = {d: pulp.LpVariable(f"driver_{sanitize_var_name(d)}", cat="Binary") for d in drivers}
chip_vars = {d: pulp.LpVariable(f"chip_{sanitize_var_name(d)}", cat="Binary") for d in drivers}
constructor_vars = {c: pulp.LpVariable(f"constructor_{sanitize_var_name(c)}", cat="Binary") for c in constructors}

model += (
    pulp.lpSum(driver_expected_points[d] * driver_vars[d] for d in drivers)
    + pulp.lpSum(driver_expected_points[d] * chip_vars[d] for d in drivers)
    + pulp.lpSum(constructor_expected_points[c] * constructor_vars[c] for c in constructors)
), "Total_Expected_Points"

model += (pulp.lpSum(driver_costs[d] * driver_vars[d] for d in drivers) + 
          pulp.lpSum(constructor_costs[c] * constructor_vars[c] for c in constructors) <= BUDGET_CAP), "Budget_Cap"

model += (pulp.lpSum(driver_vars[d] for d in drivers) == NUM_DRIVERS), "Exactly_5_Drivers"
model += (pulp.lpSum(constructor_vars[c] for c in constructors) == NUM_CONSTRUCTORS), "Exactly_2_Constructors"

for d in drivers:
    model += (chip_vars[d] <= driver_vars[d]), f"Chip_Constraint_{sanitize_var_name(d)}"
model += (pulp.lpSum(chip_vars[d] for d in drivers) == 1), "Exactly_One_Chip"

solver = pulp.PULP_CBC_CMD(msg=False)
status = model.solve(solver)

if pulp.LpStatus[model.status] != "Optimal":
    sys.exit(f"[ERROR] Solver could not find an optimal solution. Status: {pulp.LpStatus[model.status]}")

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
