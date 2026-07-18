from pathlib import Path
import hashlib
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "source_data" / "extension_v0_54"
if not DATA.exists():
    DATA = HERE.parents[1] / "source_data" / "source_data" / "extension_v0_54"

def check(name, condition, detail):
    print(("PASS" if condition else "FAIL"), name, "-", detail)
    return bool(condition)

checks = []
predictive = json.loads((DATA / "predictive_transition_boundary" / "audit.json").read_text())
checks.append(check("predictive increment", predictive["metrics"]["increment_macro_R2"] > 0.039, str(predictive["metrics"])))
checks.append(check("predictive lower CI", predictive["metrics"]["bootstrap_95_CI"][0] > 0, str(predictive["metrics"]["bootstrap_95_CI"])))

window = pd.read_csv(DATA / "functional_window_confirmation" / "outputs" / "functional_window_confirmation_summary_v0_1.csv")
checks.append(check("window rows", len(window) == 3, f"rows={len(window)}"))
checks.append(check("prospective role recurrence", int(window["role_recurrence"].sum()) == 0, f"passes={int(window['role_recurrence'].sum())}/3"))
checks.append(check("prospective location recurrence", int(window["window_location_recurrence"].sum()) == 1, f"passes={int(window['window_location_recurrence'].sum())}/3"))

actuator = pd.read_csv(DATA / "actuator_attribution" / "crossmodel_actuator_component_summary.csv")
def total(control, column):
    return int(actuator.loc[actuator.control.eq(control), column].sum())
checks.append(check("gated flips", total("policy_boundary_guarded", "strict_conflict_to_clean") == 39, str(total("policy_boundary_guarded", "strict_conflict_to_clean"))))
checks.append(check("operator-only flips", total("policy_gate_operator_only", "strict_conflict_to_clean") == 39, str(total("policy_gate_operator_only", "strict_conflict_to_clean"))))
checks.append(check("precursor-only flips", total("policy_gate_precursor_only", "strict_conflict_to_clean") == 1, str(total("policy_gate_precursor_only", "strict_conflict_to_clean"))))

crosswalk = json.loads((DATA / "label_crosswalk" / "label_crosswalk_audit.json").read_text())
checks.append(check("label crosswalk", crosswalk["status"] == "PASS" and all(crosswalk["checks"].values()), crosswalk["status"]))

if not all(checks):
    raise SystemExit(1)
print(f"PASS {sum(checks)}/{len(checks)} public v0.54 checks")
