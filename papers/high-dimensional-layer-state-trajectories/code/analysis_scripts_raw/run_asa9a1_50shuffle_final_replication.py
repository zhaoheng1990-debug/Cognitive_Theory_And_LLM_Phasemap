r"""Run ASA-9A.1 50-shuffle final replication from recovered ASA-9A runner.

This script imports the recovered ASA-9A base driver:
inputs/asa9a_cross_model_driver_PATCHED.py

It patches only the output directory and shuffle count:
outputs/asa9a1_50shuffle_final_replication_output
N_SHUFFLES = 50

If the base runner or required artifacts are unavailable, it writes blocker
reports rather than fabricating a replication table.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("outputs/asa9a1_50shuffle_final_replication_output")
RECOVERY_DIR = Path("outputs/nature_missing_experiment_recovery_output")
BASE_RUNNER = Path("inputs/asa9a_cross_model_driver_PATCHED.py")
ASA8D_ENGINE = Path("inputs/GPT_194_ASA_8D.py")
N_SHUFFLES = 50
MODELS = ["qwen", "llama", "gemma"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for ASA-9A.1 outputs.")
    parser.add_argument("--recovery-dir", type=Path, default=RECOVERY_DIR, help="Optional recovery-output directory recorded in inventories.")
    parser.add_argument("--base-runner", type=Path, default=BASE_RUNNER, help="ASA-9A base driver to import.")
    parser.add_argument("--asa8d-engine", type=Path, default=ASA8D_ENGINE, help="ASA-8D closed-loop engine dependency.")
    parser.add_argument("--n-shuffles", type=int, default=N_SHUFFLES)
    parser.add_argument("--models", default=",".join(MODELS), help="Comma-separated model keys to run.")
    parser.add_argument("--check-inputs-only", action="store_true", help="Write an input-availability report without importing the base runner.")
    return parser.parse_args()


def configure_from_args(args: argparse.Namespace) -> None:
    global OUTPUT_DIR, RECOVERY_DIR, BASE_RUNNER, ASA8D_ENGINE, N_SHUFFLES, MODELS
    OUTPUT_DIR = args.output_dir
    RECOVERY_DIR = args.recovery_dir
    BASE_RUNNER = args.base_runner
    ASA8D_ENGINE = args.asa8d_engine
    N_SHUFFLES = args.n_shuffles
    MODELS = [x.strip() for x in args.models.split(",") if x.strip()]


def report_path(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def path_status(path: Path | str) -> dict[str, object]:
    p = Path(path)
    return {
        "path": report_path(p),
        "exists": p.exists(),
        "status": "FOUND" if p.exists() else "MISSING",
    }


def write_input_check_report() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "base_runner": path_status(BASE_RUNNER),
        "asa8d_engine": path_status(ASA8D_ENGINE),
        "recovery_dir": report_path(RECOVERY_DIR),
        "output_dir": report_path(OUTPUT_DIR),
        "n_shuffles": N_SHUFFLES,
        "models": MODELS,
        "mode": "preflight_or_runtime_dependency_check",
    }
    write_json(OUTPUT_DIR / "asa9a1_input_check_report.json", report)
    return report


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def blocker(reason: str, details: dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = [
        "# ASA-9A.1 blocker report",
        "",
        f"Status: `{reason}`",
        "",
        "ASA-9A.1 cannot be reconstructed from aggregate summaries. It requires the ASA-9A base runner and intervention artifacts.",
        "",
        "## Details",
        "",
    ]
    report.extend(f"- {k}: {v}" for k, v in details.items())
    (OUTPUT_DIR / "asa9a1_blocker_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "asa9a1_rebuild_plan.md").write_text(
        "\n".join([
            "# ASA-9A.1 rebuild plan",
            "",
            "1. Recover or rebuild the ASA-9A base runner.",
            "2. Verify the ASA-8D closed-loop engine dependency.",
            "3. Preserve Qwen/Llama/Gemma model-specific windows.",
            "4. Run with N_SHUFFLES = 50.",
            "5. Export per-sample intervention outputs, shuffle nulls, policy metrics and mechanism metrics.",
        ]) + "\n",
        encoding="utf-8",
    )
    write_csv(OUTPUT_DIR / "asa9a1_missing_or_blocked_files.csv", [
        {"item": "ASA-9A.1", "status": reason, "details": json.dumps(details, ensure_ascii=False)}
    ])


def import_base_runner():
    spec = importlib.util.spec_from_file_location("asa9a_base_runner", str(BASE_RUNNER))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create import spec for base runner.")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["asa9a_base_runner"] = mod
    spec.loader.exec_module(mod)
    return mod


def z_score(real: float | None, mean: float | None, std: float | None) -> float | None:
    if real is None or mean is None or std in (None, 0):
        return None
    return (real - mean) / std


def q95(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.quantile(0.95))


def read_summary(model_dir: Path, model_key: str) -> dict[str, object]:
    candidates = [
        model_dir / f"{model_key}_asa8d_summary.json",
        model_dir / "asa8d_summary.json",
        model_dir / ("Qwen_asa8d_summary.json" if model_key == "qwen" else f"{model_key}_asa8d_summary.json"),
    ]
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def postprocess() -> None:
    model_rows = []
    main_rows = []
    shuffle_rows = []
    policy_rows = []
    mechanism_rows = []
    action_rows = []
    missing_rows = []

    for model_key in MODELS:
        model_dir = OUTPUT_DIR / model_key
        cfg_path = model_dir / "asa9a_model_run_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        model_rows.append({
            "model": model_key,
            "model_path": cfg.get("model_cfg", {}).get("model_path"),
            "operator_layers": cfg.get("model_cfg", {}).get("operator_layers"),
            "precursor_layers": cfg.get("model_cfg", {}).get("precursor_layers"),
            "decision_layers": cfg.get("model_cfg", {}).get("decision_layers"),
            "n_shuffles": cfg.get("n_shuffles"),
            "status": "FOUND_CONFIG" if cfg else "MISSING_CONFIG",
        })
        summary = read_summary(model_dir, model_key)
        metrics = summary.get("main_metrics", {}) if summary else {}
        shuffle_path = next(iter(list(model_dir.glob("*shuffle_validation.csv")) + [model_dir / "asa8d_shuffle_validation.csv"]), None)
        shuf = pd.read_csv(shuffle_path) if shuffle_path and shuffle_path.exists() else pd.DataFrame()
        if shuf.empty:
            missing_rows.append({"item": model_key, "status": "MISSING_SHUFFLE_VALIDATION", "path": report_path(shuffle_path)})
        else:
            shuf["model"] = model_key
            shuffle_rows.extend(shuf.to_dict(orient="records"))
        real = metrics.get("real_predmech_specificity")
        fixed = metrics.get("fixed_combo_specificity")
        sb_mean = metrics.get("shuffle_both_mean")
        sb_std = metrics.get("shuffle_both_std")
        if sb_mean is None and not shuf.empty:
            both_cols = [c for c in shuf.columns if "both" in c.lower() and ("specificity" in c.lower() or "score" in c.lower())]
            if both_cols:
                sb_mean = float(shuf[both_cols[0]].mean())
                sb_std = float(shuf[both_cols[0]].std())
        real_gt_q95 = metrics.get("exceeds_shuffle_both_q95")
        if real_gt_q95 is None and real is not None and not shuf.empty:
            both_cols = [c for c in shuf.columns if "both" in c.lower() and ("specificity" in c.lower() or "score" in c.lower())]
            real_gt_q95 = bool(real > q95(shuf[both_cols[0]])) if both_cols else None
        main_rows.append({
            "model": model_key,
            "real_predmech_specificity": real,
            "fixed_combo_specificity": fixed,
            "gain_over_fixed": metrics.get("gain_over_fixed"),
            "shuffle_mechanism_mean": metrics.get("shuffle_mechanism_mean"),
            "shuffle_mechanism_std": metrics.get("shuffle_mechanism_std"),
            "shuffle_policy_mean": metrics.get("shuffle_policy_mean"),
            "shuffle_policy_std": metrics.get("shuffle_policy_std"),
            "shuffle_both_mean": sb_mean,
            "shuffle_both_std": sb_std,
            "z_vs_shuffle_mechanism": metrics.get("z_vs_shuffle_mechanism"),
            "z_vs_shuffle_policy": metrics.get("z_vs_shuffle_policy"),
            "z_vs_shuffle_both": metrics.get("z_vs_shuffle_both") or z_score(real, sb_mean, sb_std),
            "real_gt_shuffle_both_q95": real_gt_q95,
            "policy_train_accuracy": summary.get("policy_train_accuracy") or metrics.get("policy_train_accuracy"),
            "policy_macro_f1": summary.get("policy_macro_f1") or metrics.get("policy_macro_f1"),
            "mechanism_train_accuracy": summary.get("mechanism_train_accuracy") or metrics.get("mechanism_train_accuracy"),
            "mechanism_macro_f1": summary.get("mechanism_macro_f1") or metrics.get("mechanism_macro_f1"),
            "source_summary": str(model_dir),
        })
        for json_name, target, rows in [
            ("asa8d_real_policy_audit.json", "policy", policy_rows),
            ("asa8d_real_mechanism_audit.json", "mechanism", mechanism_rows),
        ]:
            p = model_dir / json_name
            if p.exists():
                obj = json.loads(p.read_text(encoding="utf-8"))
                obj["model"] = model_key
                rows.append(obj)
        actions_path = model_dir / "asa8d_real_policy_actions.csv"
        if actions_path.exists():
            actions = pd.read_csv(actions_path)
            if "action" in actions.columns:
                dist = actions["action"].value_counts(normalize=True).reset_index()
                dist.columns = ["action", "fraction"]
            else:
                dist = actions.astype(str).value_counts(normalize=True).reset_index(name="fraction")
            dist["model"] = model_key
            action_rows.extend(dist.to_dict(orient="records"))

    write_csv(OUTPUT_DIR / "asa9a1_model_inventory.csv", model_rows)
    write_csv(OUTPUT_DIR / "asa9a1_50shuffle_main_table.csv", main_rows)
    write_csv(OUTPUT_DIR / "asa9a1_shuffle_null_distribution.csv", shuffle_rows)
    write_csv(OUTPUT_DIR / "asa9a1_by_model_policy_metrics.csv", policy_rows)
    write_csv(OUTPUT_DIR / "asa9a1_by_model_mechanism_metrics.csv", mechanism_rows)
    write_csv(OUTPUT_DIR / "asa9a1_policy_action_distribution.csv", action_rows)
    write_csv(OUTPUT_DIR / "asa9a1_missing_or_blocked_files.csv", missing_rows, ["item", "status", "path"])

    pass_count = 0
    borderline = 0
    for row in main_rows:
        real = row.get("real_predmech_specificity")
        fixed = row.get("fixed_combo_specificity")
        zboth = row.get("z_vs_shuffle_both")
        gtq95 = row.get("real_gt_shuffle_both_q95")
        passed = real is not None and fixed is not None and real > fixed and zboth is not None and zboth > 2 and bool(gtq95)
        if passed:
            pass_count += 1
        elif real is not None and fixed is not None and real > fixed:
            borderline += 1
    if pass_count == 3:
        verdict = "PASS_STRONG_FINAL_REPLICATION"
    elif pass_count == 2:
        verdict = "PASS_MIXED_FINAL_REPLICATION"
    elif missing_rows:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "FAIL"
    verdict_obj = {
        "verdict": verdict,
        "n_shuffles_required": 50,
        "models": MODELS,
        "models_passing": pass_count,
        "models_borderline_gain_only": borderline,
        "missing_or_blocked_count": len(missing_rows),
    }
    write_json(OUTPUT_DIR / "asa9a1_verdict.json", verdict_obj)
    lines = [
        "# ASA-9A.1 50-shuffle final replication",
        "",
        "This window inherits MethodologyKernel v1.1.",
        "If any experimental conclusion conflicts with this kernel, the conflict must be explicitly stated and converted into a theory revision, downgrade, or caveat.",
        "",
        f"Base runner: `{report_path(BASE_RUNNER)}`",
        f"Output directory: `{report_path(OUTPUT_DIR)}`",
        f"N_SHUFFLES: {N_SHUFFLES}",
        "",
        f"Verdict: `{verdict}`",
        "",
        "## Main table",
        "",
    ]
    for row in main_rows:
        lines.append(
            f"- {row['model']}: real={row.get('real_predmech_specificity')}, fixed={row.get('fixed_combo_specificity')}, "
            f"gain={row.get('gain_over_fixed')}, z_both={row.get('z_vs_shuffle_both')}, gt_q95={row.get('real_gt_shuffle_both_q95')}"
        )
    if missing_rows:
        lines += ["", "## Missing or blocked", ""]
        lines.extend(f"- {r['item']}: {r['status']} ({r['path']})" for r in missing_rows)
    (OUTPUT_DIR / "asa9a1_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_from_args(args)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_check = write_input_check_report()
    if args.check_inputs_only:
        print(json.dumps(input_check, ensure_ascii=False, indent=2, default=str))
        return
    if not BASE_RUNNER.exists():
        blocker("BLOCKED_MISSING_ASA9A_BASE_RUNNER", {"base_runner": report_path(BASE_RUNNER)})
        return
    if not ASA8D_ENGINE.exists():
        blocker("BLOCKED_MISSING_ASA8D_ENGINE", {"asa8d_engine": report_path(ASA8D_ENGINE)})
        return

    shutil.copy2(BASE_RUNNER, OUTPUT_DIR / "asa9a1_base_runner_copy.py")
    shutil.copy2(ASA8D_ENGINE, OUTPUT_DIR / "asa9a1_asa8d_engine_copy.py")
    write_json(OUTPUT_DIR / "asa9a1_input_inventory.json", {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "base_runner": report_path(BASE_RUNNER),
        "asa8d_engine": report_path(ASA8D_ENGINE),
        "output_dir": report_path(OUTPUT_DIR),
        "n_shuffles": N_SHUFFLES,
        "models": MODELS,
    })
    write_json(OUTPUT_DIR / "asa9a1_base_runner_inventory.json", {
        "base_runner_found": True,
        "base_runner_path": report_path(BASE_RUNNER),
        "asa8d_engine_found": True,
        "asa8d_engine_path": report_path(ASA8D_ENGINE),
        "patched_fields": {
            "ROOT_SAVE_DIR": report_path(OUTPUT_DIR),
            "N_SHUFFLES": N_SHUFFLES,
            "MODELS_TO_RUN": MODELS,
        },
    })

    try:
        base = import_base_runner()
        base.ROOT_SAVE_DIR = OUTPUT_DIR
        base.N_SHUFFLES = N_SHUFFLES
        base.MODELS_TO_RUN = tuple(MODELS)
        base.main()
        postprocess()
    except Exception as exc:
        err = {
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "base_runner": report_path(BASE_RUNNER),
        }
        write_json(OUTPUT_DIR / "asa9a1_runtime_error.json", err)
        blocker("BLOCKED_RUNTIME_ERROR", err)


if __name__ == "__main__":
    main()
