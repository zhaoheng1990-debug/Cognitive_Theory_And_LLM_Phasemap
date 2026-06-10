
# ============================================================
# ASA-9A Driver: Cross-Model Closed-Loop Generalization Audit
#
# This driver reuses the already validated ASA-8D engine.
#
# Requirement:
#   Put this file in the same folder as either:
#       asa8d_matched_shuffle_closed_loop_validation_PATCHED.py
#   or:
#       GPT_194_ASA_8D.py
#
# Why driver style?
#   ASA-8D already validated the full predicted-mechanism closed-loop
#   control pipeline and matched-null logic. ASA-9A should minimize
#   new code risk by only changing:
#
#       model_path
#       layer windows
#       output directory
#       n_shuffles
#
# Goal:
#   Run ASA-8D-style matched-null validation across:
#       qwen / llama / gemma
#
# Interpretation:
#   - all PASS_STRONG: cross-model validated closed-loop trajectory control
#   - mixed PASS: model-specific controllability, needs window refinement
#   - only Qwen PASS: Qwen-specific phenomenon for now
#
# Default:
#   n_shuffles = 20 for first pass.
#   If promising, rerun final with n_shuffles = 50.
# ============================================================

import os
import sys
import json
import copy
import shutil
import importlib.util
from pathlib import Path
import pandas as pd

# ============================================================
# EDIT THESE PATHS IF NEEDED
# ============================================================

MODEL_CONFIGS = {
    "qwen": {
        "model_path": r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main",
        "operator_layers": tuple(range(15, 20)),
        "precursor_layers": (17, 18, 19),
        "decision_layers": tuple(range(20, 26)),
    },
    "llama": {
        "model_path": r"D:\model\Llama-3.2-1B-Instruct",
        "operator_layers": tuple(range(8, 12)),
        "precursor_layers": (10, 11, 12),
        "decision_layers": tuple(range(12, 16)),
    },
    "gemma": {
        "model_path": r"D:\model\gemma-2-2b-it",
        "operator_layers": tuple(range(13, 18)),
        "precursor_layers": (15, 16, 17),
        "decision_layers": tuple(range(18, 24)),
    },
}

MODELS_TO_RUN = ("qwen", "llama", "gemma")

ROOT_SAVE_DIR = Path("./asa9a_outputs")
N_SHUFFLES = 20      # first pass; use 50 for final validation
BATCH_SIZE = 4
N_GRAPHS = 96

ASA8D_CANDIDATE_FILES = [
    "asa8d_matched_shuffle_closed_loop_validation_PATCHED.py",
    "GPT_194_ASA_8D.py",
    "asa8d_matched_shuffle_closed_loop_validation.py",
]

# ============================================================
# IMPORT ASA-8D ENGINE
# ============================================================

def find_asa8d_file():
    here = Path(__file__).resolve().parent
    for name in ASA8D_CANDIDATE_FILES:
        p = here / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find ASA-8D engine. Put one of these files in the same folder:\n"
        + "\n".join(ASA8D_CANDIDATE_FILES)
    )

def import_module_from_file(path):
    spec = importlib.util.spec_from_file_location("asa8d_engine", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["asa8d_engine"] = mod
    spec.loader.exec_module(mod)
    return mod

# ============================================================
# CONFIG PATCHING
# ============================================================

def patch_config(engine, model_key, model_cfg, root_save_dir):
    cfg = copy.deepcopy(engine.CFG)

    # Core model/profile.
    cfg.model_key = model_key
    cfg.model_path = model_cfg["model_path"]
    cfg.save_dir = str(root_save_dir / model_key)

    cfg.operator_layers = tuple(model_cfg["operator_layers"])
    cfg.precursor_layers = tuple(model_cfg["precursor_layers"])
    cfg.decision_layers = tuple(model_cfg["decision_layers"])

    # Compute controls.
    cfg.n_shuffles = N_SHUFFLES
    cfg.batch_size = BATCH_SIZE
    cfg.n_graphs = N_GRAPHS

    return cfg

def read_model_outputs(model_dir, model_key):
    summary_path = model_dir / "asa8d_summary.json"
    verdict_path = model_dir / "asa8d_verdict.csv"
    spec_path = model_dir / "asa8d_specificity_summary.csv"
    shuffle_path = model_dir / "asa8d_shuffle_validation.csv"

    row = {
        "model": model_key,
        "status": "missing_outputs",
        "summary_path": str(summary_path),
    }

    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        row.update({
            "status": "ok",
            "verdict": summary.get("verdict"),
            "real_predmech_specificity": summary.get("main_metrics", {}).get("real_predmech_specificity"),
            "fixed_combo_specificity": summary.get("main_metrics", {}).get("fixed_combo_specificity"),
            "gain_over_fixed": summary.get("main_metrics", {}).get("gain_over_fixed"),
            "z_vs_shuffle_both": summary.get("main_metrics", {}).get("z_vs_shuffle_both"),
            "z_vs_shuffle_mech_policy": summary.get("main_metrics", {}).get("z_vs_shuffle_mech_policy"),
            "z_vs_shuffle_policy_label": summary.get("main_metrics", {}).get("z_vs_shuffle_policy_label"),
            "exceeds_shuffle_both_q95": summary.get("main_metrics", {}).get("exceeds_shuffle_both_q95"),
            "deltaU_pca_explained_variance": summary.get("deltaU_pca_explained_variance"),
            "n_policy_train_prompts": summary.get("n_policy_train_prompts"),
            "n_test_prompts": summary.get("n_test_prompts"),
        })

    if verdict_path.exists():
        try:
            v = pd.read_csv(verdict_path)
            if len(v):
                for c in v.columns:
                    row[f"verdict_csv_{c}"] = v.iloc[0][c]
        except Exception as e:
            row["verdict_csv_error"] = repr(e)

    return row

# ============================================================
# MAIN
# ============================================================

def main():
    ROOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    engine_file = find_asa8d_file()
    print(f"[ASA-9A] Using ASA-8D engine: {engine_file}")
    engine = import_module_from_file(engine_file)

    summaries = []

    for model_key in MODELS_TO_RUN:
        print("\n" + "=" * 88)
        print(f"[ASA-9A] Running model: {model_key}")
        print("=" * 88)

        model_cfg = MODEL_CONFIGS[model_key]
        model_save_dir = ROOT_SAVE_DIR / model_key
        model_save_dir.mkdir(parents=True, exist_ok=True)

        cfg = patch_config(engine, model_key, model_cfg, ROOT_SAVE_DIR)

        # Write the exact config used for traceability.
        with open(model_save_dir / "asa9a_model_run_config.json", "w", encoding="utf-8") as f:
            json.dump({
                "model_key": model_key,
                "model_cfg": {
                    "model_path": model_cfg["model_path"],
                    "operator_layers": list(model_cfg["operator_layers"]),
                    "precursor_layers": list(model_cfg["precursor_layers"]),
                    "decision_layers": list(model_cfg["decision_layers"]),
                },
                "n_shuffles": N_SHUFFLES,
                "n_graphs": N_GRAPHS,
                "batch_size": BATCH_SIZE,
                "engine_file": str(engine_file),
            }, f, indent=2, ensure_ascii=False)

        try:
            # Run ASA-8D engine with patched config.
            engine.main(cfg)
            row = read_model_outputs(model_save_dir, model_key)
            summaries.append(row)
        except Exception as e:
            print(f"[ASA-9A][ERROR] {model_key}: {repr(e)}")
            row = {
                "model": model_key,
                "status": "error",
                "error": repr(e),
                "model_path": model_cfg["model_path"],
                "operator_layers": list(model_cfg["operator_layers"]),
                "precursor_layers": list(model_cfg["precursor_layers"]),
                "decision_layers": list(model_cfg["decision_layers"]),
            }
            summaries.append(row)
            with open(model_save_dir / "asa9a_error.json", "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2, ensure_ascii=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(ROOT_SAVE_DIR / "asa9a_cross_model_summary.csv", index=False, encoding="utf-8-sig")

    with open(ROOT_SAVE_DIR / "asa9a_cross_model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    n_total = len([x for x in summaries if x.get("status") == "ok"])
    n_pass_strong = sum(x.get("verdict") == "PASS_STRONG_VALIDATED_CLOSED_LOOP" for x in summaries)
    n_pass_lite = sum("PASS_LITE" in str(x.get("verdict", "")) for x in summaries)
    n_gain = sum("GAIN_OVER_FIXED" in str(x.get("verdict", "")) for x in summaries)

    final = {
        "audit": "ASA-9A Cross-Model Closed-Loop Generalization Audit",
        "models_to_run": list(MODELS_TO_RUN),
        "n_total_completed": n_total,
        "n_pass_strong": int(n_pass_strong),
        "n_pass_lite": int(n_pass_lite),
        "n_gain_over_fixed_but_null_weak": int(n_gain),
        "n_shuffles": N_SHUFFLES,
        "summary_csv": "asa9a_cross_model_summary.csv",
        "interpretation_rules": {
            "all_pass_strong": "Cross-model validated mechanism-aware trajectory control.",
            "mixed_pass": "Model-specific controllability; refine model-specific windows or policy features.",
            "qwen_only": "Qwen-specific for now; not yet general.",
        }
    }

    with open(ROOT_SAVE_DIR / "asa9a_final_verdict.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print("\n[ASA-9A] Complete.")
    print(json.dumps(final, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
