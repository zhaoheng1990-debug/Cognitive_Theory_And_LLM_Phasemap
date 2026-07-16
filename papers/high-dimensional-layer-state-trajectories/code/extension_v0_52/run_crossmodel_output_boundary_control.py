"""Cross-checkpoint candidate-output boundary control using frozen model-specific windows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


EXECUTOR_SOURCE = Path(__file__).resolve().parent / "local_transition_executor.py"
QWEN_CONTROLLER_SOURCE = Path(__file__).resolve().parent / "boundary_controller.py"
EXISTING_QWEN_OUTPUT = Path("inputs/qwen_existing_output")

MODEL_WINDOWS = {
    "qwen": {
        "operator_layers": (15, 16, 17, 18, 19),
        "precursor_layers": (17, 18, 19),
        "decision_layers": (20, 21, 22, 23, 24, 25),
    },
    "llama": {
        "operator_layers": (8, 9, 10, 11),
        "precursor_layers": (10, 11, 12),
        "decision_layers": (12, 13, 14, 15),
    },
    "gemma": {
        "operator_layers": (13, 14, 15, 16, 17),
        "precursor_layers": (15, 16, 17),
        "decision_layers": (18, 19, 20, 21, 22, 23),
    },
}

FULL_ACTIONS = (
    (0.0, 0.0),
    (0.3, 0.3),
    (0.6, 0.3),
    (0.6, 0.6),
    (0.9, 0.3),
    (0.9, 0.6),
    (0.9, 0.9),
    (1.0, 0.6),
    (1.0, 0.9),
    (1.0, 1.2),
)


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_WINDOWS), default=["llama", "gemma"])
    parser.add_argument(
        "--output-root", type=Path, default=Path("crossmodel_output_boundary_control")
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--include-existing-qwen", action="store_true")
    return parser.parse_args()


def run_model(model_key: str, out_dir: Path, smoke: bool) -> dict:
    qmod = load_module(f"boundary_controller_{model_key}", QWEN_CONTROLLER_SOURCE)
    asa = qmod.load_executor(EXECUTOR_SOURCE)
    cfg = asa.Config()
    cfg.model_key = model_key
    cfg.model_path = "auto"
    cfg.save_dir = str(out_dir / "local_transition_runtime")
    cfg.n_graphs = 12 if smoke else 96
    cfg.batch_size = 4
    cfg.n_shuffles = 0
    cfg.include_answer_control = False
    for field, value in MODEL_WINDOWS[model_key].items():
        setattr(cfg, field, value)
    actions = FULL_ACTIONS[:4] if smoke else FULL_ACTIONS
    cfg.policy_actions = tuple(actions)
    asa.CFG = cfg
    asa.make_combo_hook_factory = qmod.norm_tracing_hook_factory(asa)
    asa.set_seed(cfg.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "executor_source": str(EXECUTOR_SOURCE),
                "controller_source": str(QWEN_CONTROLLER_SOURCE),
                "executor_config": asdict(cfg),
                "actions": actions,
                "smoke": smoke,
                "policy_objective": "train-only paired candidate-boundary margin",
                "gate": {"mean_p_closure_min": 0.50, "mean_entropy_max": 0.90},
                "confidence_gate_role": "mechanism-confidence and entropy gate only",
            },
            handle,
            indent=2,
        )

    print(f"[{model_key}] load model and tokenizer", flush=True)
    model, tokenizer = asa.load_model_and_tokenizer(cfg)
    dataset_dir = out_dir / "local_transition_runtime"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset = asa.build_dataset(tokenizer, cfg, dataset_dir)
    base, hidden, _ = asa.extract_baseline(model, tokenizer, dataset, cfg, dataset_dir)
    train_index, test_index, split = asa.make_split(base, cfg, dataset_dir)
    split.to_csv(out_dir / "frozen_group_split.csv", index=False)
    base, pca = qmod.refit_delta_u_on_train(asa, base, train_index, cfg, out_dir)

    clean = base[base["condition"].eq("clean")].set_index("graph_id")
    anchors = {
        int(graph): {layer: float(row[f"R_L{layer}"]) for layer in cfg.decision_layers}
        for graph, row in clean.iterrows()
    }
    stable_operators = asa.fit_stable_operators(base, hidden, train_index, cfg, dataset_dir)
    precursor_directions, probes, velocity = asa.fit_precursor_and_probes(
        base, hidden, train_index, cfg, dataset_dir
    )

    train = base.iloc[train_index].reset_index(drop=True)
    test = base.iloc[test_index].reset_index(drop=True)
    train_features = asa.base_state_features_for_samples(
        base, hidden, probes, velocity, train_index, cfg
    )
    test_features = asa.base_state_features_for_samples(
        base, hidden, probes, velocity, test_index, cfg
    )

    grid = qmod.run_grid(
        asa,
        model,
        tokenizer,
        train,
        cfg,
        pca,
        anchors,
        stable_operators,
        precursor_directions,
        actions,
        out_dir,
    )
    merged_grid, labels = qmod.derive_train_labels(grid, train, out_dir)
    global_action = qmod.select_global_action(merged_grid, actions, out_dir)
    print(f"[{model_key}] selected global action {global_action}", flush=True)

    mechanism_classifier, mechanism_audit = asa.fit_mechanism_classifier(
        train_features, cfg, dataset_dir
    )
    train_predicted = asa.attach_predicted_mechanism(train_features, mechanism_classifier)
    test_predicted = asa.attach_predicted_mechanism(test_features, mechanism_classifier)
    policy_classifier, policy_audit = asa.fit_gain_policy(
        train_predicted,
        labels,
        asa.POLICY_PRED_MECH_COLS,
        cfg,
        shuffle_policy_labels=False,
        shuffle_id=0,
    )
    policy_actions = asa.policy_predict_actions(
        policy_classifier, test_predicted, asa.POLICY_PRED_MECH_COLS
    )
    guarded_actions = qmod.apply_confidence_gate(policy_actions, test_predicted)
    shuffled_classifier, shuffled_audit = asa.fit_gain_policy(
        train_predicted,
        labels,
        asa.POLICY_PRED_MECH_COLS,
        cfg,
        shuffle_policy_labels=True,
        shuffle_id=0,
    )
    shuffled_actions = asa.policy_predict_actions(
        shuffled_classifier, test_predicted, asa.POLICY_PRED_MECH_COLS
    )
    fixed_actions = qmod.uniform_actions(
        test, cfg.operator_layers, global_action, "training_global"
    )
    action_tables = {
        "policy_fixed_global": fixed_actions,
        "policy_boundary": policy_actions,
        "policy_boundary_guarded": guarded_actions,
        "shuffle_boundary_labels": shuffled_actions,
    }
    for name, table in action_tables.items():
        table.to_csv(out_dir / f"actions_{name}.csv", index=False)
    with (out_dir / "classifier_audits.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "mechanism": mechanism_audit,
                "boundary_policy": policy_audit,
                "shuffle_policy": shuffled_audit,
            },
            handle,
            indent=2,
        )

    controls = {
        "none": None,
        "same_combo_fixed": None,
        "policy_fixed_global": fixed_actions,
        "policy_boundary": policy_actions,
        "policy_boundary_guarded": guarded_actions,
        "shuffle_boundary_labels": shuffled_actions,
    }
    results = {}
    traces = {}
    for control, table in controls.items():
        result, trace = qmod.evaluate_control(
            asa,
            model,
            tokenizer,
            test,
            cfg,
            pca,
            anchors,
            stable_operators,
            precursor_directions,
            control,
            table,
            out_dir,
        )
        results[control] = result
        traces[control] = trace
    summary, _ = qmod.summarize(results, traces, test, action_tables, out_dir)

    top1_controls = {
        "none": None,
        "same_combo_fixed": None,
        "policy_boundary": policy_actions,
        "policy_boundary_guarded": guarded_actions,
        "shuffle_boundary_labels": shuffled_actions,
    }
    top1_results = {
        control: qmod.evaluate_vocab_top1(
            asa,
            model,
            tokenizer,
            test,
            cfg,
            stable_operators,
            precursor_directions,
            control,
            table,
            out_dir,
        )
        for control, table in top1_controls.items()
    }
    top1_summary = qmod.summarize_vocab_top1(top1_results, out_dir)

    final = qmod.verdict(summary)
    primary = top1_summary.set_index("control").loc["policy_boundary"]
    guarded = top1_summary.set_index("control").loc["policy_boundary_guarded"]
    final.update(
        {
            "model": model_key,
            "n_train_prompts": int(len(train)),
            "n_test_prompts": int(len(test)),
            "n_test_graphs": int(test["graph_id"].nunique()),
            "n_test_closure": int(test["mechanism"].eq("closure").sum()),
            "selected_global_action": list(global_action),
            "primary_strict_conflict_to_clean_top1": int(
                primary["closure_strict_conflict_to_clean"]
            ),
            "guarded_strict_conflict_to_clean_top1": int(
                guarded["closure_strict_conflict_to_clean"]
            ),
            "primary_nonclosure_top1_change_rate": float(
                primary["nonclosure_top1_change_rate"]
            ),
            "guarded_nonclosure_top1_change_rate": float(
                guarded["nonclosure_top1_change_rate"]
            ),
        }
    )
    with (out_dir / "verdict.json").open("w", encoding="utf-8") as handle:
        json.dump(final, handle, indent=2)
    print(json.dumps(final, indent=2), flush=True)
    return final


def aggregate(output_root: Path, include_existing_qwen: bool) -> pd.DataFrame:
    sources = {model: output_root / model for model in MODEL_WINDOWS}
    if include_existing_qwen:
        sources["qwen"] = EXISTING_QWEN_OUTPUT
    rows = []
    for model, source in sources.items():
        verdict_path = source / "verdict.json"
        top1_path = source / "vocab_top1_summary.csv"
        if not verdict_path.exists() or not top1_path.exists():
            continue
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        top1 = pd.read_csv(top1_path).set_index("control")
        for control in ("policy_boundary", "policy_boundary_guarded", "shuffle_boundary_labels"):
            row = top1.loc[control]
            rows.append(
                {
                    "model": model,
                    "control": control,
                    "closure_n": int(row["closure_n"]),
                    "strict_conflict_to_clean_top1": int(
                        row["closure_strict_conflict_to_clean"]
                    ),
                    "nonclosure_top1_change_rate": float(row["nonclosure_top1_change_rate"]),
                    "verdict": verdict["verdict"],
                    "source": str(source.resolve()),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(
        output_root / "crossmodel_output_boundary_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return frame


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for model_key in args.models:
        run_model(model_key, args.output_root / model_key, args.smoke)
    aggregate(args.output_root, args.include_existing_qwen)


if __name__ == "__main__":
    main()
