"""Frozen group-held-out audit of predecision trajectory coordinates.

This runner compares inexpensive candidate-logit readbacks with a predecision
DeltaU coordinate, an ordered readback prefix and the native cutoff hidden
state. It intentionally does not search layer positions or use final outputs
as a prediction target.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
EXECUTOR_SOURCE = HERE.parent / "extension_v0_52" / "local_transition_executor.py"

MODEL_SPECS = {
    "qwen": {
        "path_env": "QWEN_MODEL_PATH",
        "decision_layers": (20, 21, 22, 23, 24, 25),
        "cutoff": 22,
        "recent_layers": (20, 21, 22),
    },
    "llama": {
        "path_env": "LLAMA_MODEL_PATH",
        "decision_layers": (12, 13, 14, 15),
        "cutoff": 13,
        "recent_layers": (12, 13),
    },
    "gemma": {
        "path_env": "GEMMA_MODEL_PATH",
        "decision_layers": (18, 19, 20, 21, 22, 23),
        "cutoff": 20,
        "recent_layers": (18, 19, 20),
    },
}

FEATURES = (
    "raw_current",
    "raw_recent",
    "matched_current",
    "matched_recent",
    "DeltaU_pre",
    "ordered_prefix",
    "cutoff_hidden",
)
PRIMARY_CONTRASTS = (
    ("DeltaU_pre", "matched_recent"),
    ("ordered_prefix", "matched_recent"),
    ("cutoff_hidden", "matched_recent"),
)
MECHANISM_TO_ID = {"stable": 0, "competition": 1, "closure": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--graphs", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sign-flips", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=2026080301)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_executor():
    spec = importlib.util.spec_from_file_location("v052_executor", EXECUTOR_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {EXECUTOR_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def model_path(model_key: str) -> str:
    spec = MODEL_SPECS[model_key]
    value = os.environ.get(spec["path_env"])
    if not value:
        raise RuntimeError(
            f"Set {spec['path_env']} to the local checkpoint path before running {model_key}."
        )
    return value


def pca_component_with_training_orientation(
    train_matrix: np.ndarray,
    full_matrix: np.ndarray,
    train_y: np.ndarray,
) -> tuple[PCA, np.ndarray]:
    pca = PCA(n_components=1)
    pca.fit(train_matrix)
    transformed = pca.transform(full_matrix)[:, 0]
    transformed_train = pca.transform(train_matrix)[:, 0]
    stable = transformed_train[train_y == MECHANISM_TO_ID["stable"]]
    closure = transformed_train[train_y == MECHANISM_TO_ID["closure"]]
    if stable.size == 0 or closure.size == 0:
        raise RuntimeError("Every training fold must retain stable and closure prompts")
    if float(stable.mean()) < float(closure.mean()):
        pca.components_[0] *= -1.0
        transformed *= -1.0
    return pca, transformed.astype(np.float32)


def logistic(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            C=1.0,
            random_state=seed,
        ),
    )


def hidden_classifier(n_train: int, dim: int, seed: int):
    components = max(1, min(32, n_train - 1, dim))
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=components, svd_solver="randomized", random_state=seed),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            C=1.0,
            random_state=seed,
        ),
    )


def bh_adjust(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def paired_group_sign_flip(
    reference: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    seed: int,
) -> dict:
    group_values = []
    for graph_id in np.unique(groups):
        mask = groups == graph_id
        ref = f1_score(target[mask], reference[mask], labels=(0, 1, 2), average="macro", zero_division=0)
        cand = f1_score(target[mask], candidate[mask], labels=(0, 1, 2), average="macro", zero_division=0)
        group_values.append(float(cand - ref))
    values = np.asarray(group_values, dtype=float)
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    exceed = 0
    completed = 0
    while completed < repeats:
        chunk = min(10000, repeats - completed)
        signs = rng.choice(np.array((-1.0, 1.0)), size=(chunk, len(values)))
        null = (signs * values[None, :]).mean(axis=1)
        exceed += int(np.sum(null >= observed))
        completed += chunk
    return {
        "macro_f1_difference": observed,
        "group_count": int(len(values)),
        "sign_flips": int(repeats),
        "p": float((exceed + 1) / (repeats + 1)),
    }


def extract_model(executor, model_key: str, args: argparse.Namespace, output_dir: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    spec = MODEL_SPECS[model_key]
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = executor.Config()
    cfg.model_key = model_key
    cfg.model_path = model_path(model_key)
    cfg.save_dir = str(output_dir / "task_input")
    cfg.n_graphs = 4 if args.smoke else args.graphs
    cfg.batch_size = 2 if args.smoke else args.batch_size
    cfg.decision_layers = tuple(spec["decision_layers"])
    cfg.operator_layers = tuple()
    cfg.precursor_layers = tuple()
    executor.CFG = cfg
    executor.set_seed(42)
    model, tokenizer = executor.load_model_and_tokenizer(cfg)
    task_input = output_dir / "task_input"
    task_input.mkdir(parents=True, exist_ok=True)
    data = executor.build_dataset(tokenizer, cfg, task_input)
    if data["graph_id"].nunique() != cfg.n_graphs:
        raise RuntimeError("Unexpected graph count in deterministic task constructor")

    needed_layers = tuple(range(int(spec["cutoff"]) + 1))
    cutoff = int(spec["cutoff"])
    if cutoff not in needed_layers:
        raise RuntimeError("Cutoff must be contained in the predecision readback set")
    margins = {layer: np.zeros(len(data), dtype=np.float32) for layer in needed_layers}
    hidden = None
    weight = executor.get_lm_head_weight(model).detach().float().to(model.device)
    texts = data["text"].tolist()
    clean_ids_all = data["clean_token_id"].to_numpy(np.int64)
    conflict_ids_all = data["conflict_token_id"].to_numpy(np.int64)

    with torch.no_grad():
        for start in range(0, len(data), cfg.batch_size):
            end = min(len(data), start + cfg.batch_size)
            batch = tokenizer(
                texts[start:end],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_len,
            ).to(model.device)
            outputs = model(**batch, output_hidden_states=True, use_cache=False)
            positions = executor.last_positions(batch["attention_mask"])
            indices = torch.arange(end - start, device=model.device)
            clean_ids = torch.as_tensor(clean_ids_all[start:end], device=model.device)
            conflict_ids = torch.as_tensor(conflict_ids_all[start:end], device=model.device)
            for layer in needed_layers:
                state = outputs.hidden_states[layer + 1][indices, positions, :].detach().float()
                margins[layer][start:end] = (
                    (state * weight[clean_ids]).sum(dim=1) - (state * weight[conflict_ids]).sum(dim=1)
                ).cpu().numpy().astype(np.float32)
                if layer == cutoff:
                    current = state.cpu().numpy().astype(np.float16)
                    if hidden is None:
                        hidden = np.empty((len(data), current.shape[1]), dtype=np.float16)
                    hidden[start:end] = current
            del outputs, batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if hidden is None:
        raise RuntimeError("Cutoff hidden-state extraction failed")
    frame = data.copy()
    for layer in needed_layers:
        frame[f"R_L{layer}"] = margins[layer]
    clean_rows = frame[frame["condition"].eq("clean")].set_index("graph_id")
    if clean_rows.index.nunique() != cfg.n_graphs:
        raise RuntimeError("Each graph must have exactly one clean reference prompt")
    for layer in needed_layers:
        frame[f"dR_L{layer}"] = frame.apply(
            lambda row: float(row[f"R_L{layer}"] - clean_rows.loc[int(row["graph_id"]), f"R_L{layer}"]),
            axis=1,
        )
    frame["mechanism_id"] = frame["mechanism"].map(MECHANISM_TO_ID).astype(int)
    frame.to_csv(output_dir / "prompt_readbacks.csv", index=False, encoding="utf-8-sig")
    np.save(output_dir / "cutoff_hidden_states_float16.npy", hidden)
    manifest = {
        "model": model_key,
        "model_path": cfg.model_path,
        "model_path_exists": bool(Path(cfg.model_path).exists()),
        "decision_layers": list(spec["decision_layers"]),
        "cutoff": cutoff,
        "recent_layers": list(needed_layers),
        "n_graphs": int(cfg.n_graphs),
        "n_prompts": int(len(frame)),
        "prompt_sha256": sha256(output_dir / "prompt_readbacks.csv"),
        "hidden_sha256": sha256(output_dir / "cutoff_hidden_states_float16.npy"),
        "executor_sha256": sha256(EXECUTOR_SOURCE),
        "final_layer_used": False,
    }
    (output_dir / "extraction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return frame, hidden, manifest


def evaluate_model(model_key: str, frame: pd.DataFrame, hidden: np.ndarray, args: argparse.Namespace, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec = MODEL_SPECS[model_key]
    groups = frame["graph_id"].to_numpy(int)
    target = frame["mechanism_id"].to_numpy(int)
    raw_columns = [f"R_L{layer}" for layer in spec["recent_layers"]]
    matched_columns = [f"dR_L{layer}" for layer in spec["recent_layers"]]
    prefix_columns = [f"dR_L{layer}" for layer in range(int(spec["cutoff"]) + 1)]
    feature_rows = []
    prediction_rows = []
    split_rows = []
    oof = {name: np.full(len(frame), -1, dtype=int) for name in FEATURES}
    splitter = GroupKFold(n_splits=args.folds)

    for fold, (train, test) in enumerate(splitter.split(frame, target, groups)):
        train_y = target[train]
        train_prefix = frame.iloc[train][prefix_columns].to_numpy(np.float32)
        full_prefix = frame[prefix_columns].to_numpy(np.float32)
        pca, delta_pre = pca_component_with_training_orientation(train_prefix, full_prefix, train_y)
        feature_data = {
            "raw_current": frame[[raw_columns[-1]]].to_numpy(np.float32),
            "raw_recent": frame[raw_columns].to_numpy(np.float32),
            "matched_current": frame[[matched_columns[-1]]].to_numpy(np.float32),
            "matched_recent": frame[matched_columns].to_numpy(np.float32),
            "DeltaU_pre": delta_pre[:, None],
            "ordered_prefix": full_prefix,
            "cutoff_hidden": hidden.astype(np.float32),
        }
        split_rows.extend(
            {
                "model": model_key,
                "fold": fold,
                "graph_id": int(graph_id),
                "split": "test" if idx in set(test.tolist()) else "train",
            }
            for idx, graph_id in enumerate(groups)
        )
        for feature_index, name in enumerate(FEATURES):
            data = feature_data[name]
            clf = hidden_classifier(len(train), data.shape[1], args.seed + fold * 100 + feature_index) if name == "cutoff_hidden" else logistic(args.seed + fold * 100 + feature_index)
            clf.fit(data[train], train_y)
            predicted = clf.predict(data[test]).astype(int)
            oof[name][test] = predicted
            fold_f1 = f1_score(target[test], predicted, labels=(0, 1, 2), average="macro", zero_division=0)
            feature_rows.append(
                {
                    "model": model_key,
                    "fold": fold,
                    "feature": name,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "test_macro_f1": float(fold_f1),
                    "delta_u_pre_explained_variance": float(pca.explained_variance_ratio_[0]),
                }
            )

    if any(np.any(prediction < 0) for prediction in oof.values()):
        raise RuntimeError("Incomplete out-of-fold predictions")
    for idx, row in frame.iterrows():
        record = {
            "model": model_key,
            "prompt_id": row["prompt_id"],
            "graph_id": int(row["graph_id"]),
            "condition": row["condition"],
            "mechanism": row["mechanism"],
            "target": int(target[idx]),
        }
        record.update({f"prediction_{name}": int(oof[name][idx]) for name in FEATURES})
        prediction_rows.append(record)

    performance = []
    for name in FEATURES:
        performance.append(
            {
                "model": model_key,
                "feature": name,
                "oof_macro_f1": float(f1_score(target, oof[name], labels=(0, 1, 2), average="macro", zero_division=0)),
            }
        )
    comparisons = []
    for contrast_index, (candidate_name, reference_name) in enumerate(PRIMARY_CONTRASTS):
        audit = paired_group_sign_flip(
            oof[reference_name],
            oof[candidate_name],
            target,
            groups,
            args.sign_flips,
            args.seed + 10000 * (list(MODEL_SPECS).index(model_key) + 1) + contrast_index,
        )
        comparisons.append(
            {
                "model": model_key,
                "candidate": candidate_name,
                "reference": reference_name,
                **audit,
            }
        )
    return pd.DataFrame(performance), pd.DataFrame(prediction_rows), pd.DataFrame(comparisons), pd.DataFrame(feature_rows), pd.DataFrame(split_rows)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    executor = load_executor()
    performance_frames = []
    prediction_frames = []
    comparison_frames = []
    fold_frames = []
    split_frames = []
    manifests = []
    for model_key in args.models:
        print(f"[EXTRACT] {model_key}", flush=True)
        model_dir = args.output_root / model_key
        frame, hidden, manifest = extract_model(executor, model_key, args, model_dir)
        print(f"[EVALUATE] {model_key}", flush=True)
        performance, predictions, comparisons, folds, splits = evaluate_model(model_key, frame, hidden, args, model_dir)
        performance_frames.append(performance)
        prediction_frames.append(predictions)
        comparison_frames.append(comparisons)
        fold_frames.append(folds)
        split_frames.append(splits)
        manifests.append(manifest)

    performance = pd.concat(performance_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    comparisons = pd.concat(comparison_frames, ignore_index=True)
    folds = pd.concat(fold_frames, ignore_index=True)
    splits = pd.concat(split_frames, ignore_index=True)
    comparisons["q"] = bh_adjust(comparisons["p"].to_numpy(float))
    gate_rows = []
    for model_key, group in comparisons.groupby("model", sort=True):
        primary = group[group["candidate"].isin(("DeltaU_pre", "ordered_prefix"))]
        passed = bool(len(primary) == 2 and (primary["macro_f1_difference"] > 0).all() and (primary["q"] <= 0.05).all())
        gate_rows.append({"model": model_key, "supports_distinct_predecision_ordered_readback": passed})
    gates = pd.DataFrame(gate_rows)
    config = {
        "protocol": "v0.56 PROTOCOL.md",
        "models": args.models,
        "folds": args.folds,
        "graphs": 4 if args.smoke else args.graphs,
        "sign_flips": args.sign_flips,
        "seed": args.seed,
        "primary_contrasts": [list(row) for row in PRIMARY_CONTRASTS],
        "bh_family": "nine checkpoint-by-contrast tests in the coordinate-audit family",
        "target": "prompt-defined mechanism regime",
        "final_layer_used": False,
        "final_output_target_used": False,
    }
    performance.to_csv(args.output_root / "coordinate_feature_performance.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(args.output_root / "coordinate_oof_predictions.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(args.output_root / "coordinate_primary_contrasts.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(args.output_root / "coordinate_fold_performance.csv", index=False, encoding="utf-8-sig")
    splits.drop_duplicates().to_csv(args.output_root / "coordinate_group_splits.csv", index=False, encoding="utf-8-sig")
    gates.to_csv(args.output_root / "coordinate_gates.csv", index=False, encoding="utf-8-sig")
    (args.output_root / "analysis_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output_root / "model_extraction_manifests.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(comparisons.to_string(index=False), flush=True)
    print(gates.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
