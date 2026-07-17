"""Audit task-conditioned metrics across a documented Pythia training run."""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
import transformers
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
from transformers import AutoModelForCausalLM


DEFAULT_REVISIONS = (
    "step0", "step128", "step1000", "step4000", "step16000", "step64000", "step143000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="EleutherAI/pythia-160m")
    parser.add_argument("--revisions", nargs="+", default=list(DEFAULT_REVISIONS))
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path(os.environ.get("HF_TRAINING_CACHE", "models")),
    )
    parser.add_argument(
        "--state-root", type=Path,
        default=Path("inputs/pythia_training_states"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/pythia_training_coordinates"),
    )
    parser.add_argument("--layer-shuffles", type=int, default=999)
    parser.add_argument("--label-nulls", type=int, default=199)
    parser.add_argument("--seed", type=int, default=2026071713)
    return parser.parse_args()


def make_word_folds(subset: pd.DataFrame, n_splits: int = 3) -> np.ndarray:
    fold_by_word: dict[str, int] = {}
    for category, frame in subset.groupby("category", sort=True):
        words = sorted(frame["clean_label"].astype(str).unique())
        for index, word in enumerate(words):
            fold_by_word[word] = index % n_splits
    folds = subset["clean_label"].astype(str).map(fold_by_word).to_numpy(dtype=int)
    for fold in range(n_splits):
        test = subset.iloc[np.flatnonzero(folds == fold)]
        train = subset.iloc[np.flatnonzero(folds != fold)]
        if set(test["category"]) != set(subset["category"]):
            raise RuntimeError(f"Fold {fold} lacks a held-out category")
        if set(train["category"]) != set(subset["category"]):
            raise RuntimeError(f"Fold {fold} lacks a training category")
        if set(test["clean_label"]) & set(train["clean_label"]):
            raise RuntimeError(f"Fold {fold} leaks clean answer words")
    return folds


def fixed_interior_permutations(
    n_layers: int, n_shuffles: int, rng: np.random.Generator
) -> np.ndarray:
    base = np.arange(n_layers)
    permutations = np.tile(base, (n_shuffles, 1))
    for index in range(n_shuffles):
        permutations[index, 1:-1] = rng.permutation(base[1:-1])
    return permutations


def row_spearman(values: np.ndarray) -> np.ndarray:
    depth = np.arange(values.shape[1], dtype=float)
    depth_rank = rankdata(depth)
    centered_depth = depth_rank - depth_rank.mean()
    denominator_depth = np.sqrt(np.sum(centered_depth**2))
    output = np.zeros(values.shape[0], dtype=float)
    for index, row in enumerate(values):
        row_rank = rankdata(row)
        centered = row_rank - row_rank.mean()
        denominator = denominator_depth * np.sqrt(np.sum(centered**2))
        output[index] = float(np.sum(centered_depth * centered) / denominator) if denominator else 0.0
    return output


def order_audit(values: np.ndarray, permutations: np.ndarray) -> dict[str, object]:
    real_prompt = row_spearman(values)
    real = float(real_prompt.mean())
    null = np.empty(len(permutations), dtype=float)
    for index, permutation in enumerate(permutations):
        null[index] = float(row_spearman(values[:, permutation]).mean())
    p_upper = float((1 + np.sum(null >= real)) / (len(null) + 1))
    return {
        "native_mean_prompt_spearman": real,
        "shuffle_mean": float(null.mean()),
        "shuffle_q95": float(np.quantile(null, 0.95)),
        "shuffle_q99": float(np.quantile(null, 0.99)),
        "real_minus_shuffle_mean": float(real - null.mean()),
        "empirical_p_upper": p_upper,
        "prompt_spearman": real_prompt,
        "null_means": null,
    }


def fit_fold_metric(
    train_states: np.ndarray,
    train_labels: np.ndarray,
    seed: int,
) -> tuple[PCA, LogisticRegression]:
    normalized = normalize(train_states.astype(np.float64), norm="l2")
    n_components = min(16, normalized.shape[0] - 1, normalized.shape[1])
    pca = PCA(n_components=n_components, svd_solver="full", random_state=seed)
    reduced = pca.fit_transform(normalized)
    classifier = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=2000, class_weight="balanced",
        random_state=seed,
    )
    classifier.fit(reduced, train_labels)
    return pca, classifier


def project_metric(
    states: np.ndarray,
    pca: PCA,
    classifier: LogisticRegression,
    all_classes: np.ndarray,
) -> np.ndarray:
    n_prompts, n_layers, hidden_size = states.shape
    flat = normalize(states.reshape(-1, hidden_size).astype(np.float64), norm="l2")
    decision = classifier.decision_function(pca.transform(flat))
    if decision.ndim == 1:
        decision = np.column_stack([-decision, decision])
    logits = np.full((len(flat), len(all_classes)), -1e6, dtype=float)
    class_to_index = {int(label): index for index, label in enumerate(all_classes)}
    for source_index, label in enumerate(classifier.classes_):
        logits[:, class_to_index[int(label)]] = decision[:, source_index]
    return logits.reshape(n_prompts, n_layers, len(all_classes))


def evidence_from_logits(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rows = np.arange(len(labels))
    true = logits[rows, :, labels]
    alternative_sum = logits.sum(axis=2) - true
    return true - alternative_sum / (logits.shape[2] - 1)


def crossvalidated_metric(
    fit_states: np.ndarray,
    apply_states: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    seed: int,
    permuted_labels: np.ndarray | None = None,
) -> np.ndarray:
    all_classes = np.arange(int(labels.max()) + 1)
    logits = np.zeros((len(labels), apply_states.shape[1], len(all_classes)), dtype=float)
    fit_labels = labels if permuted_labels is None else permuted_labels
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        pca, classifier = fit_fold_metric(
            fit_states[train, -1, :], fit_labels[train], seed + int(fold)
        )
        logits[test] = project_metric(apply_states[test], pca, classifier, all_classes)
    return logits


def metric_summary(
    logits: np.ndarray,
    labels: np.ndarray,
    permutations: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    final_prediction = np.argmax(logits[:, -1, :], axis=1)
    macro_f1 = float(f1_score(labels, final_prediction, average="macro", zero_division=0))
    accuracy = float(np.mean(final_prediction == labels))
    evidence = evidence_from_logits(logits, labels)
    audit = order_audit(evidence, permutations)
    summary = {
        "final_accuracy": accuracy,
        "final_macro_f1": macro_f1,
        "evidence_first_mean": float(evidence[:, 0].mean()),
        "evidence_final_mean": float(evidence[:, -1].mean()),
        "evidence_progress_mean": float((evidence[:, -1] - evidence[:, 0]).mean()),
        **{key: value for key, value in audit.items() if key not in ("prompt_spearman", "null_means")},
    }
    return summary, evidence


def permute_group_labels(
    labels: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    unique_groups = np.unique(groups)
    group_labels = np.array([labels[np.flatnonzero(groups == group)[0]] for group in unique_groups])
    permuted_group_labels = rng.permutation(group_labels)
    mapping = dict(zip(unique_groups, permuted_group_labels))
    return np.array([mapping[group] for group in groups], dtype=int)


def load_output_weights(
    repo_id: str, revision: str, cache_dir: Path, token_ids: np.ndarray
) -> np.ndarray:
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, revision=revision, cache_dir=cache_dir, local_files_only=True,
        dtype=torch.float16, low_cpu_mem_usage=True,
    )
    weight = model.get_output_embeddings().weight.detach().float().cpu().numpy()
    selected = weight[token_ids]
    del model, weight
    gc.collect()
    return selected


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    rng = np.random.default_rng(args.seed)
    subset = pd.read_csv(args.state_root / "phase4_lexical_subset.csv")
    categories = sorted(subset["category"].astype(str).unique())
    category_to_id = {category: index for index, category in enumerate(categories)}
    labels = subset["category"].astype(str).map(category_to_id).to_numpy(dtype=int)
    groups = subset["clean_label"].astype(str).to_numpy()
    folds = make_word_folds(subset)
    fold_frame = subset[["problem_id", "category", "clean_label", "conflict_label"]].copy()
    fold_frame["category_id"] = labels
    fold_frame["fold"] = folds
    fold_frame.to_csv(args.output_dir / "heldout_word_folds.csv", index=False, encoding="utf-8-sig")

    states_by_revision = {
        revision: np.load(
            args.state_root / revision / "hidden_last_token_layers_float16.npy",
            mmap_mode="r",
        )
        for revision in args.revisions
    }
    shapes = {revision: list(states.shape) for revision, states in states_by_revision.items()}
    if len({tuple(shape) for shape in shapes.values()}) != 1:
        raise RuntimeError(f"State shapes differ: {shapes}")
    n_prompts, n_layers, _ = next(iter(states_by_revision.values())).shape
    if n_prompts != len(subset):
        raise RuntimeError("Subset and state row counts differ")
    permutations = fixed_interior_permutations(n_layers, args.layer_shuffles, rng)

    native_rows = []
    native_prompt_rows = []
    for revision in args.revisions:
        states = np.asarray(states_by_revision[revision], dtype=np.float32)
        token_ids = np.concatenate([
            subset["clean_token_id"].to_numpy(dtype=int),
            subset["conflict_token_id"].to_numpy(dtype=int),
        ])
        selected_weights = load_output_weights(args.repo_id, revision, args.cache_dir, token_ids)
        clean_weight, conflict_weight = np.split(selected_weights, 2)
        state_unit = normalize(states.reshape(-1, states.shape[-1]), norm="l2").reshape(states.shape)
        clean_unit = normalize(clean_weight, norm="l2")
        conflict_unit = normalize(conflict_weight, norm="l2")
        evidence = np.einsum("pld,pd->pl", state_unit, clean_unit) - np.einsum(
            "pld,pd->pl", state_unit, conflict_unit
        )
        audit = order_audit(evidence, permutations)
        outcome = pd.read_csv(args.state_root / revision / "lexical_prompt_outcomes.csv")
        native_rows.append({
            "revision": revision,
            "training_step": int(revision.replace("step", "")),
            "candidate_pair_accuracy": float(outcome["pair_correct"].astype(bool).mean()),
            "candidate_margin_mean": float(outcome["candidate_margin"].mean()),
            "candidate_coordinate_first_mean": float(evidence[:, 0].mean()),
            "candidate_coordinate_final_mean": float(evidence[:, -1].mean()),
            "candidate_coordinate_progress_mean": float((evidence[:, -1] - evidence[:, 0]).mean()),
            **{key: value for key, value in audit.items() if key not in ("prompt_spearman", "null_means")},
        })
        for prompt_index in range(n_prompts):
            native_prompt_rows.append({
                "revision": revision,
                "prompt_index": prompt_index,
                "problem_id": str(subset.iloc[prompt_index]["problem_id"]),
                "prompt_spearman": float(audit["prompt_spearman"][prompt_index]),
                "coordinate_first": float(evidence[prompt_index, 0]),
                "coordinate_final": float(evidence[prompt_index, -1]),
            })
    native_summary = pd.DataFrame(native_rows)
    native_summary.to_csv(args.output_dir / "model_native_candidate_metric_summary.csv", index=False)
    pd.DataFrame(native_prompt_rows).to_csv(
        args.output_dir / "model_native_candidate_metric_prompt.csv", index=False
    )

    within_rows = []
    within_evidence: dict[str, np.ndarray] = {}
    for revision in args.revisions:
        states = np.asarray(states_by_revision[revision], dtype=np.float32)
        logits = crossvalidated_metric(states, states, labels, folds, args.seed + 1000)
        summary, evidence = metric_summary(logits, labels, permutations)
        within_rows.append({
            "revision": revision,
            "training_step": int(revision.replace("step", "")),
            **summary,
        })
        within_evidence[revision] = evidence
    within_summary = pd.DataFrame(within_rows)
    within_summary.to_csv(args.output_dir / "within_checkpoint_category_metric_summary.csv", index=False)

    final_revision = args.revisions[-1]
    final_fit_states = np.asarray(states_by_revision[final_revision], dtype=np.float32)
    transfer_rows = []
    transfer_evidence: dict[str, np.ndarray] = {}
    for revision in args.revisions:
        apply_states = np.asarray(states_by_revision[revision], dtype=np.float32)
        logits = crossvalidated_metric(
            final_fit_states, apply_states, labels, folds, args.seed + 2000
        )
        summary, evidence = metric_summary(logits, labels, permutations)
        transfer_rows.append({
            "fit_revision": final_revision,
            "apply_revision": revision,
            "training_step": int(revision.replace("step", "")),
            **summary,
        })
        transfer_evidence[revision] = evidence
    transfer_summary = pd.DataFrame(transfer_rows)
    transfer_summary.to_csv(args.output_dir / "final_metric_transfer_summary.csv", index=False)

    label_null_rows = []
    null_targets = (args.revisions[0], final_revision)
    for null_index in range(args.label_nulls):
        permuted = permute_group_labels(labels, groups, rng)
        for revision in null_targets:
            states = np.asarray(states_by_revision[revision], dtype=np.float32)
            logits = crossvalidated_metric(
                states, states, labels, folds, args.seed + 10_000 + null_index,
                permuted_labels=permuted,
            )
            summary, _ = metric_summary(logits, labels, permutations[:99])
            label_null_rows.append({
                "analysis": "within_checkpoint",
                "revision": revision,
                "null_index": null_index,
                **summary,
            })
        for revision in null_targets:
            apply_states = np.asarray(states_by_revision[revision], dtype=np.float32)
            logits = crossvalidated_metric(
                final_fit_states, apply_states, labels, folds,
                args.seed + 20_000 + null_index, permuted_labels=permuted,
            )
            summary, _ = metric_summary(logits, labels, permutations[:99])
            label_null_rows.append({
                "analysis": "final_metric_transfer",
                "revision": revision,
                "null_index": null_index,
                **summary,
            })
    label_null = pd.DataFrame(label_null_rows)
    label_null.to_csv(args.output_dir / "category_label_permutation_nulls.csv", index=False)

    native_indexed = native_summary.set_index("revision")
    final_native = native_indexed.loc[final_revision]
    early_native = native_indexed.loc[args.revisions[0]]
    behavior_rho = float(spearmanr(
        native_summary["candidate_pair_accuracy"],
        native_summary["real_minus_shuffle_mean"],
    ).statistic)
    within_indexed = within_summary.set_index("revision")
    transfer_indexed = transfer_summary.set_index("apply_revision")

    def null_q95(analysis: str, revision: str, metric: str) -> float:
        values = label_null[
            label_null["analysis"].eq(analysis) & label_null["revision"].eq(revision)
        ][metric]
        return float(np.quantile(values, 0.95))

    gates = {
        "candidate_final_exceeds_shuffle_q99": bool(
            final_native["native_mean_prompt_spearman"] > final_native["shuffle_q99"]
        ),
        "candidate_training_order_gain_at_least_0_10": bool(
            final_native["real_minus_shuffle_mean"]
            - early_native["real_minus_shuffle_mean"] >= 0.10
        ),
        "candidate_order_gain_behavior_rho_at_least_0_60": bool(behavior_rho >= 0.60),
        "within_final_f1_exceeds_label_q95_and_step0_by_0_10": bool(
            within_indexed.loc[final_revision, "final_macro_f1"]
            > null_q95("within_checkpoint", final_revision, "final_macro_f1")
            and within_indexed.loc[final_revision, "final_macro_f1"]
            - within_indexed.loc[args.revisions[0], "final_macro_f1"] >= 0.10
        ),
        "transfer_final_f1_exceeds_label_q95_and_step0_by_0_10": bool(
            transfer_indexed.loc[final_revision, "final_macro_f1"]
            > null_q95("final_metric_transfer", final_revision, "final_macro_f1")
            and transfer_indexed.loc[final_revision, "final_macro_f1"]
            - transfer_indexed.loc[args.revisions[0], "final_macro_f1"] >= 0.10
        ),
    }
    decision = {
        "candidate_coordinate_gates_all_pass": bool(all(list(gates.values())[:3])),
        "heldout_category_metric_any_pass": bool(
            gates["within_final_f1_exceeds_label_q95_and_step0_by_0_10"]
            or gates["transfer_final_f1_exceeds_label_q95_and_step0_by_0_10"]
        ),
        "gates": gates,
        "candidate_order_gain_accuracy_spearman": behavior_rho,
        "label_null_q95": {
            analysis: {
                revision: null_q95(analysis, revision, "final_macro_f1")
                for revision in null_targets
            }
            for analysis in ("within_checkpoint", "final_metric_transfer")
        },
        "boundary": (
            "Task-conditioned coordinates are diagnostics on one lexical cloze set; "
            "they do not replace the high-dimensional process object or establish a "
            "universal semantic metric."
        ),
    }
    (args.output_dir / "gate_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    metadata = {
        "repo_id": args.repo_id,
        "revisions": args.revisions,
        "state_shapes": shapes,
        "n_prompts": n_prompts,
        "n_layers": n_layers,
        "n_categories": len(categories),
        "categories": categories,
        "fold_rule": "clean-label grouped, category-stratified deterministic modulo-3",
        "pca_components": 16,
        "logistic_C": 1.0,
        "layer_shuffles": args.layer_shuffles,
        "label_nulls": args.label_nulls,
        "seed": args.seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
