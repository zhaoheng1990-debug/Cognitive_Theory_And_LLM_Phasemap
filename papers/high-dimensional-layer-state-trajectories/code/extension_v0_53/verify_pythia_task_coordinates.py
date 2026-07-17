"""Independent verification of the held-out Pythia learned-metric audit."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
from transformers import AutoModelForCausalLM


REVISIONS = (
    "step0", "step128", "step1000", "step4000", "step16000", "step64000", "step143000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir", type=Path,
        default=Path("outputs/pythia_training_coordinates"),
    )
    parser.add_argument(
        "--state-root", type=Path,
        default=Path("inputs/pythia_training_states"),
    )
    parser.add_argument("--repo-id", default="EleutherAI/pythia-160m")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("models")
    )
    parser.add_argument("--seed", type=int, default=2026071713)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def word_folds(subset: pd.DataFrame) -> np.ndarray:
    assignment = {}
    for category in sorted(subset["category"].unique()):
        words = sorted(subset.loc[subset["category"].eq(category), "clean_label"].unique())
        for index, word in enumerate(words):
            assignment[word] = index % 3
    return subset["clean_label"].map(assignment).to_numpy(int)


def permutations(n_layers: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.arange(n_layers)
    result = np.tile(base, (n, 1))
    for index in range(n):
        result[index, 1:-1] = rng.permutation(base[1:-1])
    return result


def mean_row_spearman(values: np.ndarray) -> float:
    depth = rankdata(np.arange(values.shape[1]))
    depth = depth - depth.mean()
    depth_norm = np.sqrt(np.sum(depth**2))
    correlations = []
    for row in values:
        ranked = rankdata(row)
        ranked = ranked - ranked.mean()
        denominator = depth_norm * np.sqrt(np.sum(ranked**2))
        correlations.append(float(np.sum(depth * ranked) / denominator) if denominator else 0.0)
    return float(np.mean(correlations))


def order_values(values: np.ndarray, perms: np.ndarray) -> dict[str, float]:
    real = mean_row_spearman(values)
    null = np.array([mean_row_spearman(values[:, order]) for order in perms])
    return {
        "native_mean_prompt_spearman": real,
        "shuffle_q99": float(np.quantile(null, 0.99)),
        "real_minus_shuffle_mean": float(real - null.mean()),
        "empirical_p_upper": float((1 + np.sum(null >= real)) / (len(null) + 1)),
    }


def output_weight(repo: str, revision: str, cache: Path, token_ids: np.ndarray) -> np.ndarray:
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=revision, cache_dir=cache, local_files_only=True,
        dtype=torch.float16, low_cpu_mem_usage=True,
    )
    values = model.get_output_embeddings().weight[token_ids].detach().float().cpu().numpy()
    del model
    gc.collect()
    return values


def candidate_order(
    states: np.ndarray,
    clean_weight: np.ndarray,
    conflict_weight: np.ndarray,
    perms: np.ndarray,
) -> dict[str, float]:
    unit_states = normalize(states.reshape(-1, states.shape[-1])).reshape(states.shape)
    clean = normalize(clean_weight)
    conflict = normalize(conflict_weight)
    evidence = np.einsum("pld,pd->pl", unit_states, clean) - np.einsum(
        "pld,pd->pl", unit_states, conflict
    )
    return order_values(evidence, perms)


def fit_metric(states: np.ndarray, labels: np.ndarray, seed: int):
    values = normalize(states.astype(np.float64))
    pca = PCA(n_components=min(16, len(values) - 1), svd_solver="full", random_state=seed)
    reduced = pca.fit_transform(values)
    classifier = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=2000, class_weight="balanced", random_state=seed
    ).fit(reduced, labels)
    return pca, classifier


def crossvalidated_logits(
    fit_states: np.ndarray,
    apply_states: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    seed: int,
) -> np.ndarray:
    n_classes = int(labels.max()) + 1
    output = np.zeros((len(labels), apply_states.shape[1], n_classes), dtype=float)
    for fold in range(3):
        train = folds != fold
        test = folds == fold
        pca, classifier = fit_metric(fit_states[train, -1], labels[train], seed + fold)
        shape = apply_states[test].shape
        flat = normalize(apply_states[test].reshape(-1, shape[-1]).astype(np.float64))
        scores = classifier.decision_function(pca.transform(flat))
        mapped = np.full((len(flat), n_classes), -1e6, dtype=float)
        for source, label in enumerate(classifier.classes_):
            mapped[:, int(label)] = scores[:, source]
        output[test] = mapped.reshape(shape[0], shape[1], n_classes)
    return output


def macro_f1(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(f1_score(labels, np.argmax(logits[:, -1], axis=1), average="macro"))


def main() -> None:
    args = parse_args()
    native_path = args.result_dir / "model_native_candidate_metric_summary.csv"
    within_path = args.result_dir / "within_checkpoint_category_metric_summary.csv"
    transfer_path = args.result_dir / "final_metric_transfer_summary.csv"
    null_path = args.result_dir / "category_label_permutation_nulls.csv"
    decision_path = args.result_dir / "gate_decision.json"
    metadata_path = args.result_dir / "run_metadata.json"
    native = pd.read_csv(native_path).set_index("revision")
    within = pd.read_csv(within_path).set_index("revision")
    transfer = pd.read_csv(transfer_path).set_index("apply_revision")
    nulls = pd.read_csv(null_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    subset = pd.read_csv(args.state_root / "phase4_lexical_subset.csv")
    folds = word_folds(subset)
    categories = sorted(subset["category"].unique())
    labels = subset["category"].map({value: index for index, value in enumerate(categories)}).to_numpy(int)

    checks = {}
    checks["metadata_matches_frozen_counts"] = bool(
        metadata["n_prompts"] == 53 and metadata["n_layers"] == 12
        and metadata["layer_shuffles"] == 999 and metadata["label_nulls"] == 199
    )
    checks["fold_file_has_53_rows"] = len(pd.read_csv(args.result_dir / "heldout_word_folds.csv")) == 53
    leakage_free = True
    category_complete = True
    for fold in range(3):
        train = subset.iloc[np.flatnonzero(folds != fold)]
        test = subset.iloc[np.flatnonzero(folds == fold)]
        leakage_free &= not bool(set(train["clean_label"]) & set(test["clean_label"]))
        category_complete &= set(train["category"]) == set(categories)
        category_complete &= set(test["category"]) == set(categories)
    checks["answer_word_groups_do_not_leak"] = leakage_free
    checks["all_categories_present_in_each_train_and_test_fold"] = category_complete

    clean_ids = subset["clean_token_id"].to_numpy(int)
    conflict_ids = subset["conflict_token_id"].to_numpy(int)
    all_ids = np.concatenate([clean_ids, conflict_ids])
    perms = permutations(12, 999, args.seed)
    candidate_recomputed = {}
    for revision in REVISIONS:
        states = np.asarray(np.load(
            args.state_root / revision / "hidden_last_token_layers_float16.npy", mmap_mode="r"
        ), dtype=np.float32)
        weights = output_weight(args.repo_id, revision, args.cache_dir, all_ids)
        clean, conflict = np.split(weights, 2)
        candidate_recomputed[revision] = candidate_order(states, clean, conflict, perms)
    candidate_error = max(
        abs(candidate_recomputed[revision][metric] - float(native.loc[revision, metric]))
        for revision in REVISIONS
        for metric in candidate_recomputed[revision]
    )
    checks["candidate_coordinate_recomputes"] = candidate_error < 1e-10

    final_states = np.asarray(np.load(
        args.state_root / "step143000" / "hidden_last_token_layers_float16.npy"
    ), dtype=np.float32)
    early_states = np.asarray(np.load(
        args.state_root / "step0" / "hidden_last_token_layers_float16.npy"
    ), dtype=np.float32)
    within_early = macro_f1(
        crossvalidated_logits(early_states, early_states, labels, folds, args.seed + 1000), labels
    )
    within_final = macro_f1(
        crossvalidated_logits(final_states, final_states, labels, folds, args.seed + 1000), labels
    )
    transfer_early = macro_f1(
        crossvalidated_logits(final_states, early_states, labels, folds, args.seed + 2000), labels
    )
    transfer_final = macro_f1(
        crossvalidated_logits(final_states, final_states, labels, folds, args.seed + 2000), labels
    )
    category_values = {
        "within_step0": within_early,
        "within_step143000": within_final,
        "transfer_step0": transfer_early,
        "transfer_step143000": transfer_final,
    }
    category_error = max(
        abs(within_early - float(within.loc["step0", "final_macro_f1"])),
        abs(within_final - float(within.loc["step143000", "final_macro_f1"])),
        abs(transfer_early - float(transfer.loc["step0", "final_macro_f1"])),
        abs(transfer_final - float(transfer.loc["step143000", "final_macro_f1"])),
    )
    checks["heldout_category_f1_recomputes"] = category_error < 1e-12

    null_q95 = {}
    for analysis in ("within_checkpoint", "final_metric_transfer"):
        null_q95[analysis] = {}
        for revision in ("step0", "step143000"):
            values = nulls[
                nulls["analysis"].eq(analysis) & nulls["revision"].eq(revision)
            ]["final_macro_f1"]
            null_q95[analysis][revision] = float(np.quantile(values, 0.95))
    checks["label_null_has_expected_rows"] = len(nulls) == 199 * 4
    checks["label_null_quantiles_match"] = all(
        abs(null_q95[analysis][revision] - decision["label_null_q95"][analysis][revision]) < 1e-12
        for analysis in null_q95 for revision in null_q95[analysis]
    )

    behavior_rho = float(spearmanr(
        native.loc[list(REVISIONS), "candidate_pair_accuracy"],
        native.loc[list(REVISIONS), "real_minus_shuffle_mean"],
    ).statistic)
    independent_gates = {
        "candidate_final_exceeds_shuffle_q99": bool(
            candidate_recomputed["step143000"]["native_mean_prompt_spearman"]
            > candidate_recomputed["step143000"]["shuffle_q99"]
        ),
        "candidate_training_order_gain_at_least_0_10": bool(
            candidate_recomputed["step143000"]["real_minus_shuffle_mean"]
            - candidate_recomputed["step0"]["real_minus_shuffle_mean"] >= 0.10
        ),
        "candidate_order_gain_behavior_rho_at_least_0_60": bool(behavior_rho >= 0.60),
        "within_final_f1_exceeds_label_q95_and_step0_by_0_10": bool(
            within_final > null_q95["within_checkpoint"]["step143000"]
            and within_final - within_early >= 0.10
        ),
        "transfer_final_f1_exceeds_label_q95_and_step0_by_0_10": bool(
            transfer_final > null_q95["final_metric_transfer"]["step143000"]
            and transfer_final - transfer_early >= 0.10
        ),
    }
    checks["independent_gates_match"] = independent_gates == decision["gates"]

    output = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "candidate_recalculation_max_error": candidate_error,
        "category_f1_recalculation_max_error": category_error,
        "recalculated_category_f1": category_values,
        "recalculated_candidate_behavior_rho": behavior_rho,
        "independent_gates": independent_gates,
        "recalculated_label_null_q95": null_q95,
        "source_hashes": {
            path.name: sha256(path)
            for path in (native_path, within_path, transfer_path, null_path, decision_path, metadata_path)
        },
    }
    if args.report is not None:
        args.report.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    if output["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
