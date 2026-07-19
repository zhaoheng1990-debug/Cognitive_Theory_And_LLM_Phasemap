#!/usr/bin/env python
"""Frozen-protocol local-transition intervention on mixed arithmetic prompts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
EXECUTOR_SOURCE = CODE_ROOT / "extension_v0_52" / "local_transition_executor.py"
CONTROLLER_SOURCE = CODE_ROOT / "extension_v0_52" / "boundary_controller.py"
ARITHMETIC_SOURCE = HERE / "mixed_arithmetic_task.py"

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

PRIMARY_CONTROL = "guarded_policy"
CONTROL_ORDER = (
    "baseline",
    "fixed_global",
    "ungated_policy",
    PRIMARY_CONTROL,
    "label_shuffle_policy",
    "operator_only",
    "precursor_only",
    "uniform_max",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_WINDOWS), default=list(MODEL_WINDOWS))
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument("--problems", type=int, default=64)
    parser.add_argument("--address-nulls", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--interface", choices=("raw", "chat"), default="raw")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def bool_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.copy()
    return series.astype(str).str.lower().eq("true")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return centre - half, centre + half


def build_problem_table(arithmetic, n_problems: int, seed: int, max_value: int) -> pd.DataFrame:
    if n_problems > 64:
        raise ValueError("The frozen contextual-token task contains at most 64 problems")
    pools: dict[str, list[tuple[str, int]]] = {"addition": [], "subtraction": [], "multiplication": []}
    for left in range(0, max_value + 1):
        for right in range(0, max_value + 1):
            if left + right <= max_value:
                pools["addition"].append((f"{left} + {right}", left + right))
            if left >= right and left - right <= max_value:
                pools["subtraction"].append((f"{left} - {right}", left - right))
            if left >= 1 and right >= 1 and left * right <= max_value:
                pools["multiplication"].append((f"{left} * {right}", left * right))

    rng = np.random.default_rng(seed)
    for operation in pools:
        rng.shuffle(pools[operation])
    quotas = {"addition": 22, "subtraction": 21, "multiplication": 21}
    selected = []
    for operation in ("addition", "subtraction", "multiplication"):
        selected.extend((operation, expression, answer) for expression, answer in pools[operation][: quotas[operation]])
    rng.shuffle(selected)
    selected = selected[:n_problems]

    offsets = (1, -1, 2, -2, 3, -3)
    rows = []
    for problem_id, (operation, expression, answer) in enumerate(selected):
        offset = offsets[problem_id % len(offsets)]
        distractor = (answer + offset) % (max_value + 1)
        if distractor == answer:
            distractor = (answer + 1) % (max_value + 1)
        rows.append(
            {
                "problem_id": problem_id,
                "operation": operation,
                "expression": expression,
                "correct_value": answer,
                "distractor_value": distractor,
                "clean_label": arithmetic.NUMBER_WORDS[answer],
                "conflict_label": arithmetic.NUMBER_WORDS[distractor],
            }
        )
    table = pd.DataFrame(rows)
    if table["expression"].nunique() != len(table):
        raise RuntimeError("Frozen arithmetic expressions are not unique")
    return table


def candidate_token_id(tokenizer, user_prompt: str, label: str, interface: str) -> tuple[int, dict]:
    continuation = label if interface == "chat" else " " + label
    label_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    if len(label_ids) != 1:
        raise RuntimeError(f"Context candidate {label!r} is not a single token: {label_ids}")
    if interface == "chat":
        prefix_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": label},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    else:
        prefix_text = user_prompt
        full_text = user_prompt + " " + label
        prefix_ids = tokenizer(prefix_text, add_special_tokens=True)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=True)["input_ids"]
    prefix_exact = full_ids[: len(prefix_ids)] == prefix_ids
    first_appended = int(full_ids[len(prefix_ids)]) if prefix_exact and len(full_ids) > len(prefix_ids) else -1
    token_id = int(label_ids[0])
    if not prefix_exact or first_appended != token_id:
        raise RuntimeError(
            f"Context token mismatch for {label!r}: prefix_exact={prefix_exact}, "
            f"first_appended={first_appended}, standalone={token_id}"
        )
    return token_id, {
        "label": label,
        "token_id": token_id,
        "decoded": tokenizer.decode([token_id]),
        "prefix_exact": prefix_exact,
        "first_appended_matches": first_appended == token_id,
        "single_token": True,
        "interface": interface,
    }


def build_task_dataset(
    arithmetic, tokenizer, n_problems: int, seed: int, interface: str
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    max_value = 20 if interface == "raw" else 10
    problems = build_problem_table(arithmetic, n_problems, seed, max_value)
    rows = []
    for problem in problems.itertuples(index=False):
        candidate_one = problem.clean_label if problem.problem_id % 2 == 0 else problem.conflict_label
        candidate_two = problem.clean_label if problem.problem_id % 2 == 1 else problem.conflict_label
        for condition, mechanism in arithmetic.CONDITIONS:
            rows.append(
                {
                    "prompt_id": f"arith_ctx_{problem.problem_id:03d}_{condition}",
                    "problem_id": int(problem.problem_id),
                    "operation": problem.operation,
                    "condition": condition,
                    "mechanism": mechanism,
                    "mechanism_id": arithmetic.MECHANISM_ID[mechanism],
                    "expression": problem.expression,
                    "clean_label": problem.clean_label,
                    "conflict_label": problem.conflict_label,
                    "prompt": arithmetic.make_prompt(
                        condition,
                        problem.expression,
                        candidate_one,
                        candidate_two,
                        problem.conflict_label,
                    ),
                }
            )
    source = pd.DataFrame(rows)
    source["graph_id"] = source["problem_id"].astype(int)
    if interface == "chat":
        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError("Chat-interface diagnostic requires a checkpoint-native template")
        source["text"] = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in source["prompt"]
        ]
        wrapping = "checkpoint-native chat template"
    else:
        source["text"] = source["prompt"]
        wrapping = "raw prompt matching the relation-graph actuator boundary"
    audit_rows = []
    clean_ids = []
    conflict_ids = []
    for row in source.itertuples(index=False):
        clean_id, clean_audit = candidate_token_id(
            tokenizer, row.prompt, row.clean_label, interface
        )
        conflict_id, conflict_audit = candidate_token_id(
            tokenizer, row.prompt, row.conflict_label, interface
        )
        clean_ids.append(clean_id)
        conflict_ids.append(conflict_id)
        audit_rows.append({"prompt_id": row.prompt_id, "role": "clean", **clean_audit})
        audit_rows.append({"prompt_id": row.prompt_id, "role": "conflict", **conflict_audit})
    source["clean_token_id"] = clean_ids
    source["conflict_token_id"] = conflict_ids
    dataset = source[
        [
            "prompt_id",
            "graph_id",
            "problem_id",
            "operation",
            "condition",
            "mechanism",
            "mechanism_id",
            "expression",
            "clean_label",
            "conflict_label",
            "clean_token_id",
            "conflict_token_id",
            "prompt",
            "text",
        ]
    ].copy()
    return dataset, pd.DataFrame(audit_rows), wrapping


def refit_train_coordinate(controller, executor, base: pd.DataFrame, train_index: np.ndarray, cfg, out_dir: Path):
    out, pca = controller.refit_delta_u_on_train(executor, base, train_index, cfg, out_dir)
    out["D_B_proxy"] = out[[f"dR_L{layer}" for layer in cfg.decision_layers]].abs().min(axis=1)
    return out, pca


def confidence_gate(actions: pd.DataFrame, predicted_features: pd.DataFrame) -> pd.DataFrame:
    probs = predicted_features[["p_mech_stable", "p_mech_competition", "p_mech_closure"]].to_numpy(float)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1) / math.log(3.0)
    source = predicted_features[["prompt_id", "p_mech_closure"]].copy()
    source["entropy"] = entropy
    prompt_gate = source.groupby("prompt_id", as_index=False).agg(
        mean_p_closure=("p_mech_closure", "mean"),
        mean_entropy=("entropy", "mean"),
    )
    prompt_gate["gate_open"] = (
        prompt_gate["mean_p_closure"].ge(0.50) & prompt_gate["mean_entropy"].le(0.90)
    )
    out = actions.merge(prompt_gate, on="prompt_id", how="left")
    closed = ~out["gate_open"].fillna(False)
    out.loc[closed, ["alpha", "beta"]] = 0.0
    out.loc[closed, "action_label"] = "a0.0_b0.0_confidence_gate"
    return out


def component_actions(guarded: pd.DataFrame) -> dict[str, pd.DataFrame]:
    guarded = guarded.copy()
    open_mask = bool_mask(guarded["gate_open"])

    operator_only = guarded.copy()
    operator_only["beta"] = 0.0
    operator_only["action_label"] = np.where(open_mask, "gated_operator_only", "gated_zero")

    precursor_only = guarded.copy()
    precursor_only["alpha"] = 0.0
    precursor_only["action_label"] = np.where(open_mask, "gated_precursor_only", "gated_zero")

    uniform_max = guarded.copy()
    uniform_max.loc[:, ["alpha", "beta"]] = 0.0
    uniform_max.loc[open_mask, "alpha"] = 1.0
    uniform_max.loc[open_mask, "beta"] = 1.2
    uniform_max["action_label"] = np.where(open_mask, "gated_uniform_a1.0_b1.2", "gated_zero")

    return {
        "operator_only": operator_only,
        "precursor_only": precursor_only,
        "uniform_max": uniform_max,
    }


def evaluate_full_vocab_and_coordinate(
    executor,
    model,
    tokenizer,
    test_df: pd.DataFrame,
    cfg,
    pca,
    clean_anchors: dict[int, dict[int, float]],
    stable_ops,
    precursor_dirs,
    control: str,
    actions: pd.DataFrame | None,
    out_dir: Path,
    cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_path = out_dir / f"full_vocab_{control}.csv"
    trace_path = out_dir / f"trace_{control}.csv"
    if cache and result_path.exists():
        cached = pd.read_csv(result_path)
        if len(cached) == len(test_df):
            trace = pd.DataFrame()
            if trace_path.exists() and trace_path.stat().st_size:
                try:
                    trace = pd.read_csv(trace_path)
                except pd.errors.EmptyDataError:
                    trace = pd.DataFrame()
            print(f"[{cfg.model_key}] reuse {control}", flush=True)
            return cached, trace

    print(f"[{cfg.model_key}] evaluate {control}", flush=True)
    weight = executor.get_lm_head_weight(model).detach().float().to(model.device)
    action_lookup = executor.make_action_lookup(actions)
    layer_modules = executor.get_layers(model)
    hook_layers = sorted(set(cfg.operator_layers) | set(cfg.precursor_layers))
    trace_buffer: list[dict] = []
    rows: list[dict] = []

    texts = test_df["text"].tolist()
    clean_ids_all = test_df["clean_token_id"].to_numpy(np.int64)
    conflict_ids_all = test_df["conflict_token_id"].to_numpy(np.int64)
    prompt_ids_all = test_df["prompt_id"].astype(str).tolist()

    with torch.no_grad():
        for start in range(0, len(test_df), cfg.batch_size):
            end = min(len(test_df), start + cfg.batch_size)
            batch = test_df.iloc[start:end].reset_index(drop=True)
            inputs = tokenizer(
                texts[start:end],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_len,
            ).to(model.device)
            clean_ids = torch.tensor(clean_ids_all[start:end], dtype=torch.long, device=model.device)
            conflict_ids = torch.tensor(conflict_ids_all[start:end], dtype=torch.long, device=model.device)
            prompt_ids = prompt_ids_all[start:end]

            handles = []
            if control != "baseline":
                factory = executor.make_combo_hook_factory(
                    cfg=cfg,
                    stable_ops=stable_ops,
                    precursor_dirs=precursor_dirs,
                    control_mode=control,
                    W_device=weight,
                    action_lookup=action_lookup,
                    trace_buffer=trace_buffer,
                    clean_ids_batch=clean_ids,
                    conflict_ids_batch=conflict_ids,
                    prompt_ids_batch=prompt_ids,
                )
                for layer in hook_layers:
                    if layer < len(layer_modules):
                        handles.append(layer_modules[layer].register_forward_hook(factory(layer)))

            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            for handle in handles:
                handle.remove()

            position = executor.last_positions(inputs["attention_mask"])
            index = torch.arange(end - start, device=model.device)
            d_matrix = []
            for layer in cfg.decision_layers:
                state = outputs.hidden_states[layer + 1][index, position, :].detach().float()
                clean_score = torch.sum(state * weight[clean_ids].float(), dim=1)
                conflict_score = torch.sum(state * weight[conflict_ids].float(), dim=1)
                margin = (clean_score - conflict_score).cpu().numpy().astype(np.float32)
                d_matrix.append(
                    [
                        margin[row_index] - clean_anchors[int(row.graph_id)][int(layer)]
                        for row_index, row in batch.iterrows()
                    ]
                )
            d_matrix_np = np.asarray(d_matrix, dtype=np.float32).T
            delta_u = pca.transform(d_matrix_np)[:, 0]

            logits = outputs.logits[index, position, :].detach().float()
            top_ids = logits.argmax(dim=1)
            clean_logits = logits[index, clean_ids]
            conflict_logits = logits[index, conflict_ids]
            clean_ranks = (logits > clean_logits[:, None]).sum(dim=1) + 1
            conflict_ranks = (logits > conflict_logits[:, None]).sum(dim=1) + 1

            for row_index, row in batch.iterrows():
                top_id = int(top_ids[row_index].cpu())
                clean_id = int(clean_ids[row_index].cpu())
                conflict_id = int(conflict_ids[row_index].cpu())
                top_class = "clean" if top_id == clean_id else "conflict" if top_id == conflict_id else "other"
                rows.append(
                    {
                        "prompt_id": row["prompt_id"],
                        "graph_id": int(row["graph_id"]),
                        "condition": row["condition"],
                        "mechanism": row["mechanism"],
                        "control": control,
                        "clean_label": row["clean_label"],
                        "conflict_label": row["conflict_label"],
                        "clean_token_id": clean_id,
                        "conflict_token_id": conflict_id,
                        "task_coordinate": float(delta_u[row_index]),
                        "top1_token_id": top_id,
                        "top1_token": tokenizer.decode([top_id]),
                        "top1_class": top_class,
                        "clean_rank": int(clean_ranks[row_index].cpu()),
                        "conflict_rank": int(conflict_ranks[row_index].cpu()),
                        "clean_minus_conflict_margin": float((clean_logits[row_index] - conflict_logits[row_index]).cpu()),
                    }
                )
            del outputs, inputs, logits
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result = pd.DataFrame(rows)
    trace = pd.DataFrame(trace_buffer)
    result.to_csv(result_path, index=False)
    trace.to_csv(trace_path, index=False)
    return result, trace


def summarize_control(
    model_key: str,
    control: str,
    result: pd.DataFrame,
    baseline: pd.DataFrame,
    trace: pd.DataFrame,
    actions: pd.DataFrame | None,
) -> tuple[dict, pd.DataFrame]:
    base = baseline[
        ["prompt_id", "top1_class", "top1_token_id", "clean_minus_conflict_margin", "task_coordinate"]
    ].rename(
        columns={
            "top1_class": "baseline_top1_class",
            "top1_token_id": "baseline_top1_token_id",
            "clean_minus_conflict_margin": "baseline_margin",
            "task_coordinate": "baseline_task_coordinate",
        }
    )
    merged = result.merge(base, on="prompt_id", how="left")
    merged["margin_shift"] = merged["clean_minus_conflict_margin"] - merged["baseline_margin"]
    merged["task_coordinate_shift"] = merged["task_coordinate"] - merged["baseline_task_coordinate"]
    merged["top1_changed"] = ~merged["top1_token_id"].eq(merged["baseline_top1_token_id"])
    merged["baseline_eligible"] = merged["mechanism"].eq("closure") & merged["baseline_top1_class"].eq("conflict")
    merged["strict_conflict_to_clean"] = merged["baseline_eligible"] & merged["top1_class"].eq("clean")
    merged["pair_crossed"] = merged["baseline_margin"].lt(0) & merged["clean_minus_conflict_margin"].ge(0)
    merged["pair_crossed_third_token_block"] = (
        merged["baseline_eligible"] & merged["pair_crossed"] & ~merged["top1_class"].eq("clean")
    )
    eligible = merged[merged["baseline_eligible"]]
    non_target = merged[~merged["mechanism"].eq("closure")]
    crossings = int(eligible["strict_conflict_to_clean"].sum())
    eligible_n = int(len(eligible))
    target_rate = crossings / eligible_n if eligible_n else np.nan
    collateral_changes = int(non_target["top1_changed"].sum())
    non_target_n = int(len(non_target))
    collateral_rate = collateral_changes / non_target_n if non_target_n else np.nan
    ci_low, ci_high = wilson(crossings, eligible_n)

    if len(trace):
        weights = trace["batch_size"].to_numpy(float)
        mean_norm = float(np.average(trace["mean_norm_ratio"], weights=weights))
        max_norm = float(trace["max_norm_ratio"].max())
        all_finite = bool(trace["all_finite"].astype(str).str.lower().eq("true").all())
    else:
        mean_norm, max_norm, all_finite = 1.0, 1.0, True

    if actions is not None and len(actions):
        prompt_actions = actions.groupby("prompt_id", as_index=False).agg(
            alpha_max=("alpha", "max"), beta_max=("beta", "max")
        )
        open_rate = float((prompt_actions["alpha_max"].ne(0) | prompt_actions["beta_max"].ne(0)).mean())
    else:
        open_rate = 0.0

    summary = {
        "model": model_key,
        "control": control,
        "heldout_n": int(len(merged)),
        "eligible_n": eligible_n,
        "strict_crossings": crossings,
        "target_crossing_rate": target_rate,
        "target_crossing_ci95_low": ci_low,
        "target_crossing_ci95_high": ci_high,
        "eligible_median_margin_shift": float(eligible["margin_shift"].median()) if eligible_n else np.nan,
        "eligible_mean_margin_shift": float(eligible["margin_shift"].mean()) if eligible_n else np.nan,
        "eligible_median_task_coordinate_shift": float(eligible["task_coordinate_shift"].median()) if eligible_n else np.nan,
        "pair_crossed_third_token_block": int(eligible["pair_crossed_third_token_block"].sum()),
        "non_target_n": non_target_n,
        "non_target_top1_changes": collateral_changes,
        "collateral_rate": collateral_rate,
        "specificity": target_rate - collateral_rate if eligible_n and non_target_n else np.nan,
        "action_open_rate": open_rate,
        "mean_norm_ratio": mean_norm,
        "max_norm_ratio": max_norm,
        "all_finite": all_finite,
        "underpowered_eligible_lt5": eligible_n < 5,
    }
    merged["model"] = model_key
    return summary, merged


def permute_action_bundles(actions: pd.DataFrame, prompt_ids: list[str], rng: np.random.Generator) -> pd.DataFrame:
    source = actions.copy()
    source["prompt_id"] = source["prompt_id"].astype(str)
    available = set(source["prompt_id"])
    if available != set(prompt_ids):
        missing = sorted(set(prompt_ids) - available)[:5]
        extra = sorted(available - set(prompt_ids))[:5]
        raise RuntimeError(f"Action bundle mismatch; missing={missing}, extra={extra}")
    donors = rng.permutation(np.asarray(prompt_ids, dtype=object))
    blocks = []
    for recipient, donor in zip(prompt_ids, donors):
        block = source[source["prompt_id"].eq(str(donor))].copy()
        block["donor_prompt_id"] = str(donor)
        block["prompt_id"] = str(recipient)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def run_address_nulls(
    executor,
    model,
    tokenizer,
    test_df: pd.DataFrame,
    cfg,
    pca,
    clean_anchors,
    stable_ops,
    precursor_dirs,
    primary_actions: pd.DataFrame,
    baseline: pd.DataFrame,
    out_dir: Path,
    n_nulls: int,
    seed: int,
) -> pd.DataFrame:
    null_dir = out_dir / "address_nulls"
    null_dir.mkdir(parents=True, exist_ok=True)
    summary_path = null_dir / "address_null_summary.csv"
    if summary_path.exists():
        cached = pd.read_csv(summary_path)
        if len(cached) == n_nulls:
            print(f"[{cfg.model_key}] reuse {n_nulls} address nulls", flush=True)
            return cached

    prompt_ids = test_df["prompt_id"].astype(str).tolist()
    rng = np.random.default_rng(seed)
    rows = []
    primary_signature = primary_actions[["alpha", "beta"]].sort_values(
        ["alpha", "beta"], kind="stable"
    ).reset_index(drop=True)
    for null_id in range(n_nulls):
        actions = permute_action_bundles(primary_actions, prompt_ids, rng)
        control = f"shuffle_address_null_{null_id:03d}"
        null_signature = actions[["alpha", "beta"]].sort_values(
            ["alpha", "beta"], kind="stable"
        ).reset_index(drop=True)
        dose_multiset_exact = bool(primary_signature.equals(null_signature))
        if not dose_multiset_exact:
            raise RuntimeError(f"{cfg.model_key}: address null {null_id} changed the dose multiset")
        actions.to_csv(null_dir / f"actions_{control}.csv", index=False)
        result, trace = evaluate_full_vocab_and_coordinate(
            executor,
            model,
            tokenizer,
            test_df,
            cfg,
            pca,
            clean_anchors,
            stable_ops,
            precursor_dirs,
            control,
            actions,
            null_dir,
            cache=True,
        )
        summary, _ = summarize_control(cfg.model_key, control, result, baseline, trace, actions)
        summary["null_id"] = null_id
        summary["dose_multiset_exact"] = dose_multiset_exact
        summary["alpha_sum"] = float(actions["alpha"].sum())
        summary["beta_sum"] = float(actions["beta"].sum())
        summary["nonzero_action_rows"] = int(
            (actions["alpha"].ne(0) | actions["beta"].ne(0)).sum()
        )
        rows.append(summary)
        pd.DataFrame(rows).to_csv(summary_path, index=False)
    return pd.DataFrame(rows)


def prepare_base(executor, model, tokenizer, dataset: pd.DataFrame, cfg, model_dir: Path):
    base, hidden, _ = executor.extract_baseline(model, tokenizer, dataset, cfg, model_dir)
    train_index, test_index, split = executor.make_split(base, cfg, model_dir)
    split.to_csv(model_dir / "frozen_problem_split.csv", index=False)
    return base, hidden, train_index, test_index, split


def run_model(model_key: str, args: argparse.Namespace, modules: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    executor = modules["executor"]
    controller = modules["controller"]
    arithmetic = modules["arithmetic"]
    out_dir = args.output_root / model_key
    runtime_dir = out_dir / "runtime"
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    cfg = executor.Config()
    cfg.model_key = model_key
    cfg.model_path = "auto"
    cfg.save_dir = str(runtime_dir)
    cfg.n_graphs = args.problems
    cfg.max_len = 220
    cfg.batch_size = 4
    cfg.test_size = 0.30
    cfg.seed = args.seed
    cfg.n_shuffles = 0
    cfg.include_answer_control = False
    cfg.policy_actions = FULL_ACTIONS[:4] if args.smoke else FULL_ACTIONS
    for field, value in MODEL_WINDOWS[model_key].items():
        setattr(cfg, field, value)

    executor.CFG = cfg
    executor.make_combo_hook_factory = controller.norm_tracing_hook_factory(executor)
    executor.set_seed(cfg.seed)

    config = {
        "analysis": "frozen-protocol local-transition intervention on mixed arithmetic",
        "model": model_key,
        "protocol": str(HERE / "protocol_v0_1.md"),
        "protocol_sha256": sha256(HERE / "protocol_v0_1.md"),
        "executor_source_sha256": sha256(EXECUTOR_SOURCE),
        "controller_source_sha256": sha256(CONTROLLER_SOURCE),
        "arithmetic_source_sha256": sha256(ARITHMETIC_SOURCE),
        "config": asdict(cfg),
        "actions": list(cfg.policy_actions),
        "address_nulls": args.address_nulls,
        "interface": args.interface,
        "heldout_tuning": False,
        "smoke": args.smoke,
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"[{model_key}] load model", flush=True)
    model, tokenizer = executor.load_model_and_tokenizer(cfg)
    dataset, token_audit, prompt_wrapping = build_task_dataset(
        arithmetic, tokenizer, args.problems, args.seed, args.interface
    )
    dataset.to_csv(out_dir / "arithmetic_intervention_manifest.csv", index=False, encoding="utf-8-sig")
    token_audit.to_csv(out_dir / "contextual_tokenization_audit.csv", index=False)
    config["prompt_wrapping"] = prompt_wrapping
    config["candidate_token_rule"] = (
        "single leading-space continuation token after raw prompt"
        if args.interface == "raw"
        else "single no-leading-space next token at assistant response boundary"
    )
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    base, hidden, train_index, test_index, split = prepare_base(
        executor, model, tokenizer, dataset, cfg, runtime_dir
    )
    base, pca = refit_train_coordinate(controller, executor, base, train_index, cfg, out_dir)
    stable_ops = executor.fit_stable_operators(base, hidden, train_index, cfg, runtime_dir)
    precursor_dirs, probes, velocity = executor.fit_precursor_and_probes(base, hidden, train_index, cfg, runtime_dir)

    clean = base[base["condition"].eq("clean")].set_index("graph_id")
    clean_anchors = {
        int(problem): {int(layer): float(row[f"R_L{layer}"]) for layer in cfg.decision_layers}
        for problem, row in clean.iterrows()
    }
    train_df = base.iloc[train_index].reset_index(drop=True)
    test_df = base.iloc[test_index].reset_index(drop=True)
    train_features = executor.base_state_features_for_samples(base, hidden, probes, velocity, train_index, cfg)
    test_features = executor.base_state_features_for_samples(base, hidden, probes, velocity, test_index, cfg)

    grid = controller.run_grid(
        executor,
        model,
        tokenizer,
        train_df,
        cfg,
        pca,
        clean_anchors,
        stable_ops,
        precursor_dirs,
        cfg.policy_actions,
        out_dir,
    )
    merged_grid, labels = controller.derive_train_labels(grid, train_df, out_dir)
    global_action = controller.select_global_action(merged_grid, cfg.policy_actions, out_dir)

    mechanism_classifier, mechanism_audit = executor.fit_mechanism_classifier(train_features, cfg, runtime_dir)
    train_predicted = executor.attach_predicted_mechanism(train_features, mechanism_classifier)
    test_predicted = executor.attach_predicted_mechanism(test_features, mechanism_classifier)
    policy_classifier, policy_audit = executor.fit_gain_policy(
        train_predicted,
        labels,
        executor.POLICY_PRED_MECH_COLS,
        cfg,
        shuffle_policy_labels=False,
        shuffle_id=0,
    )
    policy_actions = executor.policy_predict_actions(policy_classifier, test_predicted, executor.POLICY_PRED_MECH_COLS)
    guarded_actions = confidence_gate(policy_actions, test_predicted)
    shuffled_policy_classifier, shuffled_policy_audit = executor.fit_gain_policy(
        train_predicted,
        labels,
        executor.POLICY_PRED_MECH_COLS,
        cfg,
        shuffle_policy_labels=True,
        shuffle_id=0,
    )
    shuffled_actions = executor.policy_predict_actions(
        shuffled_policy_classifier, test_predicted, executor.POLICY_PRED_MECH_COLS
    )
    fixed_actions = controller.uniform_actions(test_df, cfg.operator_layers, global_action, "training_global")
    controls = {
        "baseline": None,
        "fixed_global": fixed_actions,
        "ungated_policy": policy_actions,
        PRIMARY_CONTROL: guarded_actions,
        "label_shuffle_policy": shuffled_actions,
    }
    controls.update(component_actions(guarded_actions))
    for name, table in controls.items():
        if table is not None:
            table.to_csv(out_dir / f"actions_{name}.csv", index=False)
    (out_dir / "classifier_audits.json").write_text(
        json.dumps(
            {
                "mechanism": mechanism_audit,
                "policy": policy_audit,
                "shuffled_policy": shuffled_policy_audit,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    results: dict[str, pd.DataFrame] = {}
    traces: dict[str, pd.DataFrame] = {}
    for control in CONTROL_ORDER:
        result, trace = evaluate_full_vocab_and_coordinate(
            executor,
            model,
            tokenizer,
            test_df,
            cfg,
            pca,
            clean_anchors,
            stable_ops,
            precursor_dirs,
            control,
            controls[control],
            out_dir,
        )
        results[control] = result
        traces[control] = trace

    summary_rows = []
    prompt_rows = []
    baseline = results["baseline"]
    for control in CONTROL_ORDER:
        summary, prompts = summarize_control(
            model_key, control, results[control], baseline, traces[control], controls[control]
        )
        summary_rows.append(summary)
        prompt_rows.append(prompts)
    summary_df = pd.DataFrame(summary_rows)
    prompt_df = pd.concat(prompt_rows, ignore_index=True)
    summary_df.to_csv(out_dir / "control_summary.csv", index=False)
    prompt_df.to_csv(out_dir / "control_prompt_rows.csv", index=False)

    baseline_reproduction = base.iloc[test_index][["prompt_id", "R_final_margin", "pair_choice"]].merge(
        baseline[["prompt_id", "clean_minus_conflict_margin"]], on="prompt_id", how="inner"
    )
    baseline_reproduction["replay_pair_choice"] = np.where(
        baseline_reproduction["clean_minus_conflict_margin"].ge(0), "clean", "conflict"
    )
    extraction_sign_mismatch = ~baseline_reproduction["pair_choice"].eq(
        baseline_reproduction["replay_pair_choice"]
    )
    mismatch_outside_band = extraction_sign_mismatch & (
        baseline_reproduction["R_final_margin"].abs().gt(0.125)
        | baseline_reproduction["clean_minus_conflict_margin"].abs().gt(0.125)
    )

    repeat_dir = out_dir / "reproduction_repeat"
    repeat_dir.mkdir(parents=True, exist_ok=True)
    repeat_none, _ = evaluate_full_vocab_and_coordinate(
        executor,
        model,
        tokenizer,
        test_df,
        cfg,
        pca,
        clean_anchors,
        stable_ops,
        precursor_dirs,
        "baseline",
        None,
        repeat_dir,
        cache=False,
    )
    repeat_compare = baseline[
        ["prompt_id", "top1_token_id", "clean_minus_conflict_margin"]
    ].merge(
        repeat_none[["prompt_id", "top1_token_id", "clean_minus_conflict_margin"]],
        on="prompt_id",
        suffixes=("_first", "_repeat"),
        how="inner",
    )
    repeat_compare["pair_first"] = np.where(
        repeat_compare["clean_minus_conflict_margin_first"].ge(0), "clean", "conflict"
    )
    repeat_compare["pair_repeat"] = np.where(
        repeat_compare["clean_minus_conflict_margin_repeat"].ge(0), "clean", "conflict"
    )
    reproduction = {
        "n_test_rows": int(len(test_df)),
        "prompt_ids_exact": bool(set(test_df["prompt_id"].astype(str)) == set(baseline["prompt_id"].astype(str))),
        "baseline_margin_max_abs_delta": float(
            np.max(
                np.abs(
                    baseline_reproduction["R_final_margin"].to_numpy(float)
                    - baseline_reproduction["clean_minus_conflict_margin"].to_numpy(float)
                )
            )
        ),
        "extraction_pair_sign_disagreements": int(extraction_sign_mismatch.sum()),
        "extraction_pair_sign_disagreements_outside_tolerance_band": int(
            mismatch_outside_band.sum()
        ),
        "fp16_margin_tolerance": 0.125,
        "repeat_top1_exact": bool(
            repeat_compare["top1_token_id_first"].eq(
                repeat_compare["top1_token_id_repeat"]
            ).all()
        ),
        "repeat_pair_sign_exact": bool(
            repeat_compare["pair_first"].eq(repeat_compare["pair_repeat"]).all()
        ),
        "repeat_margin_max_abs_delta": float(
            np.max(
                np.abs(
                    repeat_compare["clean_minus_conflict_margin_first"].to_numpy(float)
                    - repeat_compare["clean_minus_conflict_margin_repeat"].to_numpy(float)
                )
            )
        ),
        "all_single_token_candidates": bool(token_audit["single_token"].astype(bool).all()),
        "all_context_prefixes_exact": bool(token_audit["prefix_exact"].astype(bool).all()),
        "all_first_appended_tokens_match": bool(
            token_audit["first_appended_matches"].astype(bool).all()
        ),
    }
    (out_dir / "baseline_reproduction.json").write_text(
        json.dumps(reproduction, indent=2), encoding="utf-8"
    )
    if (
        not reproduction["prompt_ids_exact"]
        or reproduction["baseline_margin_max_abs_delta"] > reproduction["fp16_margin_tolerance"]
        or reproduction["extraction_pair_sign_disagreements_outside_tolerance_band"] > 0
        or not reproduction["repeat_top1_exact"]
        or not reproduction["repeat_pair_sign_exact"]
        or reproduction["repeat_margin_max_abs_delta"] > reproduction["fp16_margin_tolerance"]
        or not reproduction["all_single_token_candidates"]
        or not reproduction["all_context_prefixes_exact"]
        or not reproduction["all_first_appended_tokens_match"]
    ):
        raise RuntimeError(f"{model_key}: baseline reproduction gate failed: {reproduction}")

    null_df = run_address_nulls(
        executor,
        model,
        tokenizer,
        test_df,
        cfg,
        pca,
        clean_anchors,
        stable_ops,
        precursor_dirs,
        guarded_actions,
        baseline,
        out_dir,
        args.address_nulls,
        args.seed + 10000 * (list(MODEL_WINDOWS).index(model_key) + 1),
    )

    del model, tokenizer, hidden, stable_ops, precursor_dirs, probes, velocity
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary_df, null_df


def pooled_decision(summary: pd.DataFrame, nulls: pd.DataFrame, n_nulls: int, smoke: bool) -> dict:
    primary = summary[summary["control"].eq(PRIMARY_CONTROL)].copy()
    eligible_total = int(primary["eligible_n"].sum())
    crossing_total = int(primary["strict_crossings"].sum())
    non_target_total = int(primary["non_target_n"].sum())
    collateral_total = int(primary["non_target_top1_changes"].sum())
    target_rate = crossing_total / eligible_total if eligible_total else np.nan
    collateral_rate = collateral_total / non_target_total if non_target_total else np.nan
    specificity = target_rate - collateral_rate

    pooled_null_rows = []
    for null_id, group in nulls.groupby("null_id"):
        eligible = int(group["eligible_n"].sum())
        crossings = int(group["strict_crossings"].sum())
        non_target = int(group["non_target_n"].sum())
        collateral = int(group["non_target_top1_changes"].sum())
        null_target_rate = crossings / eligible if eligible else np.nan
        null_collateral_rate = collateral / non_target if non_target else np.nan
        pooled_null_rows.append(
            {
                "null_id": int(null_id),
                "eligible_n": eligible,
                "strict_crossings": crossings,
                "target_crossing_rate": null_target_rate,
                "non_target_n": non_target,
                "non_target_top1_changes": collateral,
                "collateral_rate": null_collateral_rate,
                "specificity": null_target_rate - null_collateral_rate,
            }
        )
    pooled_null = pd.DataFrame(pooled_null_rows)
    q95 = float(pooled_null["specificity"].quantile(0.95)) if len(pooled_null) else np.nan
    empirical_p = (
        float((1 + pooled_null["specificity"].ge(specificity).sum()) / (1 + len(pooled_null)))
        if len(pooled_null)
        else np.nan
    )

    powered_positive = int(
        (
            primary["eligible_n"].ge(5)
            & primary["eligible_median_margin_shift"].gt(0)
            & primary["strict_crossings"].gt(0)
        ).sum()
    )
    selective_models = int(primary["collateral_rate"].le(0.05).sum())
    numerically_valid = bool(primary["all_finite"].astype(bool).all())
    accept = (
        powered_positive >= 2
        and selective_models >= 2
        and specificity > q95
        and crossing_total > 0
        and numerically_valid
    )
    any_positive_shift = bool(primary["eligible_median_margin_shift"].gt(0).any())
    any_pair_block = bool(primary["pair_crossed_third_token_block"].gt(0).any())
    if smoke:
        status = "smoke_only"
    elif accept:
        status = "confirmed_cross_task_transfer"
    elif any_positive_shift or any_pair_block:
        status = "partial_protocol_transfer"
    else:
        status = "no_transferable_evidence"
    decision = {
        "status": status,
        "primary_control": PRIMARY_CONTROL,
        "eligible_n_pooled": eligible_total,
        "strict_crossings_pooled": crossing_total,
        "target_crossing_rate_pooled": target_rate,
        "non_target_n_pooled": non_target_total,
        "non_target_top1_changes_pooled": collateral_total,
        "collateral_rate_pooled": collateral_rate,
        "specificity_pooled": specificity,
        "address_null_specificity_q95": q95,
        "address_null_empirical_p_one_sided": empirical_p,
        "powered_positive_models": powered_positive,
        "models_at_or_below_collateral_ceiling": selective_models,
        "numerically_valid": numerically_valid,
        "n_address_nulls": n_nulls,
        "claim_boundary": (
            "Protocol-level generalization after training-only refitting on a bounded mixed-arithmetic "
            "candidate task; not zero-shot parameter transfer, open-ended correctness control, deployment "
            "control or a universal dynamical law."
        ),
    }
    return decision, pooled_null


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.problems = min(args.problems, 12)
        args.address_nulls = min(args.address_nulls, 2)
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    modules = {
        "executor": load_module("mixed_arithmetic_executor", EXECUTOR_SOURCE),
        "controller": load_module("mixed_arithmetic_boundary_controller", CONTROLLER_SOURCE),
        "arithmetic": load_module("mixed_arithmetic_task", ARITHMETIC_SOURCE),
    }
    summaries = []
    nulls = []
    for model_key in args.models:
        summary, null = run_model(model_key, args, modules)
        summaries.append(summary)
        nulls.append(null)
    summary_df = pd.concat(summaries, ignore_index=True)
    null_df = pd.concat(nulls, ignore_index=True)
    summary_df.to_csv(args.output_root / "crossmodel_control_summary.csv", index=False)
    null_df.to_csv(args.output_root / "crossmodel_address_null_summary.csv", index=False)
    decision, pooled_null = pooled_decision(summary_df, null_df, args.address_nulls, args.smoke)
    pooled_null.to_csv(args.output_root / "pooled_address_null_summary.csv", index=False)
    (args.output_root / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
