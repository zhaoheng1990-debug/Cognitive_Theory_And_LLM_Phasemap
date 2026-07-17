"""Separate block kinematics from trained embedding/output semantic coordinates."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from analyze_order_geometry_controls import (
    bootstrap_ci,
    layer_conditioned_donor_steps,
    prompt_metrics_from_steps,
    steps_from_states,
)


MODEL_PATH = Path(
    os.environ.get("QWEN_MODEL_PATH", "models/qwen")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lexical-manifest",
        type=Path,
        default=Path("inputs/lexical_task/lexical_task_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("architecture_semantic_geometry_v0_1/phase3_component_decomposition"),
    )
    parser.add_argument("--max-problems", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--random-seed", type=int, default=1701)
    parser.add_argument("--analysis-seed", type=int, default=2026071703)
    parser.add_argument("--semantic-label-nulls", type=int, default=5000)
    parser.add_argument("--donor-nulls", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=3000)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_models(seed: int):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    trained = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    set_seed(seed)
    config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    config.use_cache = False
    random_model = AutoModelForCausalLM.from_config(
        config, trust_remote_code=True, dtype=dtype
    )
    if torch.cuda.is_available():
        trained = trained.to("cuda")
        random_model = random_model.to("cuda")
    trained.eval()
    random_model.eval()
    trained.config.use_cache = False
    random_model.config.use_cache = False
    return trained, random_model


def continuation_id(tokenizer, label: str) -> int:
    ids = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise RuntimeError(f"Label {label!r} is not a single continuation token: {ids}")
    return int(ids[0])


def prepare_lexical_subset(manifest_path: Path, max_problems: int) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    subset = manifest[manifest["condition"].eq("clean")].copy()
    subset = subset.sort_values("problem_id").iloc[:max_problems].reset_index(drop=True)
    subset["category"] = subset["expression"].str.replace("category=", "", regex=False)
    if subset["problem_id"].nunique() != len(subset):
        raise ValueError("Expected one clean prompt per lexical problem")
    return subset


def apply_chat_template(tokenizer, prompts: list[str]) -> list[str]:
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompts
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        for prompt in prompts
    ]


def extract_states_and_candidates(
    model,
    tokenizer,
    prompts: list[str],
    clean_ids: np.ndarray,
    conflict_ids: np.ndarray,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_layers = int(model.config.num_hidden_layers)
    hidden_size = int(model.config.hidden_size)
    states = np.zeros((len(prompts), n_layers, hidden_size), dtype=np.float16)
    margins = np.zeros(len(prompts), dtype=np.float32)
    pair_correct = np.zeros(len(prompts), dtype=bool)
    device = next(model.parameters()).device
    rendered = apply_chat_template(tokenizer, prompts)
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        encoded = tokenizer(
            rendered[start:end],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            output = model.model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        rows = torch.arange(end - start, device=device)
        if tokenizer.padding_side == "left":
            position = torch.full(
                (end - start,), encoded["attention_mask"].shape[1] - 1, device=device, dtype=torch.long
            )
        else:
            position = encoded["attention_mask"].sum(dim=1) - 1
        final_state = output.last_hidden_state[rows, position, :]
        clean = torch.as_tensor(clean_ids[start:end], device=device, dtype=torch.long)
        conflict = torch.as_tensor(conflict_ids[start:end], device=device, dtype=torch.long)
        output_weight = model.get_output_embeddings().weight
        clean_logit = torch.sum(final_state * output_weight[clean], dim=-1)
        conflict_logit = torch.sum(final_state * output_weight[conflict], dim=-1)
        batch_margin = clean_logit - conflict_logit
        margins[start:end] = batch_margin.detach().float().cpu().numpy()
        pair_correct[start:end] = (batch_margin > 0).detach().cpu().numpy()
        for layer_index in range(n_layers):
            values = output.hidden_states[layer_index + 1][rows, position, :]
            states[start:end, layer_index, :] = (
                values.detach().float().cpu().numpy().astype(np.float16)
            )
        print(f"extracted {end}/{len(prompts)}", flush=True)
        del output, encoded, final_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not np.isfinite(states).all() or not np.isfinite(margins).all():
        raise RuntimeError("Non-finite component-decomposition output")
    return states, margins, pair_correct


def unique_semantic_tokens(
    lexical: pd.DataFrame, tokenizer
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    words = []
    for _, row in lexical.iterrows():
        words.append((str(row["clean_label"]), str(row["category"])))
    unique_pairs = sorted(set(words))
    counts = Counter(word for word, _ in unique_pairs)
    # A word assigned to multiple categories cannot serve as an unambiguous label.
    unambiguous = [(word, category) for word, category in unique_pairs if counts[word] == 1]
    token_rows = []
    for word, category in unambiguous:
        token_rows.append(
            {"word": word, "category": category, "token_id": continuation_id(tokenizer, word)}
        )
    frame = pd.DataFrame(token_rows).drop_duplicates("token_id").reset_index(drop=True)
    category_codes = pd.Categorical(frame["category"]).codes.astype(int)
    return frame["token_id"].to_numpy(dtype=int), category_codes, frame


def semantic_enrichment(
    rows: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    n_nulls: int,
) -> dict[str, float]:
    values = np.asarray(rows, dtype=np.float32)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
    cosine = values @ values.T
    upper = np.triu_indices(len(values), k=1)
    pair_cosine = cosine[upper]
    same = labels[upper[0]] == labels[upper[1]]
    if not np.any(same) or np.all(same):
        return {
            "semantic_enrichment": np.nan,
            "semantic_enrichment_null_mean": np.nan,
            "semantic_enrichment_null_q95": np.nan,
            "semantic_enrichment_null_p": np.nan,
        }
    observed = float(pair_cosine[same].mean() - pair_cosine[~same].mean())
    null = np.empty(n_nulls, dtype=float)
    for index in range(n_nulls):
        shuffled = rng.permutation(labels)
        null_same = shuffled[upper[0]] == shuffled[upper[1]]
        null[index] = float(
            pair_cosine[null_same].mean() - pair_cosine[~null_same].mean()
        )
    return {
        "semantic_enrichment": observed,
        "semantic_enrichment_null_mean": float(null.mean()),
        "semantic_enrichment_null_q95": float(np.quantile(null, 0.95)),
        "semantic_enrichment_null_p": float((1 + np.sum(null >= observed)) / (n_nulls + 1)),
    }


def trajectory_summary(
    states: np.ndarray,
    rng: np.random.Generator,
    donor_nulls: int,
    bootstrap: int,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for representation in ("unit_hidden_state", "raw_residual_state"):
        steps = steps_from_states(states, representation)
        real = prompt_metrics_from_steps(steps)
        donor = {name: [] for name in real}
        for _ in range(donor_nulls):
            null_metrics = prompt_metrics_from_steps(layer_conditioned_donor_steps(steps, rng))
            for name, values in null_metrics.items():
                donor[name].append(float(np.nanmean(values)))
        for name, values in real.items():
            prefix = f"{representation}_{name}"
            ci_low, ci_high = bootstrap_ci(values, rng, bootstrap)
            result[f"{prefix}_mean"] = float(np.nanmean(values))
            result[f"{prefix}_ci_low"] = ci_low
            result[f"{prefix}_ci_high"] = ci_high
            result[f"{prefix}_donor_null_mean"] = float(np.mean(donor[name]))
            result[f"{prefix}_donor_null_q95"] = float(np.quantile(donor[name], 0.95))
    return result


def copy_trained_coordinates(trained, target) -> None:
    with torch.no_grad():
        target.get_input_embeddings().weight.copy_(trained.get_input_embeddings().weight)
        if target.get_output_embeddings().weight.data_ptr() != target.get_input_embeddings().weight.data_ptr():
            target.get_output_embeddings().weight.copy_(trained.get_output_embeddings().weight)
        target.model.norm.load_state_dict(trained.model.norm.state_dict())
    target.tie_weights()


def randomize_coordinates_in_place(model, seed: int) -> None:
    set_seed(seed)
    with torch.no_grad():
        model.get_input_embeddings().reset_parameters()
        if model.get_output_embeddings().weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
            model.get_output_embeddings().reset_parameters()
        for parameter in model.model.norm.parameters():
            parameter.fill_(1.0)
    model.tie_weights()


def posthoc_readout_rows(
    states: np.ndarray,
    clean_rows: np.ndarray,
    conflict_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    final_state = np.asarray(states[:, -1, :], dtype=np.float32)
    clean = np.sum(final_state * clean_rows, axis=1)
    conflict = np.sum(final_state * conflict_rows, axis=1)
    margin = clean - conflict
    return margin, margin > 0


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lexical = prepare_lexical_subset(args.lexical_manifest, args.max_problems)
    lexical.to_csv(args.output_dir / "phase3_lexical_subset.csv", index=False, encoding="utf-8-sig")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    clean_ids = np.asarray(
        [continuation_id(tokenizer, label) for label in lexical["clean_label"]], dtype=int
    )
    conflict_ids = np.asarray(
        [continuation_id(tokenizer, label) for label in lexical["conflict_label"]], dtype=int
    )
    semantic_ids, semantic_labels, token_frame = unique_semantic_tokens(lexical, tokenizer)
    token_frame.to_csv(
        args.output_dir / "phase3_semantic_token_audit.csv", index=False, encoding="utf-8-sig"
    )

    trained, random_model = load_models(args.random_seed)
    conditions: list[tuple[str, object]] = [
        ("trained_blocks_trained_coordinates", trained),
        ("random_blocks_random_coordinates", random_model),
    ]
    results: list[dict] = []
    saved_states: dict[str, np.ndarray] = {}

    for condition_index, (condition, model) in enumerate(conditions):
        states, margins, pair_correct = extract_states_and_candidates(
            model,
            tokenizer,
            lexical["prompt"].tolist(),
            clean_ids,
            conflict_ids,
            args.batch_size,
            args.max_length,
        )
        saved_states[condition] = states
        np.save(args.output_dir / f"{condition}_hidden_float16.npy", states)
        np.save(args.output_dir / f"{condition}_candidate_margins.npy", margins)
        rng = np.random.default_rng(args.analysis_seed + condition_index * 10_000)
        row = {
            "condition": condition,
            "candidate_pair_accuracy": float(np.mean(pair_correct)),
            "candidate_margin_mean": float(np.mean(margins)),
            "candidate_margin_median": float(np.median(margins)),
        }
        row.update(trajectory_summary(states, rng, args.donor_nulls, args.bootstrap))
        semantic_index = torch.as_tensor(
            semantic_ids.copy(), device=model.get_output_embeddings().weight.device, dtype=torch.long
        )
        weight = (
            model.get_output_embeddings().weight[semantic_index].detach().float().cpu().numpy()
        )
        row.update(
            {f"output_{key}": value for key, value in semantic_enrichment(
                weight, semantic_labels, rng, args.semantic_label_nulls
            ).items()}
        )
        embedding = (
            model.get_input_embeddings().weight[semantic_index].detach().float().cpu().numpy()
        )
        row.update(
            {f"embedding_{key}": value for key, value in semantic_enrichment(
                embedding, semantic_labels, rng, args.semantic_label_nulls
            ).items()}
        )
        results.append(row)
        del weight, embedding

    rng = np.random.default_rng(args.analysis_seed + 50_000)
    unique_needed = np.unique(np.r_[clean_ids, conflict_ids, semantic_ids])
    needed_position = {int(token_id): index for index, token_id in enumerate(unique_needed)}
    output_tensor = trained.get_output_embeddings().weight
    permutation = rng.permutation(output_tensor.shape[0])
    permuted_source_ids = torch.as_tensor(
        permutation[unique_needed], device=output_tensor.device, dtype=torch.long
    )
    permuted_needed_rows = (
        output_tensor[permuted_source_ids].detach().float().cpu().numpy()
    )
    # Estimate the trained scale without materializing the full vocabulary matrix in float32.
    scale_sample_ids = torch.as_tensor(
        rng.choice(output_tensor.shape[0], size=min(4096, output_tensor.shape[0]), replace=False),
        device=output_tensor.device,
        dtype=torch.long,
    )
    gaussian_scale = float(
        output_tensor[scale_sample_ids].detach().float().std().cpu().item()
    )
    gaussian_needed_rows = rng.normal(
        0.0,
        gaussian_scale,
        size=(len(unique_needed), output_tensor.shape[1]),
    ).astype(np.float32)

    for post_index, (condition, needed_rows) in enumerate(
        (
            ("trained_blocks_row_permuted_output", permuted_needed_rows),
            ("trained_blocks_gaussian_output", gaussian_needed_rows),
        )
    ):
        states = saved_states["trained_blocks_trained_coordinates"]
        clean_rows = needed_rows[[needed_position[int(token_id)] for token_id in clean_ids]]
        conflict_rows = needed_rows[
            [needed_position[int(token_id)] for token_id in conflict_ids]
        ]
        margins, pair_correct = posthoc_readout_rows(states, clean_rows, conflict_rows)
        local_rng = np.random.default_rng(args.analysis_seed + 60_000 + post_index)
        row = {
            "condition": condition,
            "candidate_pair_accuracy": float(np.mean(pair_correct)),
            "candidate_margin_mean": float(np.mean(margins)),
            "candidate_margin_median": float(np.median(margins)),
            "trajectory_reused_from": "trained_blocks_trained_coordinates",
        }
        row.update(
            {f"output_{key}": value for key, value in semantic_enrichment(
                needed_rows[
                    [needed_position[int(token_id)] for token_id in semantic_ids]
                ],
                semantic_labels,
                local_rng,
                args.semantic_label_nulls,
            ).items()}
        )
        results.append(row)

    copy_trained_coordinates(trained, random_model)
    states, margins, pair_correct = extract_states_and_candidates(
        random_model,
        tokenizer,
        lexical["prompt"].tolist(),
        clean_ids,
        conflict_ids,
        args.batch_size,
        args.max_length,
    )
    condition = "random_blocks_trained_coordinates"
    np.save(args.output_dir / f"{condition}_hidden_float16.npy", states)
    rng = np.random.default_rng(args.analysis_seed + 70_000)
    row = {
        "condition": condition,
        "candidate_pair_accuracy": float(np.mean(pair_correct)),
        "candidate_margin_mean": float(np.mean(margins)),
        "candidate_margin_median": float(np.median(margins)),
    }
    row.update(trajectory_summary(states, rng, args.donor_nulls, args.bootstrap))
    semantic_index = torch.as_tensor(
        semantic_ids.copy(), device=random_model.get_output_embeddings().weight.device, dtype=torch.long
    )
    weight = (
        random_model.get_output_embeddings().weight[semantic_index]
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    row.update(
        {f"output_{key}": value for key, value in semantic_enrichment(
            weight, semantic_labels, rng, args.semantic_label_nulls
        ).items()}
    )
    results.append(row)
    del random_model, weight, states
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    randomize_coordinates_in_place(trained, args.random_seed + 1)
    states, margins, pair_correct = extract_states_and_candidates(
        trained,
        tokenizer,
        lexical["prompt"].tolist(),
        clean_ids,
        conflict_ids,
        args.batch_size,
        args.max_length,
    )
    condition = "trained_blocks_random_coordinates"
    np.save(args.output_dir / f"{condition}_hidden_float16.npy", states)
    rng = np.random.default_rng(args.analysis_seed + 80_000)
    row = {
        "condition": condition,
        "candidate_pair_accuracy": float(np.mean(pair_correct)),
        "candidate_margin_mean": float(np.mean(margins)),
        "candidate_margin_median": float(np.median(margins)),
    }
    row.update(trajectory_summary(states, rng, args.donor_nulls, args.bootstrap))
    semantic_index = torch.as_tensor(
        semantic_ids.copy(), device=trained.get_output_embeddings().weight.device, dtype=torch.long
    )
    weight = (
        trained.get_output_embeddings().weight[semantic_index].detach().float().cpu().numpy()
    )
    row.update(
        {f"output_{key}": value for key, value in semantic_enrichment(
            weight, semantic_labels, rng, args.semantic_label_nulls
        ).items()}
    )
    results.append(row)

    result_frame = pd.DataFrame(results)
    result_frame.to_csv(
        args.output_dir / "phase3_component_summary.csv", index=False, encoding="utf-8-sig"
    )
    gate = {
        "phase": 3,
        "tested_architecture": "Qwen2.5-1.5B-Instruct",
        "trained_output_semantic_enrichment_passes": bool(
            result_frame.loc[
                result_frame["condition"].eq("trained_blocks_trained_coordinates"),
                "output_semantic_enrichment_null_p",
            ].iloc[0]
            <= 0.05
        ),
        "trained_candidate_pair_accuracy": float(
            result_frame.loc[
                result_frame["condition"].eq("trained_blocks_trained_coordinates"),
                "candidate_pair_accuracy",
            ].iloc[0]
        ),
        "condition_scope": (
            "within-architecture component stress test; tied input/output weights prevent "
            "independent causal separation of E and W"
        ),
    }
    (args.output_dir / "phase3_gate_decision.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    config = {
        "model": str(MODEL_PATH),
        "task": "64-problem clean lexical-category subset",
        "conditions": result_frame["condition"].tolist(),
        "random_initialization_seed": args.random_seed,
        "semantic_label_nulls": args.semantic_label_nulls,
        "donor_nulls": args.donor_nulls,
        "bootstrap": args.bootstrap,
        "tied_input_output_weights": bool(trained.config.tie_word_embeddings),
        "claim_boundary": (
            "component-level dissociation in one tied-weight Qwen checkpoint; no claim that W is a manifold"
        ),
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
