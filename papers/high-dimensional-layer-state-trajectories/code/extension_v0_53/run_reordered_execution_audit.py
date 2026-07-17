"""Compare reordered trajectory geometry with native output behaviour."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = Path(os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen2.5-1.5B-Instruct"))
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase2-dir",
        type=Path,
        default=Path("inputs/execution_states"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/execution_order"),
    )
    parser.add_argument("--audit-prompts", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=2026071712)
    return parser.parse_args()


def continuation_id(tokenizer, text: str) -> int:
    ids = tokenizer(" " + str(text), add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise RuntimeError(f"Candidate {text!r} is not one continuation token: {ids}")
    return int(ids[0])


def logits_from_states(model, states: np.ndarray, batch_size: int = 8) -> np.ndarray:
    device = next(model.parameters()).device
    rows = []
    for start in range(0, len(states), batch_size):
        hidden = torch.from_numpy(
            np.array(states[start : start + batch_size, -1], copy=True)
        ).to(
            device=device, dtype=next(model.parameters()).dtype
        )
        with torch.no_grad():
            logits = model.get_output_embeddings()(hidden)
        rows.append(logits.float().cpu().numpy())
    return np.concatenate(rows, axis=0)


def direct_native_logits(model, tokenizer, prompts: list[str]) -> np.ndarray:
    device = next(model.parameters()).device
    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=300
    ).to(device)
    with torch.no_grad():
        output = model(**encoded, use_cache=False, return_dict=True)
    return output.logits[:, -1].float().cpu().numpy()


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), EPS)


def js_divergence(left_logits: np.ndarray, right_logits: np.ndarray) -> np.ndarray:
    left = softmax(left_logits)
    right = softmax(right_logits)
    middle = 0.5 * (left + right)
    left_term = np.where(left > 0, left * (np.log(left + EPS) - np.log(middle + EPS)), 0.0)
    right_term = np.where(
        right > 0, right * (np.log(right + EPS) - np.log(middle + EPS)), 0.0
    )
    return 0.5 * (left_term.sum(axis=1) + right_term.sum(axis=1))


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, repeats: int):
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_flip_p(values: np.ndarray, rng: np.random.Generator, repeats: int) -> float:
    values = np.asarray(values, dtype=float)
    observed = float(values.mean())
    exceed = 0
    chunk = 10000
    completed = 0
    while completed < repeats:
        count = min(chunk, repeats - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(count, len(values)))
        means = (signs * values).mean(axis=1)
        exceed += int(np.sum(means >= observed))
        completed += count
    return float((exceed + 1) / (repeats + 1))


def expected_candidate(row: pd.Series) -> str | None:
    if row["mechanism"] in {"stable", "stable_shift"}:
        return "clean"
    if row["mechanism"] == "closure":
        return "conflict"
    return None


def condition_behavior(
    name: str,
    logits: np.ndarray,
    native_logits: np.ndarray,
    metadata: pd.DataFrame,
    rng: np.random.Generator,
    bootstrap: int,
) -> tuple[pd.DataFrame, dict]:
    native_top1 = native_logits.argmax(axis=1)
    top1 = logits.argmax(axis=1)
    clean_ids = metadata["clean_id"].to_numpy(int)
    conflict_ids = metadata["conflict_id"].to_numpy(int)
    rows = np.arange(len(metadata))
    native_margin = native_logits[rows, clean_ids] - native_logits[rows, conflict_ids]
    margin = logits[rows, clean_ids] - logits[rows, conflict_ids]
    native_choice = np.where(native_margin >= 0, "clean", "conflict")
    choice = np.where(margin >= 0, "clean", "conflict")
    js = js_divergence(native_logits, logits)
    native_token_logit = logits[rows, native_top1]
    native_rank = 1 + np.sum(logits > native_token_logit[:, None], axis=1)
    expected = metadata["expected_candidate"].to_numpy(object)
    unambiguous = pd.notna(expected)
    correct = np.full(len(metadata), np.nan)
    native_correct = np.full(len(metadata), np.nan)
    correct[unambiguous] = (choice[unambiguous] == expected[unambiguous]).astype(float)
    native_correct[unambiguous] = (
        native_choice[unambiguous] == expected[unambiguous]
    ).astype(float)
    frame = metadata[
        ["prompt_id", "group_id", "condition", "mechanism", "clean_answer", "conflict_answer"]
    ].copy()
    frame["executed_order"] = name
    frame["top1_id"] = top1
    frame["native_top1_id"] = native_top1
    frame["top1_matches_native"] = top1 == native_top1
    frame["candidate_margin"] = margin
    frame["native_candidate_margin"] = native_margin
    frame["candidate_choice"] = choice
    frame["native_candidate_choice"] = native_choice
    frame["candidate_choice_matches_native"] = choice == native_choice
    frame["js_divergence_from_native"] = js
    frame["native_top1_rank"] = native_rank
    frame["expected_candidate"] = expected
    frame["task_correct"] = correct
    frame["native_task_correct"] = native_correct
    js_low, js_high = bootstrap_mean_ci(js, rng, bootstrap)
    summary = {
        "executed_order": name,
        "n_prompts": len(metadata),
        "full_vocab_top1_agreement": float(np.mean(top1 == native_top1)),
        "candidate_choice_agreement": float(np.mean(choice == native_choice)),
        "mean_js_divergence": float(js.mean()),
        "mean_js_divergence_ci_low": js_low,
        "mean_js_divergence_ci_high": js_high,
        "median_native_top1_rank": float(np.median(native_rank)),
        "mean_abs_candidate_margin_change": float(np.mean(np.abs(margin - native_margin))),
        "unambiguous_n": int(unambiguous.sum()),
        "unambiguous_task_accuracy": float(np.nanmean(correct)),
        "native_unambiguous_task_accuracy": float(np.nanmean(native_correct)),
        "unambiguous_accuracy_change": float(np.nanmean(correct - native_correct)),
    }
    disruption_tests = (
        summary["full_vocab_top1_agreement"] <= 0.50,
        summary["candidate_choice_agreement"] <= 0.75,
        summary["mean_js_divergence_ci_low"] > 0.05,
    )
    summary["functional_disruption_components"] = int(sum(disruption_tests))
    summary["passes_strong_functional_disruption"] = bool(sum(disruption_tests) >= 2)
    return frame, summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(
        args.phase2_dir / "phase2_prompt_subset.csv", encoding="utf-8-sig"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    metadata["clean_id"] = [continuation_id(tokenizer, value) for value in metadata["clean_answer"]]
    metadata["conflict_id"] = [
        continuation_id(tokenizer, value) for value in metadata["conflict_answer"]
    ]
    metadata["expected_candidate"] = metadata.apply(expected_candidate, axis=1)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    state_root = args.phase2_dir / "qwen"
    state_paths = {
        "native": state_root / "trained_checkpoint_alpha_1_native_hidden_float16.npy",
        "reverse": state_root / "trained_checkpoint_alpha_1_reverse_hidden_float16.npy",
        "fixed_permutation": state_root
        / "trained_checkpoint_alpha_1_fixed_permutation_hidden_float16.npy",
    }
    states = {name: np.asarray(np.load(path, mmap_mode="r")) for name, path in state_paths.items()}
    reconstructed = {name: logits_from_states(model, values) for name, values in states.items()}

    audit_n = min(args.audit_prompts, len(metadata))
    direct = direct_native_logits(model, tokenizer, metadata["prompt"].iloc[:audit_n].tolist())
    cached = reconstructed["native"][:audit_n]
    audit_rows = np.arange(audit_n)
    audit_clean = metadata["clean_id"].iloc[:audit_n].to_numpy(int)
    audit_conflict = metadata["conflict_id"].iloc[:audit_n].to_numpy(int)
    direct_margin = direct[audit_rows, audit_clean] - direct[audit_rows, audit_conflict]
    cached_margin = cached[audit_rows, audit_clean] - cached[audit_rows, audit_conflict]
    reconstruction = {
        "audit_prompts": audit_n,
        "full_vocab_top1_agreement": float(np.mean(direct.argmax(1) == cached.argmax(1))),
        "candidate_margin_max_abs_error": float(np.max(np.abs(direct_margin - cached_margin))),
    }
    reconstruction["passes"] = bool(
        reconstruction["full_vocab_top1_agreement"] == 1.0
        and reconstruction["candidate_margin_max_abs_error"] < 0.05
    )
    if not reconstruction["passes"]:
        raise RuntimeError(f"Cached-state reconstruction failed: {reconstruction}")

    rng = np.random.default_rng(args.seed)
    prompt_frames = []
    summaries = []
    for index, name in enumerate(("native", "reverse", "fixed_permutation")):
        frame, summary = condition_behavior(
            name,
            reconstructed[name],
            reconstructed["native"],
            metadata,
            np.random.default_rng(args.seed + index * 1000),
            args.bootstrap,
        )
        prompt_frames.append(frame)
        summaries.append(summary)

    geometry = pd.read_csv(
        args.phase2_dir / "phase2_prompt_metrics.csv", encoding="utf-8-sig"
    )
    geometry = geometry[
        geometry["model"].eq("qwen")
        & geometry["state_type"].eq("trained_checkpoint")
        & geometry["residual_alpha"].eq(1.0)
        & geometry["representation"].eq("unit_hidden_state")
    ]
    native_geometry = geometry[geometry["executed_order"].eq("native")].set_index("prompt_id")
    contrasts = []
    for order in ("reverse", "fixed_permutation"):
        altered = geometry[geometry["executed_order"].eq(order)].set_index("prompt_id")
        common = native_geometry.index.intersection(altered.index)
        difference = (
            altered.loc[common, "future_chord_alignment"].to_numpy(float)
            - native_geometry.loc[common, "future_chord_alignment"].to_numpy(float)
        )
        low, high = bootstrap_mean_ci(difference, rng, args.bootstrap)
        p = sign_flip_p(difference, rng, args.permutations)
        contrasts.append(
            {
                "executed_order": order,
                "future_chord_difference_mean": float(difference.mean()),
                "future_chord_difference_ci_low": low,
                "future_chord_difference_ci_high": high,
                "future_chord_sign_flip_p": p,
                "passes_positive_geometry_contrast": bool(low > 0 and p <= 0.05),
            }
        )

    prompt_frame = pd.concat(prompt_frames, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    contrast_frame = pd.DataFrame(contrasts)
    prompt_frame.to_csv(
        args.output_dir / "layer_order_behavior_prompt_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_frame.to_csv(
        args.output_dir / "layer_order_behavior_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    contrast_frame.to_csv(
        args.output_dir / "layer_order_geometry_contrasts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reverse_geometry = bool(
        contrast_frame.set_index("executed_order").loc[
            "reverse", "passes_positive_geometry_contrast"
        ]
    )
    reverse_function = bool(
        summary_frame.set_index("executed_order").loc[
            "reverse", "passes_strong_functional_disruption"
        ]
    )
    fixed_geometry = bool(
        contrast_frame.set_index("executed_order").loc[
            "fixed_permutation", "passes_positive_geometry_contrast"
        ]
    )
    fixed_function = bool(
        summary_frame.set_index("executed_order").loc[
            "fixed_permutation", "passes_strong_functional_disruption"
        ]
    )
    gate = {
        "window": "layer_order_geometry_versus_behavior",
        "reconstruction_audit": reconstruction,
        "reverse_positive_geometry": reverse_geometry,
        "reverse_functional_disruption": reverse_function,
        "reverse_dissociation_pass": bool(reverse_geometry and reverse_function),
        "fixed_permutation_positive_geometry": fixed_geometry,
        "fixed_permutation_functional_disruption": fixed_function,
        "fixed_permutation_replication_pass": bool(fixed_geometry and fixed_function),
        "claim_boundary": (
            "greater future-chord alignment is not sufficient for native output fidelity; "
            "no claim that every reordering lowers task correctness"
        ),
    }
    (args.output_dir / "layer_order_behavior_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    config = {
        "script": str(Path(__file__).resolve()),
        "model_path": str(MODEL_PATH),
        "phase2_dir": str(args.phase2_dir.resolve()),
        "bootstrap": args.bootstrap,
        "sign_flip_permutations": args.permutations,
        "seed": args.seed,
        "protocol": str(
            (Path.cwd() / "outputs/execution_order_protocol_v0_1.md").resolve()
        ),
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
