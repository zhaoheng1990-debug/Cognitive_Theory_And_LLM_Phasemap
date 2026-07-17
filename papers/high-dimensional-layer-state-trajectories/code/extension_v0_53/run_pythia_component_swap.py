"""Run a frozen 2^4 training-checkpoint component swap in Pythia-160m."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from extract_pythia_training_trajectory import prepare_subset


COMPONENTS = ("E", "B", "N", "W")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="EleutherAI/pythia-160m")
    parser.add_argument("--early-revision", default="step0")
    parser.add_argument("--trained-revision", default="step143000")
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path(os.environ.get("HF_TRAINING_CACHE", "models")),
    )
    parser.add_argument(
        "--lexical-manifest", type=Path,
        default=Path("inputs/lexical_task/lexical_task_manifest.csv"),
    )
    parser.add_argument(
        "--prior-summary", type=Path,
        default=Path(
            "inputs/pythia_training_states/"
            "phase4_training_time_summary.csv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/pythia_component_compatibility"),
    )
    parser.add_argument("--max-problems", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026071711)
    parser.add_argument(
        "--conditions", nargs="*",
        help="Optional condition codes such as 0000 1111. Default: all 16.",
    )
    return parser.parse_args()


def component_modules(model) -> dict[str, torch.nn.Module]:
    return {
        "E": model.get_input_embeddings(),
        "B": model.gpt_neox.layers,
        "N": model.gpt_neox.final_layer_norm,
        "W": model.get_output_embeddings(),
    }


def module_fingerprint(module: torch.nn.Module) -> dict[str, object]:
    digest = hashlib.sha256()
    n_values = 0
    sum_sq = 0.0
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
        n_values += int(array.size)
        sum_sq += float(np.sum(array.astype(np.float64) ** 2))
    return {
        "sha256": digest.hexdigest(),
        "n_values": n_values,
        "l2_norm": float(np.sqrt(sum_sq)),
    }


def install_condition(
    working: torch.nn.Module,
    early: torch.nn.Module,
    trained: torch.nn.Module,
    bits: tuple[int, int, int, int],
) -> None:
    target = component_modules(working)
    early_modules = component_modules(early)
    trained_modules = component_modules(trained)
    for component, bit in zip(COMPONENTS, bits):
        source = trained_modules[component] if bit else early_modules[component]
        target[component].load_state_dict(source.state_dict(), strict=True)


def evaluate(
    model,
    tokenizer,
    subset: pd.DataFrame,
    batch_size: int,
    max_length: int,
) -> list[dict[str, object]]:
    device = next(model.parameters()).device
    prompts = subset["analysis_prompt"].astype(str).tolist()
    clean_ids = subset["clean_token_id"].to_numpy(dtype=int)
    conflict_ids = subset["conflict_token_id"].to_numpy(dtype=int)
    records: list[dict[str, object]] = []
    for start in range(0, len(subset), batch_size):
        end = min(start + batch_size, len(subset))
        encoded = tokenizer(
            prompts[start:end], return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(device)
        with torch.inference_mode():
            logits = model(**encoded, use_cache=False, return_dict=True).logits
        if tokenizer.padding_side == "left":
            positions = torch.full(
                (end - start,), encoded["attention_mask"].shape[1] - 1,
                device=device, dtype=torch.long,
            )
        else:
            positions = encoded["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(end - start, device=device)
        final_logits = logits[rows, positions]
        clean = torch.as_tensor(clean_ids[start:end].copy(), device=device, dtype=torch.long)
        conflict = torch.as_tensor(
            conflict_ids[start:end].copy(), device=device, dtype=torch.long
        )
        margin = final_logits[rows, clean] - final_logits[rows, conflict]
        top1 = torch.argmax(final_logits, dim=-1)
        for offset, row_index in enumerate(range(start, end)):
            row = subset.iloc[row_index]
            records.append({
                "prompt_index": row_index,
                "problem_id": str(row["problem_id"]),
                "category": str(row["category"]),
                "clean_label": str(row["clean_label"]),
                "conflict_label": str(row["conflict_label"]),
                "clean_token_id": int(clean_ids[row_index]),
                "conflict_token_id": int(conflict_ids[row_index]),
                "candidate_margin": float(margin[offset].float().cpu()),
                "pair_correct": bool(margin[offset] > 0),
                "top1_token_id": int(top1[offset].cpu()),
                "top1_token": tokenizer.decode([int(top1[offset].cpu())]),
            })
        del encoded, logits, final_logits
    return records


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    n = len(values)
    means = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, 1000):
        end = min(start + 1000, n_boot)
        indices = rng.integers(0, n, size=(end - start, n))
        means[start:end] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def condition_code(bits: tuple[int, int, int, int]) -> str:
    return "".join(str(bit) for bit in bits)


def analyze_factorial(
    outcomes: pd.DataFrame, n_boot: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    expected_codes = [condition_code(bits) for bits in itertools.product((0, 1), repeat=4)]
    present_codes = sorted(outcomes["condition"].unique())
    if present_codes != expected_codes:
        raise ValueError("Factorial analysis requires all 16 conditions")
    rng = np.random.default_rng(seed)
    summary_rows = []
    for code, frame in outcomes.groupby("condition", sort=True):
        accuracy_values = frame.sort_values("prompt_index")["pair_correct"].to_numpy(float)
        margin_values = frame.sort_values("prompt_index")["candidate_margin"].to_numpy(float)
        acc_ci = bootstrap_ci(accuracy_values, rng, n_boot)
        margin_ci = bootstrap_ci(margin_values, rng, n_boot)
        bits = tuple(int(x) for x in code)
        summary_rows.append({
            "condition": code,
            **{component: bit for component, bit in zip(COMPONENTS, bits)},
            "n_prompts": len(frame),
            "candidate_pair_accuracy": float(accuracy_values.mean()),
            "candidate_pair_accuracy_ci_low": acc_ci[0],
            "candidate_pair_accuracy_ci_high": acc_ci[1],
            "candidate_margin_mean": float(margin_values.mean()),
            "candidate_margin_ci_low": margin_ci[0],
            "candidate_margin_ci_high": margin_ci[1],
        })
    summary = pd.DataFrame(summary_rows)

    indexed = outcomes.set_index(["prompt_index", "condition"]).sort_index()
    prompt_ids = sorted(outcomes["prompt_index"].unique())
    difference_rows = []
    reference_codes = ["0000", "1111"]
    for metric in ("pair_correct", "candidate_margin"):
        for code in expected_codes:
            for reference in reference_codes:
                if code == reference:
                    continue
                values = np.array([
                    float(indexed.loc[(prompt, code), metric])
                    - float(indexed.loc[(prompt, reference), metric])
                    for prompt in prompt_ids
                ])
                ci = bootstrap_ci(values, rng, n_boot)
                difference_rows.append({
                    "metric": metric,
                    "condition": code,
                    "reference": reference,
                    "mean_difference": float(values.mean()),
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                })
    differences = pd.DataFrame(difference_rows)

    design_bits = np.array([[int(x) for x in code] for code in expected_codes], dtype=int)
    design_sign = design_bits * 2 - 1
    effect_rows = []
    for metric in ("pair_correct", "candidate_margin"):
        matrix = np.array([
            [float(indexed.loc[(prompt, code), metric]) for code in expected_codes]
            for prompt in prompt_ids
        ])
        for order in range(1, len(COMPONENTS) + 1):
            for component_indices in itertools.combinations(range(len(COMPONENTS)), order):
                sign = np.prod(design_sign[:, component_indices], axis=1)
                prompt_beta = np.mean(matrix * sign[None, :], axis=1)
                ci = bootstrap_ci(2.0 * prompt_beta, rng, n_boot)
                effect_rows.append({
                    "metric": metric,
                    "effect": ":".join(COMPONENTS[i] for i in component_indices),
                    "order": order,
                    "orthogonal_beta": float(prompt_beta.mean()),
                    "factorial_contrast_2beta": float(2.0 * prompt_beta.mean()),
                    "contrast_ci_low": ci[0],
                    "contrast_ci_high": ci[1],
                })
    effects = pd.DataFrame(effect_rows)

    accuracy = summary.set_index("condition")["candidate_pair_accuracy"].to_dict()
    margins = {
        code: np.array([
            float(indexed.loc[(prompt, code), "candidate_margin"]) for prompt in prompt_ids
        ])
        for code in expected_codes
    }
    single_codes = ("1000", "0100", "0010", "0001")
    additive_prompt = sum((margins[code] - margins["0000"] for code in single_codes), start=np.zeros(len(prompt_ids)))
    superadditive_prompt = (margins["1111"] - margins["0000"]) - additive_prompt
    super_ci = bootstrap_ci(superadditive_prompt, rng, n_boot)
    best_no_blocks = max(accuracy[code] for code in expected_codes if code[1] == "0")
    best_no_output = max(accuracy[code] for code in expected_codes if code[3] == "0")
    gates = {
        "training_function": bool(
            accuracy["1111"] - accuracy["0000"] >= 0.20 and accuracy["1111"] >= 0.75
        ),
        "no_single_component_sufficient": bool(
            max(accuracy[code] for code in single_codes) <= accuracy["1111"] - 0.10
        ),
        "trained_blocks_necessary": bool(best_no_blocks <= accuracy["1111"] - 0.10),
        "trained_output_head_necessary": bool(best_no_output <= accuracy["1111"] - 0.10),
        "superadditive_compatibility": bool(super_ci[0] > 0.0),
    }
    diagnostics = {
        "full_training_accuracy_gain": float(accuracy["1111"] - accuracy["0000"]),
        "best_single_component_accuracy": float(max(accuracy[code] for code in single_codes)),
        "best_no_trained_blocks_accuracy": float(best_no_blocks),
        "best_no_trained_output_head_accuracy": float(best_no_output),
        "superadditive_margin_excess_mean": float(superadditive_prompt.mean()),
        "superadditive_margin_excess_ci": list(super_ci),
        "gates": gates,
    }
    return summary, differences, effects, diagnostics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.repo_id, revision="main", cache_dir=args.cache_dir, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    subset, _ = prepare_subset(
        args.lexical_manifest, tokenizer, args.max_problems, "base_cloze"
    )
    subset.to_csv(args.output_dir / "lexical_cloze_subset.csv", index=False, encoding="utf-8-sig")

    load_kwargs = {
        "cache_dir": args.cache_dir,
        "local_files_only": True,
        "dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }
    early = AutoModelForCausalLM.from_pretrained(
        args.repo_id, revision=args.early_revision, **load_kwargs
    ).eval()
    trained = AutoModelForCausalLM.from_pretrained(
        args.repo_id, revision=args.trained_revision, **load_kwargs
    ).eval()
    if early.config.tie_word_embeddings or trained.config.tie_word_embeddings:
        raise RuntimeError("The frozen experiment requires untied input and output weights")
    if early.get_input_embeddings().weight.data_ptr() == early.get_output_embeddings().weight.data_ptr():
        raise RuntimeError("Step-0 input and output weights unexpectedly share storage")

    fingerprints = {
        revision: {
            component: module_fingerprint(module)
            for component, module in component_modules(model).items()
        }
        for revision, model in ((args.early_revision, early), (args.trained_revision, trained))
    }
    working = copy.deepcopy(early).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    working.to(device)
    conditions = args.conditions or [
        condition_code(bits) for bits in itertools.product((0, 1), repeat=4)
    ]
    invalid = [code for code in conditions if len(code) != 4 or set(code) - {"0", "1"}]
    if invalid:
        raise ValueError(f"Invalid condition codes: {invalid}")

    all_records = []
    for code in conditions:
        bits = tuple(int(x) for x in code)
        install_condition(working, early, trained, bits)
        records = evaluate(working, tokenizer, subset, args.batch_size, args.max_length)
        for record in records:
            record.update({
                "condition": code,
                **{component: bit for component, bit in zip(COMPONENTS, bits)},
            })
        all_records.extend(records)
        accuracy = np.mean([record["pair_correct"] for record in records])
        margin = np.mean([record["candidate_margin"] for record in records])
        print(f"condition {code}: accuracy={accuracy:.6f}, margin={margin:.6f}", flush=True)

    outcomes = pd.DataFrame(all_records)
    outcomes.to_csv(args.output_dir / "component_swap_prompt_outcomes.csv", index=False, encoding="utf-8-sig")
    decision: dict[str, object] = {}
    if len(conditions) == 16 and set(conditions) == set(
        condition_code(bits) for bits in itertools.product((0, 1), repeat=4)
    ):
        summary, differences, effects, diagnostics = analyze_factorial(
            outcomes, args.bootstrap, args.seed + 1
        )
        summary.to_csv(args.output_dir / "component_swap_summary.csv", index=False)
        differences.to_csv(args.output_dir / "paired_condition_differences.csv", index=False)
        effects.to_csv(args.output_dir / "orthogonal_factorial_effects.csv", index=False)

        prior = pd.read_csv(args.prior_summary)
        anchors = prior[prior["revision"].isin([args.early_revision, args.trained_revision])].set_index("revision")
        observed = summary.set_index("condition")
        anchor_checks = {
            "early_accuracy_within_one_prompt": bool(
                abs(float(observed.loc["0000", "candidate_pair_accuracy"])
                    - float(anchors.loc[args.early_revision, "candidate_pair_accuracy"])) <= 1 / len(subset)
            ),
            "trained_accuracy_within_one_prompt": bool(
                abs(float(observed.loc["1111", "candidate_pair_accuracy"])
                    - float(anchors.loc[args.trained_revision, "candidate_pair_accuracy"])) <= 1 / len(subset)
            ),
            "early_margin_within_0_02": bool(
                abs(float(observed.loc["0000", "candidate_margin_mean"])
                    - float(anchors.loc[args.early_revision, "candidate_margin_mean"])) <= 0.02
            ),
            "trained_margin_within_0_02": bool(
                abs(float(observed.loc["1111", "candidate_margin_mean"])
                    - float(anchors.loc[args.trained_revision, "candidate_margin_mean"])) <= 0.02
            ),
        }
        diagnostics["anchor_checks"] = anchor_checks
        diagnostics["gates"]["anchor_reproduction"] = bool(all(anchor_checks.values()))
        diagnostics["distributed_compatibility_support"] = bool(
            diagnostics["gates"]["anchor_reproduction"]
            and diagnostics["gates"]["training_function"]
            and diagnostics["gates"]["no_single_component_sufficient"]
            and diagnostics["gates"]["trained_blocks_necessary"]
            and diagnostics["gates"]["trained_output_head_necessary"]
        )
        diagnostics["strict_superadditive_support"] = bool(
            diagnostics["distributed_compatibility_support"]
            and diagnostics["gates"]["superadditive_compatibility"]
        )
        decision = diagnostics

    metadata = {
        "repo_id": args.repo_id,
        "early_revision": args.early_revision,
        "trained_revision": args.trained_revision,
        "n_prompts": len(subset),
        "conditions": conditions,
        "component_order": list(COMPONENTS),
        "component_code_definition": "0=step0, 1=step143000; order E,B,N,W",
        "input_output_tied": False,
        "device": str(device),
        "seed": args.seed,
        "bootstrap": args.bootstrap,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "fingerprints": fingerprints,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if decision:
        (args.output_dir / "gate_decision.json").write_text(
            json.dumps(decision, indent=2), encoding="utf-8"
        )
        print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
