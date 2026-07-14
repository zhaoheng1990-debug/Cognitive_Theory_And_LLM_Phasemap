"""Matched trained-versus-random-initialization layer-order geometry control."""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from trajectory_geometry import (
    geometry_effect_summary,
    summarize_against_endpoint_shuffles,
)


QWEN_PATH = "Qwen/Qwen2.5-1.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--trained-hidden",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--subset-manifest",
        type=Path,
        default=Path("inputs/controlled_subset_manifest.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("random_initialization_geometry"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1701, 1702, 1703])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=260)
    parser.add_argument("--shuffles", type=int, default=50)
    parser.add_argument("--shuffle-seed", type=int, default=20260714)
    parser.add_argument("--max-prompts", type=int)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_subset(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray]:
    metadata = pd.read_csv(args.metadata, encoding="utf-8-sig")
    subset = pd.read_csv(args.subset_manifest, encoding="utf-8-sig")
    if "source_row" not in subset.columns:
        raise ValueError("Subset manifest must contain source_row")
    if args.max_prompts is not None:
        subset = subset.iloc[: args.max_prompts].copy()
    source_rows = subset["source_row"].to_numpy(dtype=int)
    aligned = metadata.iloc[source_rows].reset_index(drop=True)
    if not np.array_equal(aligned["prompt_id"].to_numpy(), subset["prompt_id"].to_numpy()):
        raise RuntimeError("Subset prompt ordering does not match source metadata")
    trained_hidden = np.load(args.trained_hidden, mmap_mode="r")
    trained_states = np.asarray(trained_hidden[source_rows], dtype=np.float32)
    if trained_states.ndim != 3:
        raise ValueError(f"Unexpected trained hidden-state shape: {trained_states.shape}")
    aligned.to_csv(args.output_dir / "matched_prompt_subset.csv", index=False, encoding="utf-8-sig")
    return aligned, trained_states


def initialize_random_model(seed: int):
    set_seed(seed)
    config = AutoConfig.from_pretrained(QWEN_PATH, trust_remote_code=True)
    config.use_cache = False
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    if torch.cuda.is_available():
        model = model.to(device="cuda", dtype=torch.float16)
    else:
        model = model.to(dtype=torch.float32)
    model.eval()
    return model


def extract_random_states(
    prompts: list[str], seed: int, args: argparse.Namespace
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_PATH, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = initialize_random_model(seed)
    n_layers = int(model.config.num_hidden_layers)
    hidden_size = int(model.config.hidden_size)
    states = np.zeros((len(prompts), n_layers, hidden_size), dtype=np.float16)
    device = next(model.parameters()).device
    for start in range(0, len(prompts), args.batch_size):
        end = min(len(prompts), start + args.batch_size)
        encoded = tokenizer(
            prompts[start:end],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(device)
        with torch.no_grad():
            output = model(**encoded, output_hidden_states=True, use_cache=False)
        batch_size = end - start
        row_index = torch.arange(batch_size, device=device)
        position = torch.full(
            (batch_size,), encoded["attention_mask"].shape[1] - 1, dtype=torch.long, device=device
        )
        for layer in range(n_layers):
            layer_state = output.hidden_states[layer + 1][row_index, position, :]
            states[start:end, layer, :] = layer_state.detach().float().cpu().numpy().astype(np.float16)
        del output, encoded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[random seed {seed}] extracted {end}/{len(prompts)}", flush=True)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not np.isfinite(states).all():
        raise RuntimeError(f"Random initialization seed {seed} produced non-finite states")
    return states


def summarize_state(
    states: np.ndarray,
    state_type: str,
    init_seed: int | None,
    args: argparse.Namespace,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    rng = np.random.default_rng(args.shuffle_seed)
    real_means, null_rows = summarize_against_endpoint_shuffles(
        states.astype(np.float32), args.shuffles, rng
    )
    summary = geometry_effect_summary(real_means, null_rows)
    summary.update(
        {
            "state_type": state_type,
            "init_seed": "trained" if init_seed is None else init_seed,
            "n_prompts": states.shape[0],
            "n_layers": states.shape[1],
            "hidden_dim": states.shape[2],
            "n_shuffles": args.shuffles,
        }
    )
    annotated_nulls = []
    for row in null_rows:
        row.update(
            {
                "state_type": state_type,
                "init_seed": "trained" if init_seed is None else init_seed,
            }
        )
        annotated_nulls.append(row)
    return summary, annotated_nulls


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata, trained_states = load_subset(args)
    summary_rows = []
    shuffle_rows = []

    trained_summary, trained_nulls = summarize_state(
        trained_states, "trained_checkpoint", None, args
    )
    summary_rows.append(trained_summary)
    shuffle_rows.extend(trained_nulls)

    for seed in args.seeds:
        random_states = extract_random_states(metadata["prompt"].tolist(), seed, args)
        np.save(args.output_dir / f"random_seed_{seed}_hidden_float16.npy", random_states)
        random_summary, random_nulls = summarize_state(
            random_states, "random_initialization", seed, args
        )
        summary_rows.append(random_summary)
        shuffle_rows.extend(random_nulls)
        del random_states
        gc.collect()

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        args.output_dir / "trained_vs_random_geometry_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(shuffle_rows).to_csv(
        args.output_dir / "trained_vs_random_geometry_shuffles.csv", index=False, encoding="utf-8-sig"
    )

    random_rows = summary[summary["state_type"].eq("random_initialization")]
    comparison = {
        "trained_alignment_gain": float(trained_summary["alignment_gain"]),
        "random_alignment_gain_mean": float(random_rows["alignment_gain"].mean()),
        "random_alignment_gain_min": float(random_rows["alignment_gain"].min()),
        "random_alignment_gain_max": float(random_rows["alignment_gain"].max()),
        "trained_exceeds_random_alignment_gain_range": bool(
            trained_summary["alignment_gain"] > random_rows["alignment_gain"].max()
        ),
        "trained_alignment_z": float(trained_summary["real_vs_shuffle_alignment_mean_z"]),
        "random_alignment_z_mean": float(random_rows["real_vs_shuffle_alignment_mean_z"].mean()),
        "trained_detour_reduction_log2": float(trained_summary["detour_reduction_log2"]),
        "random_detour_reduction_log2_mean": float(random_rows["detour_reduction_log2"].mean()),
        "interpretation_boundary": (
            "Random effects index architecture-compatible organization; trained-versus-random separation indexes "
            "additional parameter-dependent organization for this architecture and prompt subset only."
        ),
    }
    (args.output_dir / "trained_vs_random_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    config = {
        "analysis": "trained versus random-initialization hidden trajectory geometry",
        "architecture": "Qwen2.5-1.5B",
        "model_config_source": QWEN_PATH,
        "metadata": str(args.metadata.resolve()),
        "trained_hidden": str(args.trained_hidden.resolve()),
        "subset_manifest": str(args.subset_manifest.resolve()),
        "n_prompts": len(metadata),
        "random_initialization_seeds": args.seeds,
        "n_endpoint_preserving_shuffles": args.shuffles,
        "shuffle_seed": args.shuffle_seed,
        "prompt_wrapping": "raw prompts, matching the source extraction",
        "claim_boundary": "one architecture and controlled prompt subset; not a universal architecture-training decomposition",
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
