"""Cross-checkpoint trained-versus-random-initialization geometry control."""

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


SOURCE_ROOT = Path("inputs/trained_trajectory_arrays")
MODEL_SPECS = {
    "qwen": {
        "path": Path("Qwen/Qwen2.5-1.5B-Instruct"),
        "metadata": SOURCE_ROOT / "qwen_relation_metadata.csv",
        "trained_hidden": SOURCE_ROOT / "qwen_hidden_last_token_layers.npy",
    },
    "llama": {
        "path": Path("meta-llama/Llama-3.2-1B-Instruct"),
        "metadata": SOURCE_ROOT / "llama_relation_metadata.csv",
        "trained_hidden": SOURCE_ROOT / "llama_hidden_last_token_layers.npy",
    },
    "gemma": {
        "path": Path("google/gemma-2-2b-it"),
        "metadata": SOURCE_ROOT / "gemma_relation_metadata.csv",
        "trained_hidden": SOURCE_ROOT / "gemma_hidden_last_token_layers.npy",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument(
        "--subset-manifest",
        type=Path,
        default=Path("direct_hidden_geometry_v0_1/controlled_subset_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("crossmodel_random_initialization_geometry")
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1701, 1702, 1703])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=260)
    parser.add_argument("--shuffles", type=int, default=50)
    parser.add_argument("--shuffle-seed", type=int, default=20260717)
    parser.add_argument("--max-prompts", type=int)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_subset(
    model_key: str, subset_manifest: Path, max_prompts: int | None
) -> tuple[pd.DataFrame, np.ndarray]:
    spec = MODEL_SPECS[model_key]
    metadata = pd.read_csv(spec["metadata"], encoding="utf-8-sig")
    subset = pd.read_csv(subset_manifest, encoding="utf-8-sig")
    if "source_row" not in subset.columns:
        raise ValueError("Subset manifest must contain source_row")
    if max_prompts is not None:
        subset = subset.iloc[:max_prompts].copy()
    source_rows = subset["source_row"].to_numpy(dtype=int)
    aligned = metadata.iloc[source_rows].reset_index(drop=True)
    if not np.array_equal(aligned["prompt_id"].to_numpy(), subset["prompt_id"].to_numpy()):
        raise RuntimeError(f"{model_key}: subset prompt ordering does not match source metadata")
    trained_hidden = np.load(spec["trained_hidden"], mmap_mode="r")
    trained_states = np.asarray(trained_hidden[source_rows], dtype=np.float32)
    if trained_states.ndim != 3:
        raise ValueError(f"{model_key}: unexpected trained hidden-state shape {trained_states.shape}")
    return aligned, trained_states


def initialize_random_model(model_key: str, seed: int):
    set_seed(seed)
    model_path = MODEL_SPECS[model_key]["path"]
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = False
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, dtype=dtype)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model


def extract_random_states(
    model_key: str,
    prompts: list[str],
    seed: int,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    model_path = MODEL_SPECS[model_key]["path"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = initialize_random_model(model_key, seed)
    n_layers = int(model.config.num_hidden_layers)
    hidden_size = int(model.config.hidden_size)
    states = np.zeros((len(prompts), n_layers, hidden_size), dtype=np.float16)
    device = next(model.parameters()).device
    for start in range(0, len(prompts), batch_size):
        end = min(len(prompts), start + batch_size)
        encoded = tokenizer(
            prompts[start:end],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            output = model(**encoded, output_hidden_states=True, use_cache=False)
        batch_n = end - start
        row_index = torch.arange(batch_n, device=device)
        position = torch.full(
            (batch_n,), encoded["attention_mask"].shape[1] - 1, dtype=torch.long, device=device
        )
        for layer in range(n_layers):
            state = output.hidden_states[layer + 1][row_index, position, :]
            states[start:end, layer, :] = state.detach().float().cpu().numpy().astype(np.float16)
        del output, encoded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[{model_key} random seed {seed}] extracted {end}/{len(prompts)}", flush=True)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not np.isfinite(states).all():
        raise RuntimeError(f"{model_key} seed {seed}: non-finite random hidden states")
    return states


def summarize_state(
    states: np.ndarray,
    model_key: str,
    state_type: str,
    init_seed: int | None,
    shuffles: int,
    shuffle_seed: int,
) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(shuffle_seed)
    real_means, null_rows = summarize_against_endpoint_shuffles(
        states.astype(np.float32), shuffles, rng
    )
    summary = geometry_effect_summary(real_means, null_rows)
    summary.update(
        {
            "model": model_key,
            "state_type": state_type,
            "init_seed": "trained" if init_seed is None else init_seed,
            "n_prompts": states.shape[0],
            "n_layers": states.shape[1],
            "hidden_dim": states.shape[2],
            "n_shuffles": shuffles,
        }
    )
    for row in null_rows:
        row.update(
            {
                "model": model_key,
                "state_type": state_type,
                "init_seed": "trained" if init_seed is None else init_seed,
            }
        )
    return summary, null_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    all_nulls = []
    comparison_rows = []

    for model_offset, model_key in enumerate(args.models):
        model_dir = args.output_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        metadata, trained_states = load_subset(model_key, args.subset_manifest, args.max_prompts)
        metadata.to_csv(model_dir / "matched_prompt_subset.csv", index=False, encoding="utf-8-sig")

        trained_summary, trained_nulls = summarize_state(
            trained_states,
            model_key,
            "trained_checkpoint",
            None,
            args.shuffles,
            args.shuffle_seed + model_offset * 100,
        )
        model_summaries = [trained_summary]
        all_summaries.append(trained_summary)
        all_nulls.extend(trained_nulls)

        for seed in args.seeds:
            cache_path = model_dir / f"random_seed_{seed}_hidden_float16.npy"
            if cache_path.exists():
                states = np.load(cache_path)
                expected = (len(metadata), trained_states.shape[1], trained_states.shape[2])
                if states.shape != expected or not np.isfinite(states).all():
                    raise RuntimeError(f"{cache_path}: invalid cached shape or values")
                print(f"[{model_key} random seed {seed}] reuse cached states", flush=True)
            else:
                states = extract_random_states(
                    model_key,
                    metadata["prompt"].tolist(),
                    seed,
                    args.batch_size,
                    args.max_length,
                )
                np.save(cache_path, states)
            random_summary, random_nulls = summarize_state(
                states,
                model_key,
                "random_initialization",
                seed,
                args.shuffles,
                args.shuffle_seed + model_offset * 100,
            )
            model_summaries.append(random_summary)
            all_summaries.append(random_summary)
            all_nulls.extend(random_nulls)
            del states
            gc.collect()

        model_frame = pd.DataFrame(model_summaries)
        model_frame.to_csv(
            model_dir / "trained_vs_random_geometry_summary.csv", index=False, encoding="utf-8-sig"
        )
        random_frame = model_frame[model_frame["state_type"].eq("random_initialization")]
        comparison_rows.append(
            {
                "model": model_key,
                "trained_alignment_gain": float(trained_summary["alignment_gain"]),
                "random_alignment_gain_mean": float(random_frame["alignment_gain"].mean()),
                "random_alignment_gain_min": float(random_frame["alignment_gain"].min()),
                "random_alignment_gain_max": float(random_frame["alignment_gain"].max()),
                "all_random_alignment_gains_positive": bool((random_frame["alignment_gain"] > 0).all()),
                "trained_exceeds_random_range": bool(
                    float(trained_summary["alignment_gain"]) > random_frame["alignment_gain"].max()
                ),
                "trained_alignment_z": float(trained_summary["real_vs_shuffle_alignment_mean_z"]),
                "random_alignment_z_mean": float(
                    random_frame["real_vs_shuffle_alignment_mean_z"].mean()
                ),
            }
        )

    pd.DataFrame(all_summaries).to_csv(
        args.output_dir / "crossmodel_trained_vs_random_geometry_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(all_nulls).to_csv(
        args.output_dir / "crossmodel_trained_vs_random_geometry_shuffles.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        args.output_dir / "crossmodel_trained_vs_random_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    config = {
        "analysis": "cross-checkpoint trained versus random-initialization trajectory geometry",
        "model_specs": {
            key: {name: str(value) for name, value in MODEL_SPECS[key].items()}
            for key in args.models
        },
        "subset_manifest": str(args.subset_manifest.resolve()),
        "n_prompts": int(len(metadata)),
        "random_initialization_seeds": args.seeds,
        "n_endpoint_preserving_shuffles": args.shuffles,
        "shuffle_seed": args.shuffle_seed,
        "prompt_wrapping": "raw prompts matching source extraction",
        "claim_boundary": (
            "within-architecture decomposition of architecture-compatible and parameter-dependent order; "
            "not a universal architecture law"
        ),
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
