"""Track kinematic and semantic coordinates through a documented training run."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

from analyze_component_geometry import (
    semantic_enrichment,
    trajectory_summary,
)


DEFAULT_REVISIONS = [
    "step0",
    "step128",
    "step1000",
    "step4000",
    "step16000",
    "step64000",
    "step143000",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="EleutherAI/pythia-160m")
    parser.add_argument("--revisions", nargs="+", default=DEFAULT_REVISIONS)
    parser.add_argument(
        "--lexical-manifest",
        type=Path,
        default=Path("inputs/lexical_task/lexical_task_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("architecture_semantic_geometry_v0_1/phase4_training_time"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("HF_TRAINING_CACHE", "hf_cache")),
    )
    parser.add_argument("--max-problems", type=int, default=64)
    parser.add_argument(
        "--prompt-style",
        choices=("source_instruction", "base_cloze"),
        default="source_instruction",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--analysis-seed", type=int, default=2026071704)
    parser.add_argument("--semantic-label-nulls", type=int, default=5000)
    parser.add_argument("--donor-nulls", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--weight-sample", type=int, default=2048)
    return parser.parse_args()


def single_token_id(tokenizer, label: str) -> int | None:
    ids = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
    return int(ids[0]) if len(ids) == 1 else None


def prepare_subset(
    manifest_path: Path, tokenizer, max_problems: int, prompt_style: str
):
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    clean_manifest = manifest[manifest["condition"].eq("clean")].copy()
    clean_manifest["category"] = clean_manifest["expression"].str.replace(
        "category=", "", regex=False
    )
    subset = (
        clean_manifest
        .sort_values("problem_id")
        .iloc[:max_problems]
        .copy()
    )
    subset["source_prompt"] = subset["prompt"]
    if prompt_style == "base_cloze":
        plurals = {
            "colour": "colours",
            "direction": "directions",
            "metal": "metals",
            "shape": "shapes",
            "landscape feature": "landscape features",
            "sky object": "sky objects",
            "fruit": "fruits",
        }
        category_words = {
            category: sorted(group["clean_label"].astype(str).unique().tolist())
            for category, group in clean_manifest.groupby("category")
        }
        prompts = []
        for _, row in subset.iterrows():
            examples = [
                word
                for word in category_words[str(row["category"])]
                if word != str(row["clean_label"])
            ][:2]
            if len(examples) < 2:
                raise RuntimeError(f"Insufficient cloze examples for {row['category']}")
            prompts.append(
                f"{examples[0]} and {examples[1]} are {plurals[str(row['category'])]}. "
                f"Another {row['category']} is"
            )
        subset["analysis_prompt"] = prompts
    else:
        subset["analysis_prompt"] = subset["prompt"]
    subset["clean_token_id"] = [single_token_id(tokenizer, x) for x in subset["clean_label"]]
    subset["conflict_token_id"] = [single_token_id(tokenizer, x) for x in subset["conflict_label"]]
    valid = subset["clean_token_id"].notna() & subset["conflict_token_id"].notna()
    subset = subset[valid].reset_index(drop=True)
    subset["clean_token_id"] = subset["clean_token_id"].astype(int)
    subset["conflict_token_id"] = subset["conflict_token_id"].astype(int)
    minimum_required = min(32, max(4, max_problems // 2))
    if len(subset) < minimum_required:
        raise RuntimeError(f"Only {len(subset)} lexical prompts have single-token candidates")

    token_rows = []
    for _, row in subset.iterrows():
        token_rows.append(
            {
                "word": row["clean_label"],
                "category": row["category"],
                "token_id": row["clean_token_id"],
            }
        )
    token_frame = pd.DataFrame(token_rows).drop_duplicates(["word", "category"])
    ambiguous = token_frame.groupby("word")["category"].nunique()
    ambiguous_words = set(ambiguous[ambiguous > 1].index)
    token_frame = token_frame[~token_frame["word"].isin(ambiguous_words)]
    token_frame = token_frame.drop_duplicates("token_id").reset_index(drop=True)
    token_frame["category_id"] = pd.Categorical(token_frame["category"]).codes
    return subset, token_frame


def extract_checkpoint(
    model,
    tokenizer,
    subset: pd.DataFrame,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_layers = int(model.config.num_hidden_layers)
    hidden_size = int(model.config.hidden_size)
    states = np.zeros((len(subset), n_layers, hidden_size), dtype=np.float16)
    margins = np.zeros(len(subset), dtype=np.float32)
    pair_correct = np.zeros(len(subset), dtype=bool)
    device = next(model.parameters()).device
    prompts = subset["analysis_prompt"].astype(str).tolist()
    clean_ids = subset["clean_token_id"].to_numpy(dtype=int)
    conflict_ids = subset["conflict_token_id"].to_numpy(dtype=int)
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else model.base_model
    for start in range(0, len(subset), batch_size):
        end = min(start + batch_size, len(subset))
        encoded = tokenizer(
            prompts[start:end],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            output = decoder(
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
        clean = torch.as_tensor(clean_ids[start:end].copy(), device=device, dtype=torch.long)
        conflict = torch.as_tensor(
            conflict_ids[start:end].copy(), device=device, dtype=torch.long
        )
        weight = model.get_output_embeddings().weight
        margin = torch.sum(final_state * weight[clean], dim=-1) - torch.sum(
            final_state * weight[conflict], dim=-1
        )
        margins[start:end] = margin.detach().float().cpu().numpy()
        pair_correct[start:end] = (margin > 0).detach().cpu().numpy()
        for layer_index in range(n_layers):
            values = output.hidden_states[layer_index + 1][rows, position, :]
            states[start:end, layer_index, :] = (
                values.detach().float().cpu().numpy().astype(np.float16)
            )
        print(f"extracted {end}/{len(subset)}", flush=True)
        del encoded, output, final_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return states, margins, pair_correct


def sampled_weight_geometry(
    weight: torch.Tensor, sample_ids: np.ndarray, rng: np.random.Generator
) -> dict[str, float]:
    index = torch.as_tensor(sample_ids.copy(), device=weight.device, dtype=torch.long)
    rows = weight[index].detach().float().cpu().numpy()
    rows -= rows.mean(axis=0, keepdims=True)
    covariance = (rows.T @ rows) / max(len(rows) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance.astype(np.float64))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    participation = float(eigenvalues.sum() ** 2 / max(np.sum(eigenvalues**2), 1e-12))
    unit = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-8)
    left = rng.integers(0, len(rows), size=20_000)
    right = rng.integers(0, len(rows), size=20_000)
    distinct = left != right
    pair_cosine = np.sum(unit[left[distinct]] * unit[right[distinct]], axis=1)
    return {
        "weight_participation_ratio": participation,
        "weight_participation_ratio_fraction": participation / rows.shape[1],
        "weight_pair_cosine_mean": float(np.mean(pair_cosine)),
        "weight_pair_cosine_abs_mean": float(np.mean(np.abs(pair_cosine))),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.repo_id, revision="main", cache_dir=args.cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    subset, token_frame = prepare_subset(
        args.lexical_manifest, tokenizer, args.max_problems, args.prompt_style
    )
    subset.to_csv(args.output_dir / "phase4_lexical_subset.csv", index=False, encoding="utf-8-sig")
    token_frame.to_csv(
        args.output_dir / "phase4_semantic_token_audit.csv", index=False, encoding="utf-8-sig"
    )
    semantic_ids = token_frame["token_id"].to_numpy(dtype=int)
    semantic_labels = token_frame["category_id"].to_numpy(dtype=int)
    api = HfApi()
    repo_refs = {branch.name for branch in api.list_repo_refs(args.repo_id).branches}
    missing = [revision for revision in args.revisions if revision not in repo_refs]
    if missing:
        raise ValueError(f"Unavailable revisions: {missing}")

    metadata_rows = []
    summary_rows = []
    row_records = []
    sample_rng = np.random.default_rng(args.analysis_seed)
    sample_ids: np.ndarray | None = None
    for revision_index, revision in enumerate(args.revisions):
        step = int(revision.replace("step", ""))
        print(f"loading {args.repo_id}@{revision}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.repo_id,
            revision=revision,
            cache_dir=args.cache_dir,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        model.config.use_cache = False
        info = api.model_info(args.repo_id, revision=revision)
        metadata_rows.append(
            {"revision": revision, "training_step": step, "commit_sha": info.sha}
        )
        if sample_ids is None:
            sample_ids = sample_rng.choice(
                model.config.vocab_size,
                size=min(args.weight_sample, model.config.vocab_size),
                replace=False,
            ).astype(int)
        states, margins, pair_correct = extract_checkpoint(
            model, tokenizer, subset, args.batch_size, args.max_length
        )
        revision_dir = args.output_dir / revision
        revision_dir.mkdir(parents=True, exist_ok=True)
        np.save(revision_dir / "hidden_last_token_layers_float16.npy", states)
        pd.DataFrame(
            {
                "prompt_id": subset["prompt_id"],
                "problem_id": subset["problem_id"],
                "candidate_margin": margins,
                "pair_correct": pair_correct,
            }
        ).to_csv(revision_dir / "lexical_prompt_outcomes.csv", index=False, encoding="utf-8-sig")
        rng = np.random.default_rng(args.analysis_seed + revision_index * 10_000)
        summary = {
            "revision": revision,
            "training_step": step,
            "n_prompts": len(subset),
            "candidate_pair_accuracy": float(np.mean(pair_correct)),
            "candidate_margin_mean": float(np.mean(margins)),
            "candidate_margin_median": float(np.median(margins)),
        }
        summary.update(trajectory_summary(states, rng, args.donor_nulls, args.bootstrap))
        semantic_index = torch.as_tensor(
            semantic_ids.copy(), device=next(model.parameters()).device, dtype=torch.long
        )
        output_rows = (
            model.get_output_embeddings().weight[semantic_index].detach().float().cpu().numpy()
        )
        input_rows = (
            model.get_input_embeddings().weight[semantic_index].detach().float().cpu().numpy()
        )
        summary.update(
            {f"output_{key}": value for key, value in semantic_enrichment(
                output_rows, semantic_labels, rng, args.semantic_label_nulls
            ).items()}
        )
        summary.update(
            {f"embedding_{key}": value for key, value in semantic_enrichment(
                input_rows, semantic_labels, rng, args.semantic_label_nulls
            ).items()}
        )
        summary["embedding_output_same_token_cosine_mean"] = float(
            np.mean(
                np.sum(input_rows * output_rows, axis=1)
                / np.maximum(
                    np.linalg.norm(input_rows, axis=1) * np.linalg.norm(output_rows, axis=1),
                    1e-8,
                )
            )
        )
        summary.update(sampled_weight_geometry(model.get_output_embeddings().weight, sample_ids, rng))
        summary_rows.append(summary)
        for index, prompt_id in enumerate(subset["prompt_id"]):
            row_records.append(
                {
                    "revision": revision,
                    "training_step": step,
                    "prompt_id": prompt_id,
                    "problem_id": int(subset.iloc[index]["problem_id"]),
                    "candidate_margin": float(margins[index]),
                    "pair_correct": bool(pair_correct[index]),
                }
            )
        del model, states, output_rows, input_rows
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_frame = pd.DataFrame(summary_rows).sort_values("training_step")
    summary_frame.to_csv(
        args.output_dir / "phase4_training_time_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(row_records).to_csv(
        args.output_dir / "phase4_prompt_outcomes.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(metadata_rows).to_csv(
        args.output_dir / "phase4_checkpoint_metadata.csv", index=False, encoding="utf-8-sig"
    )
    initial = summary_frame.iloc[0]
    final = summary_frame.iloc[-1]
    log_step = np.log1p(summary_frame["training_step"].to_numpy(dtype=float))
    trend_metrics = [
        "output_semantic_enrichment",
        "embedding_semantic_enrichment",
        "candidate_pair_accuracy",
        "unit_hidden_state_global_chord_alignment_mean",
        "unit_hidden_state_future_chord_alignment_mean",
        "unit_hidden_state_leave_one_out_alignment_mean",
    ]
    trends = {
        metric: float(np.corrcoef(log_step, summary_frame[metric].to_numpy(dtype=float))[0, 1])
        for metric in trend_metrics
    }
    gate = {
        "phase": 4,
        "checkpoint_series": args.repo_id,
        "n_checkpoints": len(summary_frame),
        "initial_revision": initial["revision"],
        "final_revision": final["revision"],
        "output_semantic_enrichment_initial": float(initial["output_semantic_enrichment"]),
        "output_semantic_enrichment_final": float(final["output_semantic_enrichment"]),
        "output_semantic_enrichment_final_p": float(final["output_semantic_enrichment_null_p"]),
        "candidate_pair_accuracy_initial": float(initial["candidate_pair_accuracy"]),
        "candidate_pair_accuracy_final": float(final["candidate_pair_accuracy"]),
        "training_trend_correlations": trends,
        "semantic_training_gate": bool(
            final["output_semantic_enrichment"] > initial["output_semantic_enrichment"]
            and final["output_semantic_enrichment_null_p"] <= 0.05
        ),
        "claim_boundary": (
            "one documented Pythia-160m training trajectory on a lexical diagnostic; "
            "temporal association, not a universal training law"
        ),
    }
    (args.output_dir / "phase4_gate_decision.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    config = {
        "repo_id": args.repo_id,
        "revisions": args.revisions,
        "cache_dir": str(args.cache_dir),
        "task": "single-token clean lexical-category candidate evaluation",
        "prompt_style": args.prompt_style,
        "n_prompts": len(subset),
        "n_semantic_tokens": len(token_frame),
        "semantic_label_nulls": args.semantic_label_nulls,
        "donor_nulls": args.donor_nulls,
        "bootstrap": args.bootstrap,
        "weight_sample": args.weight_sample,
        "analysis_seed": args.analysis_seed,
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
