"""Independent lexical-category task validation using shared trajectory definitions."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from external_task_shared import (
    CONDITIONS,
    MECHANISM_ID,
    MODEL_SPECS,
    delta_u_validation,
    extract_model,
    set_seed,
)
from trajectory_geometry import (
    geometry_effect_summary,
    summarize_against_endpoint_shuffles,
)


CATEGORIES = {
    "colour": ("Red", "Blue", "Green", "Yellow"),
    "direction": ("North", "South", "East", "West"),
    "metal": ("Copper", "Silver", "Gold", "Iron"),
    "shape": ("Circle", "Square", "Triangle", "Star"),
    "landscape feature": ("River", "Mountain", "Forest", "Ocean"),
    "sky object": ("Sun", "Moon", "Cloud", "Star"),
    "fruit": ("Apple", "Orange", "Lemon", "Pear"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("lexical_category_validation"))
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--problems", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=240)
    parser.add_argument("--shuffles", type=int, default=50)
    parser.add_argument("--label-shuffles", type=int, default=200)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--raw-prompts", action="store_true")
    return parser.parse_args()


def make_prompt(
    condition: str,
    category: str,
    candidate_one: str,
    candidate_two: str,
    conflict_word: str,
) -> str:
    demonstrations = [
        "Example 1: Which candidate is a colour? Red or River. Answer: Red.",
        "Example 2: Which candidate is a fruit? Stone or Pear. Answer: Pear.",
        "Now answer the new item.",
    ]
    core = [
        f"Target category: {category}.",
        f"Candidate 1: {candidate_one}",
        f"Candidate 2: {candidate_two}",
    ]
    if condition == "clean":
        qualifier = []
    elif condition == "paraphrase":
        core[0] = f"Select the word that belongs to the category {category}."
        qualifier = []
    elif condition == "untrusted_conflict":
        qualifier = [f"An unverified note claims {conflict_word}. Check the category yourself."]
    elif condition == "asserted_conflict":
        qualifier = [f"A previous annotator selected {conflict_word}. Independently classify the words."]
    elif condition == "explicit_correction":
        qualifier = [f"Initial note: {conflict_word}. Correction: that note was withdrawn; use category membership."]
    elif condition == "final_recheck":
        qualifier = [f"Draft answer: {conflict_word}. Final instruction: recheck the category and replace the draft if needed."]
    else:
        raise ValueError(condition)
    ending = ["Which candidate belongs to the target category?", "Answer with exactly one candidate word:"]
    return "\n".join(demonstrations + core + qualifier + ending)


def build_dataset(n_problems: int) -> pd.DataFrame:
    categories = list(CATEGORIES)
    rows = []
    for problem_id in range(n_problems):
        category_index = problem_id % len(categories)
        category = categories[category_index]
        distractor_category = categories[(category_index + 1 + problem_id // len(categories)) % len(categories)]
        if distractor_category == category:
            distractor_category = categories[(category_index + 1) % len(categories)]
        member_index = (problem_id // len(categories)) % 4
        clean_word = CATEGORIES[category][member_index]
        distractor_word = CATEGORIES[distractor_category][(member_index + problem_id) % 4]
        if clean_word == distractor_word:
            distractor_word = CATEGORIES[distractor_category][(member_index + 1) % 4]
        candidate_one = clean_word if problem_id % 2 == 0 else distractor_word
        candidate_two = clean_word if problem_id % 2 == 1 else distractor_word
        for condition, mechanism in CONDITIONS:
            rows.append(
                {
                    "prompt_id": f"lexical_{problem_id:03d}_{condition}",
                    "problem_id": problem_id,
                    "condition": condition,
                    "mechanism": mechanism,
                    "mechanism_id": MECHANISM_ID[mechanism],
                    "expression": f"category={category}",
                    "correct_value": clean_word,
                    "distractor_value": distractor_word,
                    "clean_label": clean_word,
                    "conflict_label": distractor_word,
                    "prompt": make_prompt(
                        condition, category, candidate_one, candidate_two, distractor_word
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_prefix = "lexical"
    set_seed(args.seed)
    dataset = build_dataset(args.problems)
    dataset.to_csv(args.output_dir / "lexical_task_manifest.csv", index=False, encoding="utf-8-sig")
    geometry_rows = []
    delta_rows = []
    geometry_null_rows = []
    label_null_rows = []
    for model_offset, model_key in enumerate(args.models):
        model_dir = args.output_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        hidden, margins, outcomes = extract_model(model_key, dataset, args, model_dir)
        geometry_rng = np.random.default_rng(args.seed + 3000 * (model_offset + 1))
        real_means, null_rows = summarize_against_endpoint_shuffles(
            hidden.astype(np.float32), args.shuffles, geometry_rng
        )
        geometry = geometry_effect_summary(real_means, null_rows)
        geometry.update(
            {
                "model": model_key,
                "n_prompts": len(outcomes),
                "n_problems": outcomes["problem_id"].nunique(),
                "n_layers": hidden.shape[1],
                "hidden_dim": hidden.shape[2],
                "pair_accuracy": float(outcomes["pair_choice"].eq("clean").mean()),
                "clean_pair_accuracy": float(
                    outcomes.loc[outcomes["condition"].eq("clean"), "pair_choice"].eq("clean").mean()
                ),
                "n_shuffles": args.shuffles,
            }
        )
        geometry_rows.append(geometry)
        for row in null_rows:
            row["model"] = model_key
            geometry_null_rows.append(row)

        delta_rng = np.random.default_rng(args.seed + 4000 * (model_offset + 1))
        annotated, delta_summary, label_null = delta_u_validation(
            model_key, margins, outcomes, args, delta_rng
        )
        annotated.to_csv(model_dir / "lexical_DeltaU_rows.csv", index=False, encoding="utf-8-sig")
        delta_rows.append(delta_summary)
        label_null_rows.append(label_null)
        del hidden, margins
        gc.collect()

    pd.DataFrame(geometry_rows).to_csv(
        args.output_dir / "external_task_geometry_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(geometry_null_rows).to_csv(
        args.output_dir / "external_task_geometry_shuffles.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(delta_rows).to_csv(
        args.output_dir / "external_task_DeltaU_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(label_null_rows, ignore_index=True).to_csv(
        args.output_dir / "external_task_DeltaU_label_nulls.csv", index=False, encoding="utf-8-sig"
    )
    config = {
        "analysis": "independent lexical-category candidate-evaluation trajectory validation",
        "models": args.models,
        "n_problems": args.problems,
        "categories": CATEGORIES,
        "conditions": [{"condition": condition, "mechanism": mechanism} for condition, mechanism in CONDITIONS],
        "candidate_order_counterbalancing": "correct word first for even problem_id and second for odd problem_id",
        "frozen_decision_windows": {
            key: list(MODEL_SPECS[key]["decision_layers"]) for key in args.models
        },
        "split_unit": "lexical problem_id",
        "test_size": args.test_size,
        "prompt_wrapping": "raw" if args.raw_prompts else "checkpoint-native chat template",
        "geometry_null": "endpoint-preserving intermediate-layer permutation",
        "n_geometry_shuffles": args.shuffles,
        "n_label_shuffles": args.label_shuffles,
        "seed": args.seed,
        "claim_boundary": "external task-family validation, not a universal transformer law or transfer of the numeric DeltaU axis",
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
