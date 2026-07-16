"""Independent arithmetic-task validation of trajectory geometry and DeltaU construction."""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModelForCausalLM, AutoTokenizer

from trajectory_geometry import (
    geometry_effect_summary,
    summarize_against_endpoint_shuffles,
)


MODEL_SPECS = {
    "qwen": {
        "path": r"Qwen/Qwen2.5-1.5B-Instruct",
        "decision_layers": (20, 21, 22, 23, 24, 25),
    },
    "llama": {
        "path": r"meta-llama/Llama-3.2-1B-Instruct",
        "decision_layers": (12, 13, 14, 15),
    },
    "gemma": {
        "path": r"google/gemma-2-2b-it",
        "decision_layers": (18, 19, 20, 21, 22, 23),
    },
}

CONDITIONS = (
    ("clean", "stable"),
    ("paraphrase", "stable"),
    ("untrusted_conflict", "competition"),
    ("asserted_conflict", "competition"),
    ("explicit_correction", "closure"),
    ("final_recheck", "closure"),
)
MECHANISM_ID = {"stable": 0, "competition": 1, "closure": 2}
NUMBER_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("arithmetic_task_validation"))
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--problems", type=int, default=64)
    parser.add_argument("--operation-family", choices=("mixed", "addition"), default="mixed")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=220)
    parser.add_argument("--shuffles", type=int, default=50)
    parser.add_argument("--label-shuffles", type=int, default=200)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--raw-prompts", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def arithmetic_problem(index: int, operation_family: str = "mixed") -> tuple[str, int, int]:
    if operation_family == "addition":
        left = 1 + index % 8
        right = 1 + (index // 8) % 8
        answer = left + right
        offset = (1, -1, 2, -2, 3, -3)[index % 6]
        distractor = answer + offset
        if distractor < 0 or distractor > 20:
            distractor = answer - offset
        return f"{left} + {right}", answer, distractor
    operation = index % 3
    if operation == 0:
        left = 1 + (7 * index) % 9
        right = 1 + (5 * index) % 9
        expression = f"{left} + {right}"
        answer = left + right
    elif operation == 1:
        left = 7 + (7 * index) % 13
        right = 1 + (3 * index) % min(9, left)
        expression = f"{left} - {right}"
        answer = left - right
    else:
        left = 2 + (3 * index) % 4
        right = 2 + (5 * index) % 3
        expression = f"{left} * {right}"
        answer = left * right
    offset = (1, -1, 2, -2, 3, -3)[index % 6]
    distractor = answer + offset
    if distractor < 0 or distractor > 20:
        distractor = answer + abs(offset)
    if distractor > 20:
        distractor = answer - abs(offset)
    return expression, answer, distractor


def make_prompt(
    condition: str,
    expression: str,
    option_a: str,
    option_b: str,
    conflict_label: str,
) -> str:
    demonstrations = [
        "Example 1: 2 + 3. Candidates: five, six. Answer: five.",
        "Example 2: 4 * 3. Candidates: eleven, twelve. Answer: twelve.",
        "Now solve the new problem.",
    ]
    option_block = f"Candidate 1: {option_a}\nCandidate 2: {option_b}"
    if condition == "clean":
        lines = [f"Calculate {expression}.", option_block]
    elif condition == "paraphrase":
        lines = [f"Work out the value of {expression} and compare the two choices.", option_block]
    elif condition == "untrusted_conflict":
        lines = [
            f"Calculate {expression}.",
            option_block,
            f"An unverified note claims that {conflict_label} is correct. Check the arithmetic yourself.",
        ]
    elif condition == "asserted_conflict":
        lines = [
            f"Calculate {expression}.",
            option_block,
            f"A previous solver selected {conflict_label}. Independently verify the result.",
        ]
    elif condition == "explicit_correction":
        lines = [
            f"Calculate {expression}.",
            option_block,
            f"Initial note: {conflict_label}. Correction: that note was withdrawn; use the arithmetic result.",
        ]
    elif condition == "final_recheck":
        lines = [
            f"Calculate {expression}.",
            option_block,
            f"Draft answer: {conflict_label}. Final instruction: recheck the calculation and replace the draft if needed.",
        ]
    else:
        raise ValueError(condition)
    lines.extend(("Which candidate is correct?", "Answer with exactly one candidate word:"))
    return "\n".join(demonstrations + lines)


def build_dataset(n_problems: int, operation_family: str = "mixed") -> pd.DataFrame:
    rows = []
    for problem_id in range(n_problems):
        expression, answer, distractor = arithmetic_problem(problem_id, operation_family)
        clean_label = NUMBER_WORDS[answer]
        conflict_label = NUMBER_WORDS[distractor]
        option_a = clean_label if problem_id % 2 == 0 else conflict_label
        option_b = clean_label if problem_id % 2 == 1 else conflict_label
        for condition, mechanism in CONDITIONS:
            rows.append(
                {
                    "prompt_id": f"arith_{problem_id:03d}_{condition}",
                    "problem_id": problem_id,
                    "condition": condition,
                    "mechanism": mechanism,
                    "mechanism_id": MECHANISM_ID[mechanism],
                    "expression": expression,
                    "correct_value": answer,
                    "distractor_value": distractor,
                    "clean_label": clean_label,
                    "conflict_label": conflict_label,
                    "prompt": make_prompt(condition, expression, option_a, option_b, conflict_label),
                }
            )
    return pd.DataFrame(rows)


def continuation_id(tokenizer, label: str) -> tuple[int, list[int]]:
    ids = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise RuntimeError(f"Label {label!r} is not a single continuation token: {ids}")
    return int(ids[0]), ids


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return model, tokenizer


def extract_model(
    model_key: str, dataset: pd.DataFrame, args: argparse.Namespace, model_dir: Path
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    model_path = MODEL_SPECS[model_key]["path"]
    model, tokenizer = load_model(model_path)
    label_ids = {}
    token_rows = []
    unique_labels = sorted(set(dataset["clean_label"]) | set(dataset["conflict_label"]))
    for label in unique_labels:
        token_id, ids = continuation_id(tokenizer, label)
        label_ids[label] = token_id
        token_rows.append({"label": label, "token_id": token_id, "token_ids": str(ids)})
    pd.DataFrame(token_rows).to_csv(model_dir / "tokenization_audit.csv", index=False)

    n_rows = len(dataset)
    n_layers = int(model.config.num_hidden_layers)
    hidden_size = int(model.config.hidden_size)
    hidden = np.zeros((n_rows, n_layers, hidden_size), dtype=np.float16)
    margins = np.zeros((n_rows, n_layers), dtype=np.float32)
    final_clean = np.zeros(n_rows, dtype=np.float32)
    final_conflict = np.zeros(n_rows, dtype=np.float32)
    lm_head = model.lm_head.weight.detach().float().to(model.device)

    for start in range(0, n_rows, args.batch_size):
        end = min(n_rows, start + args.batch_size)
        batch = dataset.iloc[start:end]
        prompts = batch["prompt"].tolist()
        if not args.raw_prompts and hasattr(tokenizer, "apply_chat_template"):
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(model.device)
        with torch.no_grad():
            output = model(**encoded, output_hidden_states=True, use_cache=False)
        batch_size = end - start
        row_index = torch.arange(batch_size, device=model.device)
        position = torch.full(
            (batch_size,), encoded["attention_mask"].shape[1] - 1, dtype=torch.long, device=model.device
        )
        clean_ids = torch.tensor(
            [label_ids[label] for label in batch["clean_label"]], dtype=torch.long, device=model.device
        )
        conflict_ids = torch.tensor(
            [label_ids[label] for label in batch["conflict_label"]], dtype=torch.long, device=model.device
        )
        for layer in range(n_layers):
            state = output.hidden_states[layer + 1][row_index, position, :].detach().float()
            hidden[start:end, layer, :] = state.cpu().numpy().astype(np.float16)
            clean_score = torch.sum(state * lm_head[clean_ids], dim=1)
            conflict_score = torch.sum(state * lm_head[conflict_ids], dim=1)
            margins[start:end, layer] = (clean_score - conflict_score).cpu().numpy()
        logits = output.logits[row_index, position, :].detach().float()
        final_clean[start:end] = logits[row_index, clean_ids].cpu().numpy()
        final_conflict[start:end] = logits[row_index, conflict_ids].cpu().numpy()
        del output, encoded, logits
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[{model_key}] extracted {end}/{n_rows}", flush=True)

    rows = dataset.copy()
    rows["final_clean_logit"] = final_clean
    rows["final_conflict_logit"] = final_conflict
    rows["final_margin"] = final_clean - final_conflict
    rows["pair_choice"] = np.where(rows["final_margin"].ge(0), "clean", "conflict")
    np.save(model_dir / "hidden_last_token_layers_float16.npy", hidden)
    np.save(model_dir / "candidate_margin_layers.npy", margins)
    artifact_prefix = getattr(args, "artifact_prefix", "arithmetic")
    rows.to_csv(model_dir / f"{artifact_prefix}_prompt_outcomes.csv", index=False, encoding="utf-8-sig")
    del model, tokenizer, lm_head
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return hidden, margins, rows


def delta_u_validation(
    model_key: str,
    margins: np.ndarray,
    rows: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, float | int | str], pd.DataFrame]:
    decision_layers = np.asarray(MODEL_SPECS[model_key]["decision_layers"], dtype=int)
    if decision_layers.max() >= margins.shape[1]:
        raise ValueError(f"Frozen decision window exceeds {model_key} layer count")
    clean_index = rows[rows["condition"].eq("clean")].set_index("problem_id").index
    if len(clean_index) != rows["problem_id"].nunique():
        raise RuntimeError("Each arithmetic problem must have one clean anchor")
    clean_rows = rows[rows["condition"].eq("clean")].set_index("problem_id")
    prompt_to_position = {prompt_id: index for index, prompt_id in enumerate(rows["prompt_id"])}
    clean_positions = {
        problem_id: prompt_to_position[prompt_id]
        for problem_id, prompt_id in clean_rows["prompt_id"].items()
    }
    d_matrix = np.vstack(
        [
            margins[index, decision_layers] - margins[clean_positions[int(row.problem_id)], decision_layers]
            for index, row in rows.iterrows()
        ]
    )
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_index, test_index = next(splitter.split(rows, groups=rows["problem_id"]))
    pca = PCA(n_components=1, random_state=args.seed)
    pca.fit(d_matrix[train_index])
    delta_u = pca.transform(d_matrix)[:, 0]
    train_stable = rows.iloc[train_index]["mechanism"].eq("stable").to_numpy()
    train_competition = rows.iloc[train_index]["mechanism"].eq("competition").to_numpy()
    if delta_u[train_index][train_stable].mean() < delta_u[train_index][train_competition].mean():
        delta_u *= -1
        pca.components_[0] *= -1

    classifier = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=args.seed
    )
    classifier.fit(delta_u[train_index, None], rows.iloc[train_index]["mechanism_id"])
    prediction = classifier.predict(delta_u[test_index, None])
    observed_f1 = float(
        f1_score(rows.iloc[test_index]["mechanism_id"], prediction, average="macro")
    )
    observed_accuracy = float(accuracy_score(rows.iloc[test_index]["mechanism_id"], prediction))
    null_f1 = []
    y_train = rows.iloc[train_index]["mechanism_id"].to_numpy()
    for shuffle_id in range(args.label_shuffles):
        shuffled_labels = rng.permutation(y_train)
        null_classifier = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=args.seed + shuffle_id + 1
        )
        null_classifier.fit(delta_u[train_index, None], shuffled_labels)
        null_prediction = null_classifier.predict(delta_u[test_index, None])
        null_f1.append(
            float(f1_score(rows.iloc[test_index]["mechanism_id"], null_prediction, average="macro"))
        )
    null_f1_array = np.asarray(null_f1, dtype=float)
    null_sd = float(np.std(null_f1_array, ddof=1))

    annotated = rows.copy()
    annotated["split"] = "train"
    annotated.loc[test_index, "split"] = "heldout"
    annotated["DeltaU_arith"] = delta_u
    for local_layer, layer in enumerate(decision_layers):
        annotated[f"dR_L{layer}"] = d_matrix[:, local_layer]
    summary = {
        "model": model_key,
        "n_train_prompts": int(len(train_index)),
        "n_heldout_prompts": int(len(test_index)),
        "n_train_problems": int(rows.iloc[train_index]["problem_id"].nunique()),
        "n_heldout_problems": int(rows.iloc[test_index]["problem_id"].nunique()),
        "decision_layers": ";".join(str(layer) for layer in decision_layers),
        "pca_explained_variance": float(pca.explained_variance_ratio_[0]),
        "heldout_macro_f1": observed_f1,
        "heldout_accuracy": observed_accuracy,
        "label_shuffle_f1_mean": float(np.mean(null_f1_array)),
        "label_shuffle_f1_sd": null_sd,
        "label_shuffle_f1_q95": float(np.quantile(null_f1_array, 0.95)),
        "heldout_f1_vs_shuffle_z": (
            (observed_f1 - float(np.mean(null_f1_array))) / null_sd if null_sd > 0 else np.nan
        ),
        "heldout_f1_exceeds_shuffle_q95": str(observed_f1 > float(np.quantile(null_f1_array, 0.95))),
    }
    null_rows = pd.DataFrame({"model": model_key, "shuffle_id": np.arange(args.label_shuffles), "macro_f1": null_f1})
    return annotated, summary, null_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_prefix = "arithmetic"
    set_seed(args.seed)
    dataset = build_dataset(args.problems, args.operation_family)
    dataset.to_csv(args.output_dir / "arithmetic_task_manifest.csv", index=False, encoding="utf-8-sig")
    geometry_rows = []
    delta_rows = []
    shuffle_rows = []
    label_null_rows = []
    for model_offset, model_key in enumerate(args.models):
        model_dir = args.output_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        hidden, margins, outcomes = extract_model(model_key, dataset, args, model_dir)
        geometry_rng = np.random.default_rng(args.seed + 1000 * (model_offset + 1))
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
                "n_shuffles": args.shuffles,
            }
        )
        geometry_rows.append(geometry)
        for row in null_rows:
            row.update({"model": model_key})
            shuffle_rows.append(row)

        delta_rng = np.random.default_rng(args.seed + 2000 * (model_offset + 1))
        annotated, delta_summary, label_null = delta_u_validation(
            model_key, margins, outcomes, args, delta_rng
        )
        annotated.to_csv(model_dir / "arithmetic_DeltaU_rows.csv", index=False, encoding="utf-8-sig")
        delta_rows.append(delta_summary)
        label_null_rows.append(label_null)
        del hidden, margins
        gc.collect()

    pd.DataFrame(geometry_rows).to_csv(
        args.output_dir / "external_task_geometry_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(shuffle_rows).to_csv(
        args.output_dir / "external_task_geometry_shuffles.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(delta_rows).to_csv(
        args.output_dir / "external_task_DeltaU_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(label_null_rows, ignore_index=True).to_csv(
        args.output_dir / "external_task_DeltaU_label_nulls.csv", index=False, encoding="utf-8-sig"
    )
    config = {
        "analysis": "independent arithmetic candidate-evaluation trajectory validation",
        "models": args.models,
        "n_problems": args.problems,
        "operation_family": args.operation_family,
        "conditions": [{"condition": condition, "mechanism": mechanism} for condition, mechanism in CONDITIONS],
        "candidate_order_counterbalancing": "correct word first for even problem_id and second for odd problem_id",
        "frozen_decision_windows": {
            key: list(MODEL_SPECS[key]["decision_layers"]) for key in args.models
        },
        "split_unit": "arithmetic problem_id",
        "test_size": args.test_size,
        "geometry_null": "endpoint-preserving intermediate-layer permutation",
        "n_geometry_shuffles": args.shuffles,
        "n_label_shuffles": args.label_shuffles,
        "prompt_wrapping": "raw" if args.raw_prompts else "checkpoint-native chat template",
        "seed": args.seed,
        "claim_boundary": "external task-family validation, not a universal transformer law or transfer of the numeric DeltaU axis",
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
