"""Standard four-choice ARC-Challenge trajectory-geometry validation."""

from __future__ import annotations

import argparse
import gc
import json
import random
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from external_task_shared import MODEL_SPECS
from trajectory_geometry import (
    geometry_effect_summary,
    summarize_against_endpoint_shuffles,
)


DATASET = "allenai/ai2_arc"
CONFIG = "ARC-Challenge"
SPLIT = "validation"
VIEWER_BASE = "https://datasets-server.huggingface.co"
CHOICE_LABELS = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("arc_four_choice_geometry"))
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--items", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--shuffles", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--raw-prompts", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def viewer_json(endpoint: str, **params) -> dict:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{VIEWER_BASE}/{endpoint}?{query}",
        headers={"User-Agent": "Paper1-trajectory-audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def fetch_arc_items(n_items: int) -> pd.DataFrame:
    selected: list[dict] = []
    offset = 0
    while len(selected) < n_items:
        payload = viewer_json(
            "rows", dataset=DATASET, config=CONFIG, split=SPLIT, offset=offset, length=100
        )
        rows = payload.get("rows", [])
        if not rows:
            break
        for wrapped in rows:
            row = wrapped["row"]
            choices = row.get("choices", {})
            labels = tuple(choices.get("label", []))
            texts = tuple(choices.get("text", []))
            answer = str(row.get("answerKey", ""))
            if labels != CHOICE_LABELS or len(texts) != 4 or answer not in CHOICE_LABELS:
                continue
            selected.append(
                {
                    "source_row": int(wrapped["row_idx"]),
                    "item_id": str(row["id"]),
                    "question": str(row["question"]),
                    "choice_A": texts[0],
                    "choice_B": texts[1],
                    "choice_C": texts[2],
                    "choice_D": texts[3],
                    "answer_key": answer,
                }
            )
            if len(selected) == n_items:
                break
        offset += len(rows)
        if offset >= int(payload.get("num_rows_total", offset)):
            break
    if len(selected) != n_items:
        raise RuntimeError(f"Requested {n_items} valid A-D items, found {len(selected)}")
    return pd.DataFrame(selected)


def format_prompt(row: pd.Series) -> str:
    return "\n".join(
        [
            "Answer this four-choice science question.",
            str(row["question"]),
            f"A. {row['choice_A']}",
            f"B. {row['choice_B']}",
            f"C. {row['choice_C']}",
            f"D. {row['choice_D']}",
            "Answer with exactly one letter (A, B, C or D):",
        ]
    )


def continuation_id(tokenizer, label: str) -> tuple[int, list[int]]:
    ids = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise RuntimeError(f"ARC label {label!r} is not one continuation token: {ids}")
    return int(ids[0]), list(ids)


def load_model(model_key: str):
    path = MODEL_SPECS[model_key]["path"]
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return model, tokenizer


def extract_model(
    model_key: str, items: pd.DataFrame, args: argparse.Namespace, output_dir: Path
) -> tuple[np.ndarray, pd.DataFrame]:
    model, tokenizer = load_model(model_key)
    token_ids: dict[str, int] = {}
    token_rows = []
    for label in CHOICE_LABELS:
        token_id, ids = continuation_id(tokenizer, label)
        token_ids[label] = token_id
        token_rows.append({"label": label, "token_id": token_id, "token_ids": str(ids)})
    pd.DataFrame(token_rows).to_csv(output_dir / "tokenization_audit.csv", index=False)

    prompts = [format_prompt(row) for _, row in items.iterrows()]
    if not args.raw_prompts and hasattr(tokenizer, "apply_chat_template"):
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]

    n_items = len(items)
    n_layers = int(model.config.num_hidden_layers)
    hidden_size = int(model.config.hidden_size)
    hidden = np.zeros((n_items, n_layers, hidden_size), dtype=np.float16)
    option_logits = np.zeros((n_items, 4), dtype=np.float32)
    device = next(model.parameters()).device

    for start in range(0, n_items, args.batch_size):
        end = min(n_items, start + args.batch_size)
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
            state = output.hidden_states[layer + 1][row_index, position, :]
            hidden[start:end, layer, :] = state.detach().float().cpu().numpy().astype(np.float16)
        logits = output.logits[row_index, position, :].detach().float()
        for option_index, label in enumerate(CHOICE_LABELS):
            option_logits[start:end, option_index] = logits[:, token_ids[label]].cpu().numpy()
        del output, encoded, logits
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[{model_key}] extracted {end}/{n_items}", flush=True)

    outcome = items.copy()
    prediction_index = option_logits.argmax(axis=1)
    outcome["predicted_label"] = [CHOICE_LABELS[index] for index in prediction_index]
    outcome["correct"] = outcome["predicted_label"].eq(outcome["answer_key"])
    answer_index = outcome["answer_key"].map({label: i for i, label in enumerate(CHOICE_LABELS)}).to_numpy()
    correct_logits = option_logits[np.arange(n_items), answer_index]
    masked = option_logits.copy()
    masked[np.arange(n_items), answer_index] = -np.inf
    outcome["correct_minus_best_distractor_logit"] = correct_logits - masked.max(axis=1)
    for option_index, label in enumerate(CHOICE_LABELS):
        outcome[f"logit_{label}"] = option_logits[:, option_index]

    np.save(output_dir / "hidden_last_token_layers_float16.npy", hidden)
    outcome.to_csv(output_dir / "arc_prompt_outcomes.csv", index=False, encoding="utf-8-sig")
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return hidden, outcome


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    items = fetch_arc_items(args.items)
    items.to_csv(args.output_dir / "arc_challenge_manifest.csv", index=False, encoding="utf-8-sig")

    summaries = []
    null_rows = []
    for model_offset, model_key in enumerate(args.models):
        model_dir = args.output_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        hidden, outcomes = extract_model(model_key, items, args, model_dir)
        rng = np.random.default_rng(args.seed + 1000 * (model_offset + 1))
        real_means, model_nulls = summarize_against_endpoint_shuffles(
            hidden.astype(np.float32), args.shuffles, rng
        )
        summary = geometry_effect_summary(real_means, model_nulls)
        summary.update(
            {
                "model": model_key,
                "n_items": len(items),
                "n_layers": hidden.shape[1],
                "hidden_dim": hidden.shape[2],
                "four_choice_accuracy": float(outcomes["correct"].mean()),
                "mean_correct_margin": float(outcomes["correct_minus_best_distractor_logit"].mean()),
                "n_shuffles": args.shuffles,
            }
        )
        summaries.append(summary)
        for row in model_nulls:
            row["model"] = model_key
            null_rows.append(row)
        del hidden
        gc.collect()

    pd.DataFrame(summaries).to_csv(
        args.output_dir / "arc_geometry_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(null_rows).to_csv(
        args.output_dir / "arc_geometry_shuffles.csv", index=False, encoding="utf-8-sig"
    )
    config = {
        "analysis": "standard four-choice ARC-Challenge trajectory geometry",
        "dataset": DATASET,
        "dataset_config": CONFIG,
        "split": SPLIT,
        "selection": "first 96 rows with exactly A-D labels, in Dataset Viewer row order",
        "dataset_viewer": VIEWER_BASE,
        "models": args.models,
        "n_items": args.items,
        "n_endpoint_preserving_shuffles": args.shuffles,
        "prompt_wrapping": "raw" if args.raw_prompts else "checkpoint-native chat template",
        "seed": args.seed,
        "claim_boundary": (
            "transfer of ordered hidden-state geometry to a standard four-choice benchmark; "
            "not transfer of the relation-graph DeltaU coordinate, mechanism labels or intervention policy"
        ),
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
