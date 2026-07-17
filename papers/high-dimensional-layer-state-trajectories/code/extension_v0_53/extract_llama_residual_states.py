"""Re-extract Llama trajectories with consistent pre-final-norm endpoints."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("LLAMA_MODEL_PATH", "meta-llama/Llama-3.2-1B-Instruct"))
RELATION_CACHE = Path(os.environ.get("LLAMA_RELATION_STATE_PATH", "inputs/llama_relation_states.npy"))
TASKS = (
    "relation_graph",
    "lexical_category",
    "arithmetic_addition",
    "arc_challenge",
    "lexical_disjoint",
    "addition_disjoint",
    "arithmetic_mixed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "inputs/llama_residual_states",
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    return parser.parse_args()


def arc_prompt(row: pd.Series) -> str:
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


def task_frame(task: str) -> tuple[pd.DataFrame, Path, bool]:
    if task == "relation_graph":
        frame = pd.read_csv(
            ROOT / "direct_hidden_geometry_v0_1/controlled_subset_manifest.csv",
            encoding="utf-8-sig",
        )
        frame = frame[["prompt_id", "prompt", "source_row"]].copy()
        return frame, RELATION_CACHE, False
    if task in {"lexical_category", "lexical_disjoint"}:
        root = ROOT / "inputs/lexical_task"
        frame = pd.read_csv(root / "lexical_task_manifest.csv", encoding="utf-8-sig")
        start = 0 if task == "lexical_category" else 96
        frame = frame.iloc[start : start + 96].copy()
        frame["source_row"] = np.arange(start, start + len(frame))
        return frame[["prompt_id", "prompt", "source_row"]], root / "llama/hidden_last_token_layers_float16.npy", True
    if task in {"arithmetic_addition", "addition_disjoint"}:
        root = ROOT / "inputs/addition_task"
        frame = pd.read_csv(root / "arithmetic_task_manifest.csv", encoding="utf-8-sig")
        start = 0 if task == "arithmetic_addition" else 96
        frame = frame.iloc[start : start + 96].copy()
        frame["source_row"] = np.arange(start, start + len(frame))
        return frame[["prompt_id", "prompt", "source_row"]], root / "llama/hidden_last_token_layers_float16.npy", True
    if task == "arithmetic_mixed":
        root = ROOT / "inputs/mixed_arithmetic_task"
        frame = pd.read_csv(root / "arithmetic_task_manifest.csv", encoding="utf-8-sig").iloc[:96].copy()
        frame["source_row"] = np.arange(len(frame))
        return frame[["prompt_id", "prompt", "source_row"]], root / "llama/hidden_last_token_layers_float16.npy", True
    if task == "arc_challenge":
        root = ROOT / "arc_multichoice_geometry_v0_1"
        frame = pd.read_csv(root / "arc_challenge_manifest.csv", encoding="utf-8-sig").iloc[:96].copy()
        frame["prompt"] = [arc_prompt(row) for _, row in frame.iterrows()]
        frame["prompt_id"] = frame["item_id"].astype(str)
        frame["dataset_source_row"] = frame["source_row"]
        frame["source_row"] = np.arange(len(frame))
        return frame[["prompt_id", "prompt", "source_row"]], root / "llama/hidden_last_token_layers_float16.npy", True
    raise ValueError(task)


def flat_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_flat = left.reshape(len(left), -1).astype(np.float64)
    right_flat = right.reshape(len(right), -1).astype(np.float64)
    numerator = np.sum(left_flat * right_flat, axis=1)
    denominator = np.linalg.norm(left_flat, axis=1) * np.linalg.norm(right_flat, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def load_model(path: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.model_path)
    layers = model.model.layers
    final_norm = model.model.norm
    n_layers = int(model.config.num_hidden_layers)
    hidden_size = int(model.config.hidden_size)
    if len(layers) != n_layers:
        raise ValueError("Decoder layer count mismatch")

    audit_rows = []
    for task in args.tasks:
        frame, cache_path, use_chat = task_frame(task)
        prompts = frame["prompt"].astype(str).tolist()
        if use_chat:
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]
        task_dir = args.output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        n = len(frame)
        saved = {
            "cached": task_dir / "cached_mixed_states_float16.npy",
            "mixed": task_dir / "reextracted_mixed_states_float16.npy",
            "consistent": task_dir / "consistent_block_states_float16.npy",
            "common": task_dir / "common_rmsnorm_states_float16.npy",
        }
        reused = all(path.exists() for path in saved.values())
        if reused:
            cached = np.asarray(np.load(saved["cached"], mmap_mode="r"), dtype=np.float16)
            mixed = np.asarray(np.load(saved["mixed"], mmap_mode="r"), dtype=np.float16)
            consistent = np.asarray(np.load(saved["consistent"], mmap_mode="r"), dtype=np.float16)
            common_rms = np.asarray(np.load(saved["common"], mmap_mode="r"), dtype=np.float16)
            print(f"[{task}] reused completed extraction", flush=True)
        else:
            mixed = np.zeros((n, n_layers, hidden_size), dtype=np.float16)
            consistent = np.zeros_like(mixed)
            common_rms = np.zeros_like(mixed)
            for start in range(0, n, args.batch_size):
                end = min(n, start + args.batch_size)
                encoded = tokenizer(
                    prompts[start:end],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                ).to(model.device)
                captured: dict[int, torch.Tensor] = {}
                handles = []
                for layer_index, layer in enumerate(layers):
                    def capture(_module, _inputs, output, index=layer_index):
                        value = output[0] if isinstance(output, tuple) else output
                        captured[index] = value[:, -1, :].detach()

                    handles.append(layer.register_forward_hook(capture))
                try:
                    with torch.no_grad():
                        output = model(
                            **encoded,
                            output_hidden_states=True,
                            use_cache=False,
                            return_dict=True,
                        )
                finally:
                    for handle in handles:
                        handle.remove()
                if set(captured) != set(range(n_layers)):
                    raise RuntimeError(f"Missing hook outputs for {task} batch {start}:{end}")
                row_index = torch.arange(end - start, device=model.device)
                position = torch.full(
                    (end - start,),
                    encoded["attention_mask"].shape[1] - 1,
                    dtype=torch.long,
                    device=model.device,
                )
                mixed_batch = torch.stack(
                    [output.hidden_states[layer + 1][row_index, position] for layer in range(n_layers)],
                    dim=1,
                )
                consistent_batch = torch.stack([captured[layer] for layer in range(n_layers)], dim=1)
                with torch.no_grad():
                    common_batch = torch.stack(
                        [final_norm(captured[layer]) for layer in range(n_layers)], dim=1
                    )
                mixed[start:end] = mixed_batch.float().cpu().numpy().astype(np.float16)
                consistent[start:end] = consistent_batch.float().cpu().numpy().astype(np.float16)
                common_rms[start:end] = common_batch.float().cpu().numpy().astype(np.float16)
                del output, encoded, mixed_batch, consistent_batch, common_batch, captured
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[{task}] extracted {end}/{n}", flush=True)

        cached_all = np.load(cache_path, mmap_mode="r")
        source_rows = frame["source_row"].to_numpy(int)
        cached = np.asarray(cached_all[source_rows], dtype=np.float16)
        if cached.shape != mixed.shape:
            raise ValueError(f"Cached shape {cached.shape} != extracted {mixed.shape}")
        cosine = flat_cosine(cached, mixed)
        cached_norm = np.linalg.norm(cached.astype(np.float32), axis=-1)
        mixed_norm = np.linalg.norm(mixed.astype(np.float32), axis=-1)
        consistent_norm = np.linalg.norm(consistent.astype(np.float32), axis=-1)
        audit_rows.append(
            {
                "task": task,
                "n_prompts": n,
                "cache_path": str(cache_path.resolve()),
                "mean_flat_cosine_cached_vs_reextracted": float(cosine.mean()),
                "min_flat_cosine_cached_vs_reextracted": float(cosine.min()),
                "mean_absolute_error_cached_vs_reextracted": float(
                    np.mean(np.abs(cached.astype(np.float32) - mixed.astype(np.float32)))
                ),
                "cached_final_to_penultimate_norm_ratio_median": float(
                    np.median(cached_norm[:, -1] / np.maximum(cached_norm[:, -2], 1e-8))
                ),
                "reextracted_final_to_penultimate_norm_ratio_median": float(
                    np.median(mixed_norm[:, -1] / np.maximum(mixed_norm[:, -2], 1e-8))
                ),
                "consistent_final_to_penultimate_norm_ratio_median": float(
                    np.median(
                        consistent_norm[:, -1] / np.maximum(consistent_norm[:, -2], 1e-8)
                    )
                ),
            }
        )
        if not reused:
            np.save(saved["cached"], cached)
            np.save(saved["mixed"], mixed)
            np.save(saved["consistent"], consistent)
            np.save(saved["common"], common_rms)
        frame[["prompt_id", "source_row"]].to_csv(
            task_dir / "prompt_manifest.csv", index=False, encoding="utf-8-sig"
        )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(args.output_dir / "extraction_coordinate_audit.csv", index=False)
    config = {
        "model_path": str(args.model_path.resolve()),
        "tasks": args.tasks,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "coordinate_definition": {
            "mixed": "output.hidden_states[1:]",
            "consistent": "decoder block forward-hook outputs before final RMSNorm",
            "common_rmsnorm": "the same model.model.norm applied to each captured block output",
        },
    }
    (args.output_dir / "extraction_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
