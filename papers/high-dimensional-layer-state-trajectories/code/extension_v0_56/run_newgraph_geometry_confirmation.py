"""Frozen new-graph geometry confirmation for Qwen2.5-1.5B and Gemma-3-12B.

The experiment is deliberately narrow. It uses 24 relation-graph groups that
are disjoint in names, relation phrases and candidate labels from the main
controlled set, then compares native all-block order against 50 shared,
endpoint-preserving intermediate-layer permutations. It is not a window scan,
task-accuracy test, or scale-law experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SEED = 2026080303
N_GRAPHS = 24
N_SHUFFLES = 50
FIXED_CONDITIONS = [
    "clean",
    "permuted",
    "redundant",
    "irrelevant",
    "weak_distractor",
    "competition_balanced",
    "direct_conflict",
    "closure_update",
    "closure_override",
    "exception_override",
]

# These names, phrases and labels do not occur in the original 96-prompt set.
ENTITY_TRIPLES = [
    ("Arden", "Brisa", "Cato"), ("Della", "Eamon", "Fiora"),
    ("Garin", "Hesta", "Ilan"), ("Jessa", "Korin", "Luma"),
    ("Marek", "Neris", "Olan"), ("Pella", "Quade", "Rumi"),
    ("Soren", "Tavi", "Uria"), ("Vaila", "Wren", "Xeran"),
    ("Ysol", "Zarek", "Amani"), ("Bram", "Celie", "Doran"),
    ("Elian", "Fara", "Galenna"), ("Hollis", "Iria", "Jarek"),
    ("Kessa", "Lorin", "Mirae"), ("Nolan", "Oria", "Perrin"),
    ("Quilla", "Ronan", "Selka"), ("Toren", "Ulla", "Varin"),
    ("Wexla", "Xavi", "Yoren"), ("Zelia", "Arlo", "Bex"),
    ("Ciran", "Dara", "Evin"), ("Fenn", "Galia", "Hiro"),
    ("Isla", "Jorin", "Kavi"), ("Leda", "Miro", "Navi"),
    ("Oriel", "Pavo", "Qira"), ("Riven", "Sela", "Tarin"),
]
RELATION_PAIRS = [
    ("routes through", "terminates at"),
    ("is catalogued beside", "is indexed as"),
    ("travels via", "is marked with"),
    ("is linked onward to", "is registered under"),
    ("passes by", "is filed at"),
    ("is traced to", "is assigned marker"),
]
LABELS = [
    "Cyan", "Magenta", "Violet", "Amber", "Ivory", "Onyx",
    "Ruby", "Jade", "Coral", "Lilac", "Mint", "Plum",
]


def make_prompt(condition: str, a: str, b: str, d: str, clean: str, conflict: str, aux: str, r1: str, r2: str) -> str:
    if condition == "clean":
        lines = ["You are given a small relation graph.", f"Fact 1: {a} {r1} {b}.", f"Fact 2: {b} {r2} {clean}."]
    elif condition == "permuted":
        lines = ["You are given a small relation graph.", f"Fact 2: {b} {r2} {clean}.", f"Fact 1: {a} {r1} {b}."]
    elif condition == "redundant":
        lines = ["You are given a small relation graph.", f"Fact 1: {a} {r1} {b}.", f"Fact 2: {b} {r2} {clean}.", f"Repeated confirmation: {a} still goes through {b}.", f"Repeated confirmation: {b} still points to {clean}."]
    elif condition == "irrelevant":
        lines = ["You are given a small relation graph.", f"Fact 1: {a} {r1} {b}.", f"Fact 2: {b} {r2} {clean}.", f"Irrelevant fact: {d} is associated with {aux}."]
    elif condition == "weak_distractor":
        lines = ["You are given a small relation graph.", f"Fact 1: {a} {r1} {b}.", f"Fact 2: {b} {r2} {clean}.", f"Weak note: an unrelated source mentions {conflict}, but does not update the graph."]
    elif condition == "competition_balanced":
        lines = ["You are given a small relation graph.", f"Fact 1: {a} {r1} {b}.", f"Fact 2: {b} {r2} {clean}.", f"Competing fact: {a} is also associated with {conflict}."]
    elif condition == "direct_conflict":
        lines = ["You are given a small relation graph.", f"Fact 1: {a} {r1} {b}.", f"Fact 2: {b} {r2} {clean}.", f"Direct conflicting fact: {a} {r2} {conflict}."]
    elif condition == "closure_update":
        lines = ["You are given a small relation graph.", f"Fact 1: {a} {r1} {b}.", f"Old record: {b} {r2} {clean}.", f"Updated record: {b} {r2} {conflict}, not {clean}."]
    elif condition == "closure_override":
        lines = ["You are given a rule system.", f"General rule: items that {r1} {b} receive label {clean}.", f"Override rule: in this case, items that {r1} {b} receive label {conflict}.", f"Fact: {a} {r1} {b}."]
    elif condition == "exception_override":
        lines = ["You are given a rule system.", f"General rule: items connected to {b} use label {clean}.", f"Exception: {a} is a special case and uses label {conflict}.", f"Fact: {a} is connected to {b}."]
    else:
        raise ValueError(f"Unfrozen condition: {condition}")
    lines += [f"Question: Which label is associated with {a}: {clean} or {conflict}?", "Answer with exactly one word:"]
    return "\n".join(lines)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_manifest(root: Path) -> pd.DataFrame:
    prompt_dir = root / "inputs" / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for graph_id, (a, b, d) in enumerate(ENTITY_TRIPLES):
        r1, r2 = RELATION_PAIRS[graph_id % len(RELATION_PAIRS)]
        clean = LABELS[(2 * graph_id) % len(LABELS)]
        conflict = LABELS[(2 * graph_id + 1) % len(LABELS)]
        aux = LABELS[(2 * graph_id + 2) % len(LABELS)]
        for condition in FIXED_CONDITIONS:
            prompt_id = f"ng{graph_id:02d}_{condition}"
            text = make_prompt(condition, a, b, d, clean, conflict, aux, r1, r2)
            prompt_file = Path("inputs") / "prompts" / f"{prompt_id}.txt"
            prompt_path = root / prompt_file
            prompt_path.write_text(text, encoding="utf-8")
            rows.append({
                "prompt_id": prompt_id,
                "graph_id": graph_id,
                "condition": condition,
                "clean_candidate": clean,
                "conflict_candidate": conflict,
                "prompt_file": str(prompt_file),
                "text": text,
            })
    frame = pd.DataFrame(rows)
    if len(frame) != N_GRAPHS * len(FIXED_CONDITIONS):
        raise RuntimeError("Frozen manifest has an unexpected prompt count")
    frame.to_csv(root / "newgraph_prompt_manifest.csv", index=False, encoding="utf-8")
    with (root / "inputs" / "gemma3_capture_manifest.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        for row in frame.itertuples(index=False):
            handle.write(f"{row.prompt_id}\t{root / row.prompt_file}\n")
    return frame


def normalize_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def geometry_metrics(states: np.ndarray) -> dict[str, np.ndarray]:
    unit = normalize_rows(states.astype(np.float32, copy=False))
    steps = np.diff(unit, axis=1)
    step_norm = np.linalg.norm(steps, axis=-1)
    chord = unit[:, -1, :] - unit[:, 0, :]
    chord_unit = normalize_rows(chord)
    alignment = np.sum(steps * chord_unit[:, None, :], axis=-1) / np.maximum(step_norm, 1e-8)
    path_length = np.sum(np.clip(1.0 - np.sum(unit[:, :-1] * unit[:, 1:], axis=-1), 0.0, None), axis=1)
    endpoint_distance = np.clip(1.0 - np.sum(unit[:, 0] * unit[:, -1], axis=-1), 0.0, None)
    detour = path_length / np.maximum(endpoint_distance, 1e-8)
    turn_cos = np.sum(steps[:, :-1] * steps[:, 1:], axis=-1) / np.maximum(step_norm[:, :-1] * step_norm[:, 1:], 1e-8)
    return {
        "chord_alignment": np.mean(alignment, axis=1),
        "detour_ratio": detour,
        "turning_curvature": np.mean(np.arccos(np.clip(turn_cos, -1.0, 1.0)), axis=1),
    }


def summarize_geometry(model_key: str, states: np.ndarray, manifest: pd.DataFrame, root: Path) -> dict[str, float]:
    if states.shape[0] != len(manifest) or states.shape[1] < 3:
        raise RuntimeError(f"Invalid {model_key} state tensor shape: {states.shape}")
    real = geometry_metrics(states)
    pd.DataFrame({"prompt_id": manifest.prompt_id, "graph_id": manifest.graph_id, **real}).to_csv(
        root / f"{model_key}_prompt_geometry.csv", index=False
    )
    rng = np.random.default_rng(SEED)
    middle = np.arange(1, states.shape[1] - 1)
    null_rows = []
    for shuffle_id in range(N_SHUFFLES):
        order = np.concatenate(([0], rng.permutation(middle), [states.shape[1] - 1]))
        metric = geometry_metrics(states[:, order, :])
        null_rows.append({"shuffle_id": shuffle_id, **{key: float(np.mean(value)) for key, value in metric.items()}})
    null = pd.DataFrame(null_rows)
    null.to_csv(root / f"{model_key}_shared_endpoint_shuffle_null.csv", index=False)
    summary: dict[str, float] = {
        "model": model_key,
        "n_prompts": int(states.shape[0]),
        "n_layers": int(states.shape[1]),
        "hidden_dim": int(states.shape[2]),
        "n_shared_endpoint_shuffles": N_SHUFFLES,
    }
    for key, values in real.items():
        observed = float(np.mean(values))
        null_values = null[key].to_numpy(float)
        direction = 1.0 if key == "chord_alignment" else -1.0
        empirical_p = (1 + int(np.sum(direction * null_values >= direction * observed))) / (N_SHUFFLES + 1)
        summary[f"real_{key}"] = observed
        summary[f"null_{key}_mean"] = float(np.mean(null_values))
        summary[f"null_{key}_sd"] = float(np.std(null_values, ddof=1))
        summary[f"null_{key}_q95"] = float(np.quantile(null_values, 0.95))
        summary[f"real_minus_null_{key}"] = observed - float(np.mean(null_values))
        summary[f"null_standardized_{key}"] = direction * (observed - float(np.mean(null_values))) / max(float(np.std(null_values, ddof=1)), 1e-8)
        summary[f"empirical_p_{key}"] = empirical_p
    return summary


def extract_qwen(manifest: pd.DataFrame, root: Path, model_path: str, batch_size: int) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda"
    ).eval()
    all_states: list[np.ndarray] = []
    for start in range(0, len(manifest), batch_size):
        text_batch = manifest.iloc[start : start + batch_size].text.tolist()
        encoded = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=False).to("cuda")
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        final_indices = encoded.attention_mask.sum(dim=1) - 1
        layers = [hidden[torch.arange(hidden.shape[0], device=hidden.device), final_indices].float().cpu().numpy() for hidden in outputs.hidden_states[1:]]
        all_states.append(np.stack(layers, axis=1))
        del outputs, encoded
        torch.cuda.empty_cache()
    states = np.concatenate(all_states, axis=0).astype(np.float32)
    np.save(root / "qwen_all_block_states.npy", states)
    del model
    torch.cuda.empty_cache()
    return states


def extract_gemma3(manifest: pd.DataFrame, root: Path, executable: Path, runtime_dir: Path, model_path: Path) -> np.ndarray:
    capture_dir = root / "gemma3_captures"
    capture_dir.mkdir(parents=True, exist_ok=True)
    command = [str(executable), str(runtime_dir), str(model_path), "--batch", str(root / "inputs" / "gemma3_capture_manifest.tsv"), str(capture_dir)]
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    (root / "gemma3_capture_runtime.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Gemma3 capture failed with code {result.returncode}; see gemma3_capture_runtime.log")
    states = []
    for prompt_id in manifest.prompt_id:
        metadata = json.loads((capture_dir / f"{prompt_id}.json").read_text(encoding="utf-8"))
        shape = (int(metadata["n_captured_layers"]), int(metadata["n_embd"]))
        array = np.fromfile(capture_dir / f"{prompt_id}.f32", dtype=np.float32)
        if array.size != shape[0] * shape[1] or shape[0] != int(metadata["n_model_layers"]):
            raise RuntimeError(f"Incomplete Gemma3 capture for {prompt_id}")
        states.append(array.reshape(shape))
    stacked = np.stack(states, axis=0)
    np.save(root / "gemma3_12b_all_block_states.npy", stacked)
    return stacked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--qwen-model", required=True, help="Local Qwen checkpoint path.")
    parser.add_argument("--qwen-model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--gemma3-model", type=Path, required=True, help="Local Gemma3 GGUF file.")
    parser.add_argument("--gemma3-model-id", default="Gemma-3-12B-it-QAT-Q4_0-GGUF")
    parser.add_argument("--gemma3-runtime", type=Path, required=True, help="Local llama.cpp runtime directory.")
    parser.add_argument("--gemma3-capture", type=Path, required=True)
    parser.add_argument("--capture-source-revision", default="llama.cpp 1a064ab0921238c1daa397d6f4a900ef33884de2")
    parser.add_argument("--qwen-batch-size", type=int, default=4)
    parser.add_argument("--reuse-extractions", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen confirmation run")
    # Preserve an ASCII subst drive exactly as supplied. Calling resolve() here
    # expands it back to the Unicode host path, which native narrow-path tools
    # cannot reliably open on this Windows runtime.
    root = args.output_root.absolute()
    root.mkdir(parents=True, exist_ok=True)
    manifest = create_manifest(root)
    prompt_content_hash = hashlib.sha256("\n".join(manifest.text.tolist()).encode("utf-8")).hexdigest()
    metadata = {
        "experiment": "v0_56_newgraph_geometry_confirmation",
        "status": "confirmatory_checkpoint_specific_geometry_only",
        "seed": SEED,
        "n_graphs": N_GRAPHS,
        "fixed_conditions": FIXED_CONDITIONS,
        "n_prompts": len(manifest),
        "n_shared_endpoint_shuffles": N_SHUFFLES,
        "prompt_wrapping": "raw fixed prompt string in both runtimes",
        "qwen_model_id": args.qwen_model_id,
        "gemma3_model_id": args.gemma3_model_id,
        "gemma3_model_sha256": sha256(args.gemma3_model),
        "gemma3_quantization": "GGUF QAT Q4_0",
        "capture_source_revision": args.capture_source_revision,
        "prompt_content_sha256": prompt_content_hash,
        "claim_boundary": "larger independent checkpoint confirmation; not a pure scale law or quantization-invariant result",
    }
    (root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    qwen_path = root / "qwen_all_block_states.npy"
    gemma_path = root / "gemma3_12b_all_block_states.npy"
    if args.reuse_extractions:
        qwen_states = np.load(qwen_path)
        gemma_states = np.load(gemma_path)
    else:
        qwen_states = extract_qwen(manifest, root, args.qwen_model, args.qwen_batch_size)
        gemma_states = extract_gemma3(manifest, root, args.gemma3_capture, args.gemma3_runtime, args.gemma3_model)
    summary = pd.DataFrame([
        summarize_geometry("Qwen2.5-1.5B-Instruct", qwen_states, manifest, root),
        summarize_geometry("Gemma-3-12B-it-QAT-Q4_0", gemma_states, manifest, root),
    ])
    summary.to_csv(root / "newgraph_geometry_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
