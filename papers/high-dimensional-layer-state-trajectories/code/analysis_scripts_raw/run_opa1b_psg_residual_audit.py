r"""Run OPA-1B PSG Residual Projection Control Audit.

This recovery runner uses existing OPA-1A-lite hidden-state artifacts as the
hidden-state input and recomputes the OPA-1B projection-control metrics with
realW, rowpermW, and gaussianR controls. It writes only to:

outputs/opa1b_psg_residual_audit_output

The run is evidence-bounded: recovered hidden states are recorded in the input
inventory, and no theory-summary numbers are used as experimental outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:
    AutoModelForCausalLM = None
    AutoTokenizer = None


OUTPUT_DIR = Path("outputs/opa1b_psg_residual_audit_output")
SOURCE_DIR = Path("inputs/opa1a_lite_crossmodel_outputs")
PROMPT_SOURCE = SOURCE_DIR / "opa1a_lite_input_used.csv"
RECOVERY_DIR = Path("outputs/nature_missing_experiment_recovery_output")

RANDOM_SEED = 42
TOPK = 50
MAX_PROMPTS = 96
N_REPEATS = 5
DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
LOCAL_FILES_ONLY = True


@dataclass(frozen=True)
class ModelCfg:
    key: str
    path: str
    hidden_path: Path
    obs_layers: list[int]


MODELS = [
    ModelCfg(
        key="qwen",
        path="<optional-local-qwen-checkpoint-path>",
        hidden_path=SOURCE_DIR / "qwen_hidden_last_token_layers.npy",
        obs_layers=list(range(7, 26)),
    ),
    ModelCfg(
        key="llama",
        path="<optional-local-llama-checkpoint-path>",
        hidden_path=SOURCE_DIR / "llama_hidden_last_token_layers.npy",
        obs_layers=list(range(4, 15)),
    ),
    ModelCfg(
        key="gemma",
        path="<optional-local-gemma-checkpoint-path>",
        hidden_path=SOURCE_DIR / "gemma_hidden_last_token_layers.npy",
        obs_layers=list(range(6, 24)),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR, help="Directory containing OPA-1A hidden-state arrays.")
    parser.add_argument("--prompt-source", type=Path, default=None, help="Prompt CSV; defaults to SOURCE_DIR/opa1a_lite_input_used.csv.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for OPA-1B audit outputs.")
    parser.add_argument("--recovery-dir", type=Path, default=RECOVERY_DIR, help="Optional recovery-output directory recorded in inventories.")
    parser.add_argument("--qwen-path", default=MODELS[0].path, help="Local Qwen checkpoint path or HF cache snapshot path.")
    parser.add_argument("--llama-path", default=MODELS[1].path, help="Local Llama checkpoint path or HF cache snapshot path.")
    parser.add_argument("--gemma-path", default=MODELS[2].path, help="Local Gemma checkpoint path or HF cache snapshot path.")
    parser.add_argument("--device", default=DEVICE, choices=["cpu", "cuda"], help="Device used for projection computation.")
    parser.add_argument("--allow-remote-model-files", action="store_true", help="Allow transformers to fetch missing files.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--max-prompts", type=int, default=MAX_PROMPTS)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--check-inputs-only", action="store_true", help="Write an input-availability report without loading models.")
    return parser.parse_args()


def configure_from_args(args: argparse.Namespace) -> None:
    global OUTPUT_DIR, SOURCE_DIR, PROMPT_SOURCE, RECOVERY_DIR
    global RANDOM_SEED, TOPK, MAX_PROMPTS, N_REPEATS, DEVICE, LOCAL_FILES_ONLY, MODELS

    OUTPUT_DIR = args.output_dir
    SOURCE_DIR = args.source_dir
    PROMPT_SOURCE = args.prompt_source if args.prompt_source is not None else SOURCE_DIR / "opa1a_lite_input_used.csv"
    RECOVERY_DIR = args.recovery_dir
    RANDOM_SEED = args.random_seed
    TOPK = args.topk
    MAX_PROMPTS = args.max_prompts
    N_REPEATS = args.n_repeats
    DEVICE = args.device
    LOCAL_FILES_ONLY = not args.allow_remote_model_files
    if DEVICE == "cuda" and (torch is None or not torch.cuda.is_available()):
        DEVICE = "cpu"

    MODELS = [
        ModelCfg("qwen", args.qwen_path, SOURCE_DIR / "qwen_hidden_last_token_layers.npy", list(range(7, 26))),
        ModelCfg("llama", args.llama_path, SOURCE_DIR / "llama_hidden_last_token_layers.npy", list(range(4, 15))),
        ModelCfg("gemma", args.gemma_path, SOURCE_DIR / "gemma_hidden_last_token_layers.npy", list(range(6, 24))),
    ]


def path_status(path: Path | str) -> dict[str, object]:
    raw = str(path)
    if raw.startswith("<") and raw.endswith(">"):
        return {"path": raw, "exists": False, "status": "PLACEHOLDER"}
    p = Path(path)
    return {"path": raw.replace("\\", "/"), "exists": p.exists(), "status": "FOUND" if p.exists() else "MISSING"}


def write_input_check_report() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(SOURCE_DIR).replace("\\", "/"),
        "prompt_source": path_status(PROMPT_SOURCE),
        "recovery_dir": str(RECOVERY_DIR).replace("\\", "/"),
        "output_dir": str(OUTPUT_DIR).replace("\\", "/"),
        "device": DEVICE,
        "local_files_only": LOCAL_FILES_ONLY,
        "parameters": {
            "random_seed": RANDOM_SEED,
            "topk": TOPK,
            "max_prompts": MAX_PROMPTS,
            "n_repeats": N_REPEATS,
        },
        "models": [
            {
                "model_key": cfg.key,
                "model_path": path_status(cfg.path),
                "hidden_path": path_status(cfg.hidden_path),
                "obs_layers": cfg.obs_layers,
            }
            for cfg in MODELS
        ],
    }
    write_json(OUTPUT_DIR / "opa1b_input_check_report.json", report)
    return report


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_simple(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    return ranks


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    return safe_corr(rankdata_simple(x[mask]), rankdata_simple(y[mask]))


def cosine_distance_vector_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    sim = x @ x.T
    idx = np.triu_indices(x.shape[0], k=1)
    return (1.0 - sim)[idx]


def pairwise_preservation(hidden: np.ndarray, centers: np.ndarray) -> tuple[float, float]:
    return safe_corr(cosine_distance_vector_np(hidden), cosine_distance_vector_np(centers)), safe_spearman(
        cosine_distance_vector_np(hidden), cosine_distance_vector_np(centers)
    )


def load_weight_matrix(model_path: str) -> tuple[torch.Tensor, dict[str, object]]:
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("torch and transformers are required for model-weight loading; use --check-inputs-only for preflight.")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=LOCAL_FILES_ONLY, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=LOCAL_FILES_ONLY,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    model.eval()
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        w = model.lm_head.weight.detach().float().cpu()
    else:
        w = model.get_input_embeddings().weight.detach().float().cpu()
    info = {
        "model_path": model_path,
        "tokenizer_vocab_size": len(tokenizer),
        "weight_shape": list(w.shape),
        "device_used_for_loading": str(next(model.parameters()).device),
    }
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return w, info


def make_projection(w: torch.Tensor, condition: str, repeat: int, generator: torch.Generator) -> torch.Tensor:
    if condition == "realW":
        return w
    if condition == "rowpermW":
        perm = torch.randperm(w.shape[0], generator=generator, device=w.device)
        return w[perm]
    if condition == "gaussianR":
        mu = w.mean(dim=0, keepdim=True)
        std = w.std(dim=0, keepdim=True).clamp_min(1e-6)
        r = torch.randn(w.shape, generator=generator, device=w.device, dtype=w.dtype) * std + mu
        target_norm = w.norm(dim=1, keepdim=True).clamp_min(1e-6)
        r_norm = r.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return r / r_norm * target_norm
    raise ValueError(condition)


def topk_center_metrics(h_np: np.ndarray, w_proj: torch.Tensor, topk: int) -> dict[str, object]:
    h = torch.from_numpy(h_np.astype("float32")).to(w_proj.device)
    h = h.to(dtype=w_proj.dtype)
    logits = h @ w_proj.T
    k = min(topk, logits.shape[1])
    vals, idx = torch.topk(logits, k=k, dim=1, largest=True, sorted=True)
    selected = w_proj[idx]
    centers = selected.mean(dim=1)
    centers_np = centers.float().cpu().numpy()
    pearson, spearman = pairwise_preservation(h_np, centers_np)
    c_norm = centers / centers.norm(dim=1, keepdim=True).clamp_min(1e-8)
    s_norm = selected / selected.norm(dim=2, keepdim=True).clamp_min(1e-8)
    spread = (1.0 - (s_norm * c_norm[:, None, :]).sum(dim=2)).mean().float().item()
    probs = torch.softmax(vals.float(), dim=1)
    entropy = (-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)).mean().item()
    return {
        "pearson_preservation": pearson,
        "spearman_preservation": spearman,
        "spread_mean": float(spread),
        "entropy_mean": float(entropy),
    }


def valid_layers(layers: Iterable[int], n_layers: int) -> list[int]:
    return [int(x) for x in layers if 0 <= int(x) < n_layers]


def verdict_from_summary(summary_rows: list[dict[str, object]], prompt_count: int, model_count: int) -> dict[str, object]:
    if prompt_count < 24 or model_count < 2 or N_REPEATS < 2:
        verdict = "INCONCLUSIVE"
    else:
        by_model = {r["model_key"]: r for r in summary_rows}
        pass_models = 0
        fail_models = 0
        for r in by_model.values():
            real_minus_rowperm = abs(float(r["real_minus_rowperm"]))
            gaussian_minus_real = float(r["gaussian_minus_real"])
            real_minus_gaussian = float(r["real_minus_gaussian"])
            if real_minus_rowperm <= 0.05 or gaussian_minus_real >= 0:
                pass_models += 1
            if real_minus_rowperm > 0.10 and real_minus_gaussian > 0.10:
                fail_models += 1
        if fail_models >= 2:
            verdict = "FAIL_REALW_RESIDUAL_SUPPORTED"
        elif pass_models == model_count:
            verdict = "PASS_STRONG"
        elif pass_models >= math.ceil(model_count / 2):
            verdict = "PASS_PSG_RESIDUAL_NOT_SUPPORTED"
        else:
            verdict = "INCONCLUSIVE"
    return {
        "verdict": verdict,
        "prompt_count": prompt_count,
        "model_count": model_count,
        "projection_repeats": N_REPEATS,
        "rules": {
            "PASS_PSG_RESIDUAL_NOT_SUPPORTED": "|real-rowperm| <= 0.05 or gaussianR >= realW in majority of models.",
            "PASS_STRONG": "All three models show realW approximately rowpermW and gaussianR comparable or stronger.",
            "INCONCLUSIVE": "prompt count < 24, model count < 2, or projection repeats < 2.",
            "FAIL_REALW_RESIDUAL_SUPPORTED": "realW consistently stronger than rowpermW and gaussianR by > 0.10.",
        },
    }


def main() -> None:
    args = parse_args()
    configure_from_args(args)
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_check = write_input_check_report()
    if args.check_inputs_only:
        print(json.dumps(input_check, ensure_ascii=False, indent=2, default=str))
        return

    if not PROMPT_SOURCE.exists():
        write_csv(OUTPUT_DIR / "opa1b_missing_or_blocked_files.csv", [
            {"item": "prompt_source", "status": "MISSING_PROMPT_SOURCE", "path": str(PROMPT_SOURCE)}
        ], ["item", "status", "path"])
        raise FileNotFoundError(f"Prompt source not found: {PROMPT_SOURCE}")

    prompt_df = pd.read_csv(PROMPT_SOURCE).head(MAX_PROMPTS).copy()
    prompt_df.to_csv(OUTPUT_DIR / "opa1b_prompt_inventory.csv", index=False, encoding="utf-8-sig")

    input_inventory = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "source_hidden_state_dir": str(SOURCE_DIR),
        "prompt_source": str(PROMPT_SOURCE),
        "max_prompts_used": MAX_PROMPTS,
        "n_repeats": N_REPEATS,
        "topk": TOPK,
        "device": DEVICE,
        "evidence_boundary": "Internal Project Evidence; recovered hidden-state artifacts, newly recomputed projection-control metrics.",
    }
    write_json(OUTPUT_DIR / "opa1b_input_inventory.json", input_inventory)

    projection_rows = []
    for condition in ["realW", "rowpermW", "gaussianR"]:
        repeats = [0] if condition == "realW" else list(range(N_REPEATS))
        for repeat in repeats:
            projection_rows.append({
                "condition": condition,
                "repeat": repeat,
                "definition": {
                    "realW": "Original lm_head/output embedding matrix.",
                    "rowpermW": "Vocabulary rows randomly permuted; row norms and vector distribution preserved.",
                    "gaussianR": "Random Gaussian matrix matched to realW per-dimension scale and row norms.",
                }[condition],
            })
    write_csv(OUTPUT_DIR / "opa1b_projection_condition_inventory.csv", projection_rows)

    model_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []

    for cfg in MODELS:
        if not cfg.hidden_path.exists():
            missing_rows.append({"item": cfg.key, "status": "MISSING_HIDDEN_STATES", "path": str(cfg.hidden_path)})
            continue
        if not Path(cfg.path).exists():
            missing_rows.append({"item": cfg.key, "status": "MISSING_MODEL_PATH", "path": cfg.path})
            continue
        hidden_full = np.load(cfg.hidden_path, mmap_mode="r")
        n_prompts = min(MAX_PROMPTS, hidden_full.shape[0])
        layers = valid_layers(cfg.obs_layers, hidden_full.shape[1])
        w_cpu, info = load_weight_matrix(cfg.path)
        w = w_cpu.to(DEVICE, dtype=torch.float16 if DEVICE == "cuda" else torch.float32)
        model_rows.append({
            "model_key": cfg.key,
            "model_path": cfg.path,
            "hidden_path": str(cfg.hidden_path),
            "hidden_shape": list(hidden_full.shape),
            "used_prompt_count": n_prompts,
            "obs_layers": ";".join(map(str, layers)),
            "weight_shape": info["weight_shape"],
        })

        for layer in layers:
            h_np = np.asarray(hidden_full[:n_prompts, layer, :], dtype=np.float32)
            for condition in ["realW", "rowpermW", "gaussianR"]:
                repeats = [0] if condition == "realW" else list(range(N_REPEATS))
                for repeat in repeats:
                    gen = torch.Generator(device=DEVICE)
                    gen.manual_seed(RANDOM_SEED + repeat + layer * 100 + (0 if condition == "realW" else 10000 if condition == "rowpermW" else 20000))
                    w_proj = make_projection(w, condition, repeat, gen)
                    m = topk_center_metrics(h_np, w_proj, TOPK)
                    metric_rows.append({
                        "model_key": cfg.key,
                        "layer": layer,
                        "projection_condition": condition,
                        "repeat": repeat,
                        **m,
                    })
                    if condition != "realW":
                        del w_proj
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        del w
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(OUTPUT_DIR / "opa1b_model_inventory.csv", model_rows)
    write_csv(OUTPUT_DIR / "opa1b_layerwise_projection_control_metrics.csv", metric_rows)
    write_csv(OUTPUT_DIR / "opa1b_missing_or_blocked_files.csv", missing_rows, ["item", "status", "path"])

    metrics_df = pd.DataFrame(metric_rows)
    summary_rows: list[dict[str, object]] = []
    if not metrics_df.empty:
        for model_key, sub in metrics_df.groupby("model_key"):
            means = sub.groupby("projection_condition")["pearson_preservation"].mean()
            real = float(means.get("realW", np.nan))
            rowperm = float(means.get("rowpermW", np.nan))
            gaussian = float(means.get("gaussianR", np.nan))
            summary_rows.append({
                "model_key": model_key,
                "mean_real_preservation": real,
                "mean_rowperm_preservation": rowperm,
                "mean_gaussian_preservation": gaussian,
                "real_minus_rowperm": real - rowperm,
                "real_minus_gaussian": real - gaussian,
                "gaussian_minus_real": gaussian - real,
                "mean_spread_real": float(sub[sub["projection_condition"] == "realW"]["spread_mean"].mean()),
                "mean_entropy_real": float(sub[sub["projection_condition"] == "realW"]["entropy_mean"].mean()),
            })
    write_csv(OUTPUT_DIR / "opa1b_by_model_summary.csv", summary_rows)
    write_csv(OUTPUT_DIR / "opa1b_psg_residual_summary.csv", summary_rows)
    verdict = verdict_from_summary(summary_rows, len(prompt_df), len(summary_rows))
    write_json(OUTPUT_DIR / "opa1b_verdict.json", verdict)
    lines = [
        "# OPA-1B PSG residual projection control audit",
        "",
        "This window inherits MethodologyKernel v1.1.",
        "If any experimental conclusion conflicts with this kernel, the conflict must be explicitly stated and converted into a theory revision, downgrade, or caveat.",
        "",
        "## Input status",
        "",
        f"- Hidden-state source: `{SOURCE_DIR}`",
        f"- Prompt source: `{PROMPT_SOURCE}`",
        f"- Prompts used: {len(prompt_df)}",
        f"- Projection repeats: {N_REPEATS} for rowpermW and gaussianR; realW is deterministic.",
        "",
        "## Verdict",
        "",
        f"`{verdict['verdict']}`",
        "",
        "## By-model summary",
        "",
    ]
    for row in summary_rows:
        lines.append(
            f"- {row['model_key']}: real={row['mean_real_preservation']:.4f}, "
            f"rowperm={row['mean_rowperm_preservation']:.4f}, gaussian={row['mean_gaussian_preservation']:.4f}, "
            f"real-rowperm={row['real_minus_rowperm']:.4f}, real-gaussian={row['real_minus_gaussian']:.4f}"
        )
    lines += [
        "",
        "## Boundary",
        "",
        "The run recomputed projection-control metrics from recovered hidden states. It does not use theory-summary numbers as data.",
    ]
    (OUTPUT_DIR / "opa1b_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
