# -*- coding: utf-8 -*-
"""
CM-1 Cross-Model Minimal Validation

Goal:
  Validate the minimal chain across Qwen / Llama / Gemma:

      TopK_init -> DeltaR(decision window) -> DeltaU(PC1) -> tau-like phase boundary

This script is designed to run locally on your machine with downloaded HuggingFace models.
It does NOT download models by default. Use local paths.

Outputs:
  cm1_outputs/
    cm1_model_summary.csv
    cm1_overall_summary.json
    <model_key>_layer_R_dataset.csv
    <model_key>_deltaR_dataset.csv
    <model_key>_pca_summary.json
    <model_key>_phase_profile.csv
    <model_key>_tau_profile.csv
    <model_key>_topk_init_summary.csv

Recommended usage:
  python cm1_cross_model_validation_with_paths.py

Local model paths are hard-coded below, but can still be overridden by CLI args.

Notes:
  - Layer numbers are compared using normalized depth s=l/L.
  - For Qwen-28L, historical tau is L20-L22, i.e. s≈0.71-0.79.
  - The script maps this interval to each model's own layer count.
"""

import argparse
import gc
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer



# -----------------------------
# Local Model Paths
# -----------------------------

QWEN_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
LLAMA_PATH = r"D:\model\Llama-3.2-1B-Instruct"
GEMMA_PATH = r"D:\model\gemma-2-2b-it"

# -----------------------------
# Config
# -----------------------------

SEED = 42
DEFAULT_N_GRAPHS = 72
MAX_LEN = 220
BATCH_SIZE = 4
TOPK = 100
INIT_LAYER_MAX_NORM = 0.25  # first quarter of model layers, capped below
DECISION_RANGE = (20 / 28, 25 / 28)  # Qwen L20-L25 normalized
TAU_RANGE = (20 / 28, 22 / 28)       # Qwen L20-L22 normalized

LABEL_POOL = [
    "Red", "Blue", "Green", "Yellow", "North", "South", "East", "West",
    "Alpha", "Beta", "Gamma", "Delta", "Circle", "Square", "Triangle", "Star",
    "Copper", "Silver", "Gold", "Iron", "River", "Forest", "Ocean", "Mountain",
]

CONDITIONS = [
    ("stable_positive", "positive"),
    ("weak_positive", "positive"),
    ("true_compete_balanced", "critical"),
    ("direct_negative", "negative"),
    ("update_negative", "negative"),
    ("exception_negative", "negative"),
]


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_name(path_or_key: str) -> str:
    return path_or_key.replace("/", "_").replace("\\", "_").replace(":", "_").replace("-", "_")


def get_layers(model) -> List[torch.nn.Module]:
    # Common decoder-only model layouts.
    for attr_path in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        obj = model
        ok = True
        for part in attr_path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return list(obj)
    raise RuntimeError("Could not find transformer layers. Add your model layout to get_layers().")


def get_lm_head_weight(model) -> torch.Tensor:
    if hasattr(model, "lm_head"):
        return model.lm_head.weight.detach()
    if hasattr(model, "embed_out"):
        return model.embed_out.weight.detach()
    raise RuntimeError("Could not find lm_head / embed_out weight.")


def continuation_ids(tokenizer, label: str) -> List[int]:
    return tokenizer(" " + label, add_special_tokens=False)["input_ids"]


def select_single_token_labels(tokenizer, min_needed: int = 12) -> List[str]:
    singles = []
    audit = []
    for lab in LABEL_POOL:
        ids = continuation_ids(tokenizer, lab)
        audit.append((lab, ids))
        if len(ids) == 1:
            singles.append(lab)
    if len(singles) < min_needed:
        # Fall back to first-token scoring; still valid as first-token answer basin probe.
        print("[WARN] Fewer than required single-token labels. Falling back to first-token labels.")
        return LABEL_POOL[:min_needed]
    return singles[:min_needed]


def layer_window(num_layers: int, norm_range: Tuple[float, float]) -> List[int]:
    a, b = norm_range
    lo = int(round(a * num_layers))
    hi = int(round(b * num_layers))
    lo = max(0, min(num_layers - 1, lo))
    hi = max(lo, min(num_layers - 1, hi))
    return list(range(lo, hi + 1))


def init_layers(num_layers: int) -> List[int]:
    # Map Qwen L0-L6 roughly to first quarter, but keep a practical cap.
    max_l = min(num_layers - 1, max(2, int(round(INIT_LAYER_MAX_NORM * num_layers))))
    return list(range(0, max_l + 1))


def eta_squared(values: np.ndarray, groups: List[str]) -> float:
    values = np.asarray(values, dtype=np.float64)
    grand = np.mean(values)
    ss_total = np.sum((values - grand) ** 2)
    if ss_total <= 1e-12:
        return 0.0
    ss_between = 0.0
    for g in sorted(set(groups)):
        idx = np.array([x == g for x in groups])
        if idx.sum() == 0:
            continue
        ss_between += idx.sum() * (np.mean(values[idx]) - grand) ** 2
    return float(ss_between / ss_total)


# -----------------------------
# Dataset
# -----------------------------

def make_prompt(a: str, b: str, clean_label: str, conflict_label: str, condition: str) -> str:
    options = f"Answer with exactly one word: {clean_label} or {conflict_label}."
    base = [
        "You are given a small relation graph.",
        f"Possible labels are: {clean_label} or {conflict_label}.",
    ]

    if condition == "stable_positive":
        facts = [
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
        ]
    elif condition == "weak_positive":
        facts = [
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Weak note: another unrelated item may be associated with {conflict_label}.",
        ]
    elif condition == "true_compete_balanced":
        facts = [
            f"Source one says: {a} belongs to {b}, and {b} is located at {clean_label}.",
            f"Source two says: {a} is located at {conflict_label}.",
            "Both sources are presented without priority.",
        ]
    elif condition == "direct_negative":
        facts = [
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Direct override: {a} is located at {conflict_label}.",
        ]
    elif condition == "update_negative":
        facts = [
            f"Old record: {b} is located at {clean_label}.",
            f"Updated record: {b} is now located at {conflict_label}.",
            f"Fact: {a} belongs to {b}.",
            "Use the updated record.",
        ]
    elif condition == "exception_negative":
        facts = [
            f"General rule: items belonging to {b} are located at {clean_label}.",
            f"Exception: {a} is a special case located at {conflict_label}.",
            f"Fact: {a} belongs to {b}.",
            "Use the exception when it applies.",
        ]
    else:
        raise ValueError(condition)

    q = [f"Question: Which label is {a} located at?", options, "Answer:"]
    return "\n".join(base + facts + q)


def build_dataset(tokenizer, n_graphs: int) -> Tuple[pd.DataFrame, Dict[str, str]]:
    labels = select_single_token_labels(tokenizer, min_needed=12)
    rows = []
    prompts = {}
    for i in range(n_graphs):
        a = f"A{i:03d}"
        b = f"B{i:03d}"
        clean_label = labels[(2 * i) % len(labels)]
        conflict_label = labels[(2 * i + 1) % len(labels)]
        c_id = continuation_ids(tokenizer, clean_label)[0]
        e_id = continuation_ids(tokenizer, conflict_label)[0]
        for cond, phase in CONDITIONS:
            prompt = make_prompt(a, b, clean_label, conflict_label, cond)
            key = f"{i}::{cond}"
            prompts[key] = prompt
            rows.append({
                "graph_id": i,
                "condition": cond,
                "phase": phase,
                "clean_label": clean_label,
                "conflict_label": conflict_label,
                "clean_token_id": c_id,
                "conflict_token_id": e_id,
                "prompt_key": key,
            })
    return pd.DataFrame(rows), prompts


# -----------------------------
# Extraction
# -----------------------------

def extract_model(model_key: str, model_path: str, outdir: Path, n_graphs: int, device: str, dtype: str) -> Dict:
    print(f"\n==== Loading {model_key}: {model_path} ====")
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16 if dtype == "bfloat16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model.to("cpu")
    model.eval()

    layers = get_layers(model)
    L = len(layers)
    decision_layers = layer_window(L, DECISION_RANGE)
    tau_expected_layers = layer_window(L, TAU_RANGE)
    init_ls = init_layers(L)

    print(f"[{model_key}] layers={L}, init={init_ls}, decision={decision_layers}, expected_tau={tau_expected_layers}")

    df_meta, prompts = build_dataset(tokenizer, n_graphs)
    W = get_lm_head_weight(model).detach()
    W_cpu = W.float().cpu()

    # Store per-row per-layer R and TopK init stats.
    layer_rows = []
    topk_rows = []

    prompt_keys = df_meta["prompt_key"].tolist()
    texts = [prompts[k] for k in prompt_keys]

    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start:start + BATCH_SIZE]
            batch_df = df_meta.iloc[start:start + BATCH_SIZE].reset_index(drop=True)
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            # hidden_states[0] = embedding, hidden_states[l+1] = after layer l
            hidden_states = outputs.hidden_states
            last_pos = inputs["attention_mask"].shape[1] - 1

            c_ids = torch.tensor(batch_df["clean_token_id"].values, device=model.device, dtype=torch.long)
            e_ids = torch.tensor(batch_df["conflict_token_id"].values, device=model.device, dtype=torch.long)
            Wc = W.index_select(0, c_ids).float()
            We = W.index_select(0, e_ids).float()

            for bi in range(len(batch_texts)):
                base = batch_df.iloc[bi].to_dict()
                rec = {k: base[k] for k in ["graph_id", "condition", "phase", "clean_label", "conflict_label", "prompt_key"]}
                for l in range(L):
                    h = hidden_states[l + 1][bi, last_pos, :].float()
                    r = torch.dot(h, Wc[bi] - We[bi]).item()
                    rec[f"R_L{l}"] = r
                layer_rows.append(rec)

                topk_rec = rec.copy()
                # Init TopK stats: entropy-ish logit spread over TopK, mean top logit, top gap.
                for l in init_ls:
                    h = hidden_states[l + 1][bi, last_pos, :].float()
                    logits = torch.matmul(W.float(), h)
                    vals, idx = torch.topk(logits, k=min(TOPK, logits.numel()))
                    probs = torch.softmax(vals, dim=0)
                    ent = -(probs * torch.log(probs + 1e-12)).sum().item() / math.log(len(vals))
                    topk_rec[f"init_L{l}_topk_mean"] = vals.mean().item()
                    topk_rec[f"init_L{l}_topk_std"] = vals.std().item()
                    topk_rec[f"init_L{l}_top_gap"] = (vals[0] - vals[-1]).item()
                    topk_rec[f"init_L{l}_topk_entropy_norm"] = ent
                topk_rows.append(topk_rec)

            del outputs, hidden_states, inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df_R = pd.DataFrame(layer_rows)
    df_topk = pd.DataFrame(topk_rows)
    df_R.to_csv(outdir / f"{model_key}_layer_R_dataset.csv", index=False, encoding="utf-8-sig")

    # Build deltaR relative to stable_positive per graph.
    delta_rows = []
    R_cols_dec = [f"R_L{l}" for l in decision_layers]
    R_cols_all = [f"R_L{l}" for l in range(L)]
    clean_df = df_R[df_R["condition"] == "stable_positive"].set_index("graph_id")

    for _, row in df_R.iterrows():
        gid = row["graph_id"]
        clean = clean_df.loc[gid]
        out = {
            "graph_id": gid,
            "condition": row["condition"],
            "phase": row["phase"],
        }
        for l in range(L):
            out[f"dR_L{l}"] = float(row[f"R_L{l}"] - clean[f"R_L{l}"])
        out["deltaR_l2_decision"] = float(np.linalg.norm([out[f"dR_L{l}"] for l in decision_layers]))
        out["R_mean_decision"] = float(np.mean([row[f"R_L{l}"] for l in decision_layers]))
        out["R_final"] = float(row[f"R_L{L-1}"])
        delta_rows.append(out)

    df_delta = pd.DataFrame(delta_rows)
    df_delta.to_csv(outdir / f"{model_key}_deltaR_dataset.csv", index=False, encoding="utf-8-sig")

    # PCA DeltaU on decision window, excluding clean zero rows? Include all to preserve phase baseline.
    X = df_delta[[f"dR_L{l}" for l in decision_layers]].values.astype(np.float32)
    pca = PCA(n_components=min(3, X.shape[1]))
    pcs = pca.fit_transform(X)
    pc1 = pcs[:, 0]

    # Orient PC1 so positive phase mean > negative phase mean.
    phases = df_delta["phase"].tolist()
    mean_pos = float(np.mean(pc1[np.array(phases) == "positive"]))
    mean_neg = float(np.mean(pc1[np.array(phases) == "negative"]))
    sign = 1.0 if mean_pos >= mean_neg else -1.0
    df_delta["DeltaU_PC1"] = pc1 * sign

    phase_means = df_delta.groupby("phase")["DeltaU_PC1"].mean().to_dict()
    monotonic = phase_means.get("positive", 0) > phase_means.get("critical", 0) > phase_means.get("negative", 0)

    # Tau proxy: eta^2 by phase per layer over raw R and deltaR.
    tau_rows = []
    for l in range(L):
        eta_R = eta_squared(df_R[f"R_L{l}"].values, df_R["phase"].tolist())
        eta_dR = eta_squared(df_delta[f"dR_L{l}"].values, df_delta["phase"].tolist())
        tau_rows.append({
            "layer": l,
            "s_norm": l / max(1, L),
            "eta2_R_phase": eta_R,
            "eta2_dR_phase": eta_dR,
            "expected_tau_band": bool(l in tau_expected_layers),
        })
    df_tau = pd.DataFrame(tau_rows)
    tau_layer_R = int(df_tau.loc[df_tau["eta2_R_phase"].idxmax(), "layer"])
    tau_layer_dR = int(df_tau.loc[df_tau["eta2_dR_phase"].idxmax(), "layer"])
    df_tau.to_csv(outdir / f"{model_key}_tau_profile.csv", index=False, encoding="utf-8-sig")

    # TopK init summary: delta to clean, factor/phase eta.
    topk_feature_cols = [c for c in df_topk.columns if c.startswith("init_L")]
    clean_topk = df_topk[df_topk["condition"] == "stable_positive"].set_index("graph_id")
    topk_delta_rows = []
    for _, row in df_topk.iterrows():
        gid = row["graph_id"]
        clean = clean_topk.loc[gid]
        out = {"graph_id": gid, "condition": row["condition"], "phase": row["phase"]}
        for c in topk_feature_cols:
            out["d_" + c] = float(row[c] - clean[c])
        topk_delta_rows.append(out)
    df_topk_delta = pd.DataFrame(topk_delta_rows)
    topk_summary = []
    for c in ["d_" + x for x in topk_feature_cols]:
        topk_summary.append({
            "feature": c,
            "eta2_phase": eta_squared(df_topk_delta[c].values, df_topk_delta["phase"].tolist()),
            "mean_abs": float(np.mean(np.abs(df_topk_delta[c].values))),
        })
    df_topk_summary = pd.DataFrame(topk_summary).sort_values("eta2_phase", ascending=False)
    df_topk_summary.to_csv(outdir / f"{model_key}_topk_init_summary.csv", index=False, encoding="utf-8-sig")

    # Save phase profile.
    prof_rows = []
    for phase, g in df_delta.groupby("phase"):
        rec = {"phase": phase, "n": len(g), "DeltaU_mean": float(g["DeltaU_PC1"].mean()), "DeltaU_std": float(g["DeltaU_PC1"].std())}
        for l in decision_layers:
            rec[f"dR_L{l}_mean"] = float(g[f"dR_L{l}"].mean())
        prof_rows.append(rec)
    df_phase = pd.DataFrame(prof_rows)
    df_phase.to_csv(outdir / f"{model_key}_phase_profile.csv", index=False, encoding="utf-8-sig")
    df_delta.to_csv(outdir / f"{model_key}_deltaR_dataset.csv", index=False, encoding="utf-8-sig")

    pca_summary = {
        "model_key": model_key,
        "model_path": model_path,
        "num_layers": L,
        "init_layers": init_ls,
        "decision_layers": decision_layers,
        "expected_tau_layers": tau_expected_layers,
        "pc_variance": pca.explained_variance_ratio_.tolist(),
        "pc1_variance": float(pca.explained_variance_ratio_[0]),
        "pc1_oriented_positive_gt_negative": bool(sign > 0),
        "phase_means_DeltaU": phase_means,
        "phase_order_positive_gt_critical_gt_negative": bool(monotonic),
        "tau_layer_R_eta_max": tau_layer_R,
        "tau_layer_dR_eta_max": tau_layer_dR,
        "tau_layer_R_s_norm": tau_layer_R / max(1, L),
        "tau_layer_dR_s_norm": tau_layer_dR / max(1, L),
        "tau_R_in_expected_band": bool(tau_layer_R in tau_expected_layers),
        "tau_dR_in_expected_band": bool(tau_layer_dR in tau_expected_layers),
        "topk_init_best_eta2_phase": float(df_topk_summary["eta2_phase"].max()) if len(df_topk_summary) else None,
        "topk_init_best_feature": str(df_topk_summary.iloc[0]["feature"]) if len(df_topk_summary) else None,
    }
    with open(outdir / f"{model_key}_pca_summary.json", "w", encoding="utf-8") as f:
        json.dump(pca_summary, f, ensure_ascii=False, indent=2)

    # Cleanup model before next one.
    del model, tokenizer, W, W_cpu
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pca_summary


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", type=str, default=QWEN_PATH, help="Local path to Qwen model")
    parser.add_argument("--llama", type=str, default=LLAMA_PATH, help="Local path to Llama model")
    parser.add_argument("--gemma", type=str, default=GEMMA_PATH, help="Local path to Gemma model")
    parser.add_argument("--outdir", type=str, default="cm1_outputs")
    parser.add_argument("--n_graphs", type=int, default=DEFAULT_N_GRAPHS)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    set_seed(SEED)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_paths = []
    if args.qwen:
        model_paths.append(("qwen", args.qwen))
    if args.llama:
        model_paths.append(("llama", args.llama))
    if args.gemma:
        model_paths.append(("gemma", args.gemma))

    print("\n==== Using local model paths ====")
    for k, p in model_paths:
        print(f"{k}: {p}")
        if not os.path.exists(p):
            print(f"[WARN] Path does not exist on this machine: {p}")
    if not model_paths:
        raise SystemExit("Please provide at least one model path: --qwen / --llama / --gemma")

    summaries = []
    for key, path in model_paths:
        try:
            summaries.append(extract_model(key, path, outdir, args.n_graphs, args.device, args.dtype))
        except Exception as e:
            print(f"[ERROR] {key} failed: {e}")
            summaries.append({"model_key": key, "model_path": path, "error": repr(e)})
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pd.DataFrame(summaries).to_csv(outdir / "cm1_model_summary.csv", index=False, encoding="utf-8-sig")
    with open(outdir / "cm1_overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print("\n==== CM-1 complete ====")
    print(f"Outputs saved to: {outdir.resolve()}")
    print(pd.DataFrame(summaries))


if __name__ == "__main__":
    main()
