# -*- coding: utf-8 -*-
r"""
GPT_208_DSTA_2A1_signed_rotation_reversal_audit.py

DSTA-2A.1: Signed Direction Rotation / Reversal Audit

目的：
在 DSTA-2A 的 BasisRotation 基础上补上一个关键盲点：
PCA principal angle 把 d 和 -d 视为同一子空间，因此会漏掉
override / negation 这类“方向反向”事件。

本实验新增：
1. SignedAlign:
      cos(v_condition, v_clean)
   取值：
      +1 同向
       0 正交 / 旋转
      -1 反向

2. ReversalScore:
      max(0, -cos(v_condition, v_clean))
   用于识别 override / negation 的反向。

3. RotationScore:
      arccos(|cos(v_condition, v_clean)|)
   忽略正负号，只看旋转幅度。
   用于识别 exception 的局部条件旋转。

4. DeltaCenter Direction:
      ΔC_l^{cond}=C_l^{cond}-C_l^{stable_clean}
   比较不同 condition 相对 clean 的漂移方向。

理论预测：
- closure_update / closure_temporal:
    high WeightShift, low-to-moderate Rotation, low Reversal
- closure_override / closure_negation:
    high ReversalScore
- closure_exception:
    high RotationScore, moderate ReversalScore
- hallucination_like:
    wrong-direction-like signed drift early or critical

硬编码模型路径：
    D:/model/models--Qwen--Qwen2.5-1.5B-Instruct/main

默认自动生成 DSTA-2A prompt 数据，不依赖外部 CSV。
如需要外部 CSV，设置 USE_EXTERNAL_CSV=True。

输出目录：
    dsta2a1_outputs/

关键输出：
    dsta2a1_signed_layer_metrics.csv
    dsta2a1_signed_window_summary.csv
    dsta2a1_signed_verdict.csv
    dsta2a1_generated_prompts.csv
"""

import os
import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


CONFIG = {
    "MODEL_PATH": r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main",
    "MODEL_PATH_FALLBACK": r"D:\model\models--Qwen--Qwen2.5B-Instruct\main",

    "USE_EXTERNAL_CSV": False,
    "INPUT_CSV": r"C:\Users\ZH\Desktop\AGI\python_script\dsta2a_closure_prompts.csv",

    "OUTPUT_DIR": r"dsta2a1_outputs",

    "LAYER_INDICES": list(range(0, 29)),
    "TOPK_LIST": [100, 500],

    "BATCH_SIZE": 4,
    "MAX_LENGTH": 224,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "DTYPE": "auto",
    "RANDOM_SEED": 42,

    "WINDOWS": {
        "init_0_6": [0, 1, 2, 3, 4, 5, 6],
        "mid_7_19": list(range(7, 20)),
        "critical_20_22": [20, 21, 22],
        "commit_23_26": [23, 24, 25, 26],
    },

    # 用 stable_clean 作为 signed comparison reference
    "REFERENCE_CONDITION": "stable_clean",

    # verdict thresholds
    "REVERSAL_STRONG": 0.15,
    "ROTATION_STRONG_DEG": 35.0,
    "SIGNED_ALIGN_NEGATIVE": -0.10,
    "DELTA_OPPOSE_STRONG": -0.20,
}


# ======================================================
# Prompt construction: mirrors DSTA-2A
# ======================================================

def build_prompt_dataset() -> pd.DataFrame:
    pairs = [
        ("Paris", "France", "Europe", "Asia"),
        ("Tokyo", "Japan", "Asia", "Europe"),
        ("Nile", "Egypt", "Africa", "Europe"),
        ("Amazon", "Brazil", "SouthAmerica", "Africa"),
        ("Tesla", "ElectricCars", "Technology", "Agriculture"),
        ("Python", "Programming", "Software", "Reptile"),
        ("Newton", "Physics", "Science", "Literature"),
        ("Shakespeare", "Drama", "Literature", "Physics"),
        ("MonaLisa", "DaVinci", "Art", "Biology"),
        ("Everest", "Himalayas", "Geography", "Medicine"),
        ("Saturn", "Planet", "Astronomy", "History"),
        ("Neuron", "Brain", "Biology", "Architecture"),
    ]

    templates = []

    def add(condition, mechanism, risk, template):
        templates.append({
            "condition": condition,
            "mechanism": mechanism,
            "risk_regime": risk,
            "template": template
        })

    add("stable_clean", "stable", "clean_basin",
        "{A} belongs to {B}. {B} is associated with {C}. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")
    add("stable_redundant", "stable", "clean_basin",
        "{A} belongs to {B}. {B} is associated with {C}. It is also stated that members of {B} normally map to {C}. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")
    add("weak_distractor", "stable", "low_risk_shift",
        "{A} belongs to {B}. {B} is associated with {C}. Unrelated note: another object is associated with {E}. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")

    add("competition_direct", "competition", "competition_low_risk",
        "{A} belongs to {B}. {B} is associated with {C}. However, one note says {A} is associated with {E}. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")
    add("competition_equal_evidence", "competition", "competition_high_risk",
        "Evidence 1: {A} belongs to {B}, and {B} is associated with {C}. Evidence 2: {A} is associated with {E}. Both evidence sources are presented as equally reliable. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")
    add("competition_source_claim", "competition", "competition_high_risk",
        "A source claims that {A} is associated with {E}. Another source says {A} belongs to {B}, and {B} is associated with {C}. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")

    add("closure_update", "closure", "closure_collapse",
        "Old rule: {B} is associated with {C}. New rule: in this graph, {B} is associated with {E}. {A} belongs to {B}. Use the new rule. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")
    add("closure_temporal", "closure", "closure_collapse",
        "Before 2020, {B} was associated with {C}. After 2020, {B} is associated with {E}. The current year is 2024. {A} belongs to {B}. Question: Which label is {A} associated with now, {C} or {E}? Answer with only one label.")
    add("closure_authority", "closure", "closure_collapse",
        "General note: {B} is associated with {C}. Verified registry: {B} is associated with {E}. {A} belongs to {B}. Follow the verified registry. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")
    add("closure_override", "closure_override", "closure_rule_override",
        "General rule: items belonging to {B} are associated with {C}. Override rule: for this task, items belonging to {B} are associated with {E}. {A} belongs to {B}. Use the override rule. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")
    add("closure_exception", "closure_exception", "closure_rule_override",
        "General rule: items belonging to {B} are associated with {C}. Exception: {A} is a special case associated with {E}. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")
    add("closure_negation", "closure_negation", "closure_collapse",
        "{A} belongs to {B}. It is false that {B} is associated with {C}. Instead, {B} is associated with {E}. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")
    add("hallucination_like", "hallucination_like", "wrong_closure",
        "A misleading summary says: {A} belongs to {B}; {B} is connected to {E}; therefore {A} is associated with {E}. The original background sometimes says {B} is associated with {C}. Question: Which label is {A} associated with in the misleading summary, {C} or {E}? Answer with only one label.")

    rows, sid = [], 0
    for A, B, C, E in pairs:
        for t in templates:
            prompt = t["template"].format(A=A, B=B, C=C, E=E)
            rows.append({
                "id": f"dsta2a1_{sid:05d}",
                "graph_id": f"{A}_{B}_{C}_{E}",
                "A": A,
                "B": B,
                "C_clean": C,
                "E_conflict": E,
                "concept": A,
                "condition": t["condition"],
                "policy_id": t["condition"],
                "mechanism": t["mechanism"],
                "risk_regime": t["risk_regime"],
                "surface_form": "synthetic_closure",
                "prompt": prompt,
            })
            sid += 1

    df = pd.DataFrame(rows)
    extra_rows = []
    for _, row in df.iterrows():
        A, B, C, E = row["A"], row["B"], row["C_clean"], row["E_conflict"]
        cond = row["condition"]
        if cond == "stable_clean":
            p2 = f"Facts: {A} -> {B}; {B} -> {C}. Choose the correct final label for {A}: {C} or {E}. Answer one label only."
        elif cond == "closure_override":
            p2 = f"Rule says {B}->{C}, but this task overrides the rule so {B}->{E}. Since {A}->{B}, choose {A}'s label: {C} or {E}. Answer one label only."
        elif cond == "closure_exception":
            p2 = f"Normally {B}->{C}. But {A} is explicitly listed as an exception mapping to {E}. Choose {A}'s label: {C} or {E}. Answer one label only."
        elif cond == "closure_negation":
            p2 = f"{A}->{B}. The statement {B}->{C} is negated. The replacement is {B}->{E}. Choose {A}'s label: {C} or {E}. Answer one label only."
        elif cond == "closure_update":
            p2 = f"Updated mapping: {B}->{E}, replacing old mapping {B}->{C}. {A}->{B}. Choose {A}'s current label: {C} or {E}. Answer one label only."
        elif cond == "closure_temporal":
            p2 = f"Old mapping before 2020: {B}->{C}. Current mapping after 2020: {B}->{E}. {A}->{B}. Current label for {A}: {C} or {E}? Answer one label only."
        elif cond == "closure_authority":
            p2 = f"Unofficial note says {B}->{C}. Verified source says {B}->{E}. {A}->{B}. Use verified source. Label for {A}: {C} or {E}? Answer one label only."
        elif cond == "hallucination_like":
            p2 = f"Misleading chain: {A}->{B}->{E}. Alternative background mentions {B}->{C}. Following the misleading chain, label {A}: {C} or {E}. Answer one label only."
        elif cond == "weak_distractor":
            p2 = f"{A}->{B}; {B}->{C}. An unrelated item maps to {E}. Choose {A}'s label: {C} or {E}. Answer one label only."
        elif cond == "competition_equal_evidence":
            p2 = f"One equally reliable path gives {A}->{B}->{C}; another equally reliable note gives {A}->{E}. Choose between {C} and {E}. Answer one label only."
        elif cond == "competition_source_claim":
            p2 = f"Source X claims {A}->{E}. Source Y says {A}->{B}->{C}. Choose the label for {A}: {C} or {E}. Answer one label only."
        else:
            p2 = row["prompt"]

        rr = row.copy()
        rr["id"] = f"{row['id']}_p2"
        rr["surface_form"] = "synthetic_closure_compact"
        rr["prompt"] = p2
        extra_rows.append(rr)

    return pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)


# ======================================================
# Utilities
# ======================================================

def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def autodetect_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    for c in df.columns:
        cl = c.lower()
        for name in candidates:
            if name.lower() in cl:
                return c
    if required:
        raise ValueError(f"Cannot find required column among {candidates}, existing={list(df.columns)}")
    return None


def get_torch_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return None


def load_model_and_tokenizer(model_path: str, device: str, dtype_name: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if not Path(model_path).exists():
        print(f"[WARN] MODEL_PATH not found: {model_path}")
        print(f"[WARN] Trying fallback: {CONFIG['MODEL_PATH_FALLBACK']}")
        model_path = CONFIG["MODEL_PATH_FALLBACK"]

    torch_dtype = get_torch_dtype(dtype_name)
    kwargs = {
        "local_files_only": True,
        "output_hidden_states": True,
    }
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    elif device == "cuda":
        kwargs["torch_dtype"] = torch.float16

    print(f"[LOAD] tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[LOAD] model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **kwargs)
    model.eval()
    model.to(device)
    return model, tokenizer


def get_lm_head_weight(model):
    emb = model.get_output_embeddings()
    if emb is None:
        raise RuntimeError("model.get_output_embeddings() returned None.")
    return emb.weight.detach()


def l2_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x) + eps)


def cos(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + eps) * (np.linalg.norm(b) + eps)))


def rotation_deg_from_abs_cos(c: float) -> float:
    cabs = min(1.0, max(0.0, abs(c)))
    return float(np.arccos(cabs) * 180.0 / np.pi)


def signed_rotation_deg(c: float) -> float:
    cc = min(1.0, max(-1.0, c))
    return float(np.arccos(cc) * 180.0 / np.pi)


# ======================================================
# Extraction
# ======================================================

@torch.no_grad()
def extract_topk_centers(model, tokenizer, prompts, layers, topk_list, batch_size, max_length, device):
    W = get_lm_head_weight(model).to(device).float()
    vocab_size, dim = W.shape
    print(f"[INFO] lm_head W shape: vocab={vocab_size}, dim={dim}")

    centers = {f"L{l}_K{k}": [] for l in layers for k in topk_list}
    scalar_rows = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        last_pos = attn.sum(dim=1) - 1

        out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
        hidden_states = out.hidden_states
        max_hidden_idx = len(hidden_states) - 1

        for bi in range(input_ids.size(0)):
            sample_idx = start + bi
            pos = int(last_pos[bi].item())

            for layer_idx in layers:
                if layer_idx > max_hidden_idx:
                    continue

                h = hidden_states[layer_idx][bi, pos, :].float()
                logits = torch.matmul(h, W.T)

                for k in topk_list:
                    kk = min(k, logits.numel())
                    vals, ids = torch.topk(logits, k=kk)
                    emb = W[ids]
                    center = emb.mean(dim=0)

                    center_norm = F.normalize(center[None, :], dim=-1)[0]
                    emb_norm = F.normalize(emb, dim=-1)
                    spread = (1.0 - torch.matmul(emb_norm, center_norm)).mean().item()

                    vals_f = vals.float()
                    probs = torch.softmax(vals_f, dim=0)
                    entropy = (-(probs * torch.log(probs + 1e-12)).sum()).item()
                    entropy_norm = entropy / math.log(kk + 1e-12)

                    key = f"L{layer_idx}_K{k}"
                    centers[key].append(center.detach().cpu().numpy().astype(np.float32))

                    scalar_rows.append({
                        "row_index": sample_idx,
                        "layer": layer_idx,
                        "topk": k,
                        "logit_top1": vals_f[0].item(),
                        "logit_mean": vals_f.mean().item(),
                        "logit_std": vals_f.std(unbiased=False).item(),
                        "logit_gap_1_2": (vals_f[0] - vals_f[1]).item() if kk > 1 else 0.0,
                        "entropy": entropy,
                        "entropy_norm": entropy_norm,
                        "spread": spread,
                        "center_norm": center.norm().item(),
                        "top_ids_head": " ".join(map(str, ids[:20].detach().cpu().tolist())),
                    })

        print(f"[EXTRACT] {min(start + batch_size, len(prompts))}/{len(prompts)}")

    centers_np = {}
    for key, vals in centers.items():
        if len(vals) == len(prompts):
            centers_np[key] = np.vstack(vals).astype(np.float32)

    return centers_np, pd.DataFrame(scalar_rows)


# ======================================================
# Signed metrics
# ======================================================

def compute_condition_centers(meta: pd.DataFrame, centers_np: Dict[str, np.ndarray], layers, topk_list):
    """
    condition_mean[(condition, layer, k)] = mean TopK center vector
    graph_condition_mean[(graph_id, condition, layer, k)] = mean center vector
    """
    cond_means = {}
    graph_cond_means = {}

    conds = sorted(meta["_condition"].unique())
    graph_ids = sorted(meta["graph_id"].unique()) if "graph_id" in meta.columns else []

    for k in topk_list:
        for l in layers:
            key = f"L{l}_K{k}"
            if key not in centers_np:
                continue
            X = centers_np[key]

            for cond in conds:
                idx = np.where(meta["_condition"].values == cond)[0]
                if len(idx):
                    cond_means[(cond, l, k)] = X[idx].mean(axis=0)

            for gid in graph_ids:
                for cond in conds:
                    idx = np.where((meta["graph_id"].values == gid) & (meta["_condition"].values == cond))[0]
                    if len(idx):
                        graph_cond_means[(gid, cond, l, k)] = X[idx].mean(axis=0)

    return cond_means, graph_cond_means


def signed_audit(meta: pd.DataFrame, centers_np: Dict[str, np.ndarray], layers, topk_list, out_dir: Path):
    ref = CONFIG["REFERENCE_CONDITION"]
    cond_means, graph_cond_means = compute_condition_centers(meta, centers_np, layers, topk_list)

    rows = []
    conds = sorted(meta["_condition"].unique())

    for k in topk_list:
        for l in layers:
            ref_key = (ref, l, k)
            if ref_key not in cond_means:
                continue

            ref_center = cond_means[ref_key]

            # Define baseline clean direction as movement from previous layer to current layer, if possible.
            # For l=0, use center itself as weak baseline.
            if l > min(layers) and (ref, l-1, k) in cond_means:
                ref_transport = cond_means[(ref, l, k)] - cond_means[(ref, l-1, k)]
            else:
                ref_transport = ref_center

            for cond in conds:
                ck = (cond, l, k)
                if ck not in cond_means:
                    continue

                c_center = cond_means[ck]
                delta_center = c_center - ref_center

                # condition transport direction
                if l > min(layers) and (cond, l-1, k) in cond_means:
                    cond_transport = cond_means[(cond, l, k)] - cond_means[(cond, l-1, k)]
                else:
                    cond_transport = c_center

                # signed alignment of condition transport vs clean transport
                transport_align = cos(cond_transport, ref_transport)
                transport_reversal = max(0.0, -transport_align)
                transport_rotation_abs_deg = rotation_deg_from_abs_cos(transport_align)
                transport_signed_angle_deg = signed_rotation_deg(transport_align)

                # signed alignment of condition center vs clean center
                center_align = cos(c_center, ref_center)
                center_reversal = max(0.0, -center_align)
                center_rotation_abs_deg = rotation_deg_from_abs_cos(center_align)
                center_signed_angle_deg = signed_rotation_deg(center_align)

                # delta-center direction alignments:
                # compare each condition delta to closure_update and closure_override when available
                delta_update_align = np.nan
                delta_override_align = np.nan
                if ("closure_update", l, k) in cond_means:
                    du = cond_means[("closure_update", l, k)] - ref_center
                    delta_update_align = cos(delta_center, du)
                if ("closure_override", l, k) in cond_means:
                    do = cond_means[("closure_override", l, k)] - ref_center
                    delta_override_align = cos(delta_center, do)

                rows.append({
                    "condition": cond,
                    "mechanism": meta.loc[meta["_condition"] == cond, "_mechanism"].iloc[0],
                    "risk_regime": meta.loc[meta["_condition"] == cond, "_risk"].iloc[0],
                    "layer": l,
                    "topk": k,
                    "n": int((meta["_condition"] == cond).sum()),

                    "center_signed_align_vs_clean": center_align,
                    "center_reversal_score": center_reversal,
                    "center_rotation_abs_deg": center_rotation_abs_deg,
                    "center_signed_angle_deg": center_signed_angle_deg,
                    "delta_center_norm": float(np.linalg.norm(delta_center)),

                    "transport_signed_align_vs_clean": transport_align,
                    "transport_reversal_score": transport_reversal,
                    "transport_rotation_abs_deg": transport_rotation_abs_deg,
                    "transport_signed_angle_deg": transport_signed_angle_deg,
                    "transport_delta_norm": float(np.linalg.norm(cond_transport - ref_transport)),

                    "delta_align_to_update": delta_update_align,
                    "delta_align_to_override": delta_override_align,
                })

    layer_df = pd.DataFrame(rows)
    layer_df.to_csv(out_dir / "dsta2a1_signed_layer_metrics.csv", index=False, encoding="utf-8-sig")

    # graph-paired version: condition vs stable_clean within same graph
    graph_rows = []
    graph_ids = sorted(meta["graph_id"].unique()) if "graph_id" in meta.columns else []
    for k in topk_list:
        for l in layers:
            for gid in graph_ids:
                ref_key = (gid, ref, l, k)
                if ref_key not in graph_cond_means:
                    continue
                ref_center = graph_cond_means[ref_key]
                for cond in conds:
                    ck = (gid, cond, l, k)
                    if ck not in graph_cond_means:
                        continue
                    c_center = graph_cond_means[ck]
                    delta = c_center - ref_center
                    center_align = cos(c_center, ref_center)
                    graph_rows.append({
                        "graph_id": gid,
                        "condition": cond,
                        "layer": l,
                        "topk": k,
                        "center_signed_align_vs_clean": center_align,
                        "center_reversal_score": max(0.0, -center_align),
                        "center_rotation_abs_deg": rotation_deg_from_abs_cos(center_align),
                        "center_signed_angle_deg": signed_rotation_deg(center_align),
                        "delta_center_norm": float(np.linalg.norm(delta)),
                    })
    graph_df = pd.DataFrame(graph_rows)
    graph_df.to_csv(out_dir / "dsta2a1_graph_paired_signed_metrics.csv", index=False, encoding="utf-8-sig")

    return layer_df, graph_df


def summarize_windows(layer_df: pd.DataFrame, out_dir: Path):
    rows = []
    for (topk, cond), subc in layer_df.groupby(["topk", "condition"]):
        for win_name, layers in CONFIG["WINDOWS"].items():
            sub = subc[subc["layer"].isin(set(layers))]
            if len(sub) == 0:
                continue
            rows.append({
                "topk": int(topk),
                "condition": cond,
                "mechanism": sub["mechanism"].iloc[0],
                "risk_regime": sub["risk_regime"].iloc[0],
                "window": win_name,
                "n_layers": int(len(sub)),
                "center_signed_align_mean": float(sub["center_signed_align_vs_clean"].mean()),
                "center_reversal_score_mean": float(sub["center_reversal_score"].mean()),
                "center_rotation_abs_deg_mean": float(sub["center_rotation_abs_deg"].mean()),
                "delta_center_norm_mean": float(sub["delta_center_norm"].mean()),
                "transport_signed_align_mean": float(sub["transport_signed_align_vs_clean"].mean()),
                "transport_reversal_score_mean": float(sub["transport_reversal_score"].mean()),
                "transport_rotation_abs_deg_mean": float(sub["transport_rotation_abs_deg"].mean()),
                "transport_delta_norm_mean": float(sub["transport_delta_norm"].mean()),
                "delta_align_to_update_mean": float(sub["delta_align_to_update"].mean()),
                "delta_align_to_override_mean": float(sub["delta_align_to_override"].mean()),
            })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "dsta2a1_signed_window_summary.csv", index=False, encoding="utf-8-sig")
    return out


def make_verdict(window_df: pd.DataFrame, out_dir: Path):
    rows = []
    for topk, subk in window_df.groupby("topk"):
        for cond, subc in subk.groupby("condition"):
            row = {"topk": int(topk), "condition": cond, "mechanism": subc["mechanism"].iloc[0], "risk_regime": subc["risk_regime"].iloc[0]}
            for win in CONFIG["WINDOWS"].keys():
                sw = subc[subc["window"] == win]
                if len(sw):
                    row[f"{win}_transport_align"] = float(sw.iloc[0]["transport_signed_align_mean"])
                    row[f"{win}_transport_reversal"] = float(sw.iloc[0]["transport_reversal_score_mean"])
                    row[f"{win}_transport_rotation_deg"] = float(sw.iloc[0]["transport_rotation_abs_deg_mean"])
                    row[f"{win}_center_align"] = float(sw.iloc[0]["center_signed_align_mean"])
                    row[f"{win}_center_reversal"] = float(sw.iloc[0]["center_reversal_score_mean"])
                    row[f"{win}_center_rotation_deg"] = float(sw.iloc[0]["center_rotation_abs_deg_mean"])
                    row[f"{win}_delta_align_to_update"] = float(sw.iloc[0]["delta_align_to_update_mean"])
                    row[f"{win}_delta_align_to_override"] = float(sw.iloc[0]["delta_align_to_override_mean"])

            crit_rev = row.get("critical_20_22_transport_reversal", np.nan)
            crit_rot = row.get("critical_20_22_transport_rotation_deg", np.nan)
            crit_align = row.get("critical_20_22_transport_align", np.nan)
            crit_delta_update = row.get("critical_20_22_delta_align_to_update", np.nan)
            crit_delta_override = row.get("critical_20_22_delta_align_to_override", np.nan)

            flags = []
            if np.isfinite(crit_rev) and crit_rev >= CONFIG["REVERSAL_STRONG"]:
                flags.append("critical_transport_reversal")
            if np.isfinite(crit_align) and crit_align <= CONFIG["SIGNED_ALIGN_NEGATIVE"]:
                flags.append("critical_negative_alignment")
            if np.isfinite(crit_rot) and crit_rot >= CONFIG["ROTATION_STRONG_DEG"]:
                flags.append("critical_transport_rotation")
            if np.isfinite(crit_delta_update) and crit_delta_update <= CONFIG["DELTA_OPPOSE_STRONG"]:
                flags.append("opposes_update_delta")
            if np.isfinite(crit_delta_override) and crit_delta_override >= 0.50:
                flags.append("aligns_override_delta")

            if "critical_transport_reversal" in flags or "critical_negative_alignment" in flags:
                verdict = "SIGNED_REVERSAL"
            elif "critical_transport_rotation" in flags:
                verdict = "SIGNED_ROTATION"
            elif "aligns_override_delta" in flags and cond not in ("closure_override", CONFIG["REFERENCE_CONDITION"]):
                verdict = "OVERRIDE_LIKE_DRIFT"
            else:
                verdict = "NO_STRONG_SIGNED_EVENT"

            # prior expectation tags
            if cond in ["closure_override", "closure_negation"]:
                expected = "expected_reversal"
            elif cond in ["closure_exception"]:
                expected = "expected_rotation_or_local_reversal"
            elif cond in ["closure_update", "closure_temporal", "closure_authority"]:
                expected = "expected_weight_transport_low_reversal"
            elif cond == "hallucination_like":
                expected = "expected_wrong_closure_signed_drift"
            else:
                expected = "control_or_competition"

            row["flags"] = ";".join(flags)
            row["expected_type"] = expected
            row["verdict"] = verdict
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "dsta2a1_signed_verdict.csv", index=False, encoding="utf-8-sig")
    return out


def main():
    np.random.seed(CONFIG["RANDOM_SEED"])
    torch.manual_seed(CONFIG["RANDOM_SEED"])

    out_dir = ensure_dir(CONFIG["OUTPUT_DIR"])
    with open(out_dir / "dsta2a1_config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)

    if CONFIG["USE_EXTERNAL_CSV"]:
        input_csv = Path(CONFIG["INPUT_CSV"])
        if not input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")
        df = pd.read_csv(input_csv)
    else:
        df = build_prompt_dataset()
        df.to_csv(out_dir / "dsta2a1_generated_prompts.csv", index=False, encoding="utf-8-sig")

    prompt_col = autodetect_col(df, ["prompt", "text", "input", "query"], True)
    concept_col = autodetect_col(df, ["concept", "topic", "entity", "subject", "A"], True)
    condition_col = autodetect_col(df, ["condition", "policy_id", "policy", "constraint", "mechanism"], True)
    mechanism_col = autodetect_col(df, ["mechanism"], False)
    risk_col = autodetect_col(df, ["risk_regime", "risk"], False)
    surface_col = autodetect_col(df, ["surface_form", "surface", "style", "template_family", "template"], False)

    meta = df.copy().reset_index(drop=True)
    meta["_id"] = meta["id"].astype(str) if "id" in meta.columns else [f"row_{i}" for i in range(len(meta))]
    meta["_prompt"] = meta[prompt_col].astype(str)
    meta["_concept"] = meta[concept_col].astype(str)
    meta["_condition"] = meta[condition_col].astype(str)
    meta["_mechanism"] = meta[mechanism_col].astype(str) if mechanism_col else meta["_condition"]
    meta["_risk"] = meta[risk_col].astype(str) if risk_col else meta["_condition"]
    meta["_surface"] = meta[surface_col].astype(str) if surface_col else "__nosurface__"

    print("[DATA] rows:", len(meta))
    print("[DATA] n_condition:", meta["_condition"].nunique())
    print("[DATA] condition counts:")
    print(meta["_condition"].value_counts().to_string())
    meta.to_csv(out_dir / "dsta2a1_dataset_resolved.csv", index=False, encoding="utf-8-sig")

    model, tokenizer = load_model_and_tokenizer(CONFIG["MODEL_PATH"], CONFIG["DEVICE"], CONFIG["DTYPE"])

    centers_np, scalar_df = extract_topk_centers(
        model=model,
        tokenizer=tokenizer,
        prompts=meta["_prompt"].tolist(),
        layers=CONFIG["LAYER_INDICES"],
        topk_list=CONFIG["TOPK_LIST"],
        batch_size=CONFIG["BATCH_SIZE"],
        max_length=CONFIG["MAX_LENGTH"],
        device=CONFIG["DEVICE"],
    )

    scalar_df = scalar_df.merge(
        meta[["_id", "_concept", "_condition", "_mechanism", "_risk", "_surface"]].reset_index().rename(columns={"index": "row_index"}),
        on="row_index",
        how="left"
    )
    scalar_df.to_csv(out_dir / "dsta2a1_scalar_topk_features.csv", index=False, encoding="utf-8-sig")

    available_layers = sorted(set(int(k.split("_")[0][1:]) for k in centers_np.keys()))
    layers = [l for l in CONFIG["LAYER_INDICES"] if l in available_layers]
    print("[INFO] available layers:", layers)

    layer_df, graph_df = signed_audit(meta, centers_np, layers, CONFIG["TOPK_LIST"], out_dir)
    window_df = summarize_windows(layer_df, out_dir)
    verdict_df = make_verdict(window_df, out_dir)

    print("\n[DONE] Outputs saved to:", out_dir.resolve())
    print("\n[SIGNED WINDOW SUMMARY HEAD]")
    print(window_df.head(80).to_string(index=False) if len(window_df) else "No summary.")
    print("\n[SIGNED VERDICT]")
    print(verdict_df.to_string(index=False) if len(verdict_df) else "No verdict.")

    print("\n[KEY FILES]")
    for name in [
        "dsta2a1_signed_layer_metrics.csv",
        "dsta2a1_signed_window_summary.csv",
        "dsta2a1_signed_verdict.csv",
        "dsta2a1_graph_paired_signed_metrics.csv",
        "dsta2a1_generated_prompts.csv",
    ]:
        print(" -", out_dir / name)


if __name__ == "__main__":
    main()
