# -*- coding: utf-8 -*-
r"""
GPT_206_DSTA_1C_condition_specific_basis_transport_audit.py

DSTA-1C: Condition-Specific Basis Transport Audit

目标：
1. 修正 DSTA-1A/1B 的核心 caveat：
   DSTA-1A/1B 的 BasisRotation 是全样本全局 PCA basis 的层间旋转，
   不能判定不同 policy_id 是否存在不同方向基运输。

2. 本实验按 policy_id / condition 单独拟合：
       D_l^{policy}
   然后计算：
       BasisRotation_l^{policy}
       WeightShift_l^{policy}

3. 直接回答：
       不同 PolicyID 是否对应不同方向基运输？
       是否存在 WeightShift 主导型策略？
       是否存在 BasisRotation 主导型策略？
       哪些 PolicyID 在 critical_20_22 出现方向基重构？

硬编码路径：
    MODEL_PATH = D:/model/models--Qwen--Qwen2.5-1.5B-Instruct/main
    INPUT_CSV  = C:/Users/ZH/Desktop/AGI/python_script/sem_dsta_prompts.csv

输入 CSV 推荐列：
    id, concept, policy_id, surface_form, prompt

输出目录：
    dsta1c_outputs/

主要输出：
    dsta1c_policy_basis_rotation_metrics.csv
    dsta1c_policy_weight_shift_metrics.csv
    dsta1c_policy_basis_vs_weight_window_summary.csv
    dsta1c_policy_transport_verdict.csv
    dsta1c_layer_policy_weights.csv
    dsta1c_scalar_topk_features.csv

设计说明：
- K=500 是主指标；K=100 作为敏感性对照。
- 每个 policy_id 内部单独拟合每层 PCA basis。
- 若某个 policy_id 样本数不足，会自动降低 BASIS_DIM；低于 3 个样本则跳过。
- 方向权重 W_l 由每个样本在 policy-specific PCA basis 上的 squared projection mass 得到。
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

from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# =========================
# CONFIG: 已按用户本地环境硬编码
# =========================
CONFIG = {
    "MODEL_PATH": r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main",
    "INPUT_CSV": r"C:\Users\ZH\Desktop\AGI\python_script\sem_dsta_prompts.csv",
    "OUTPUT_DIR": r"dsta1c_outputs",

    "LAYER_INDICES": list(range(0, 29)),

    # K=500 主指标；K=100 敏感性对照
    "TOPK_LIST": [100, 500],

    # policy 内样本通常较少，basis_dim 不宜太高
    "BASIS_DIM": 4,

    # 低于该样本数的 policy 不拟合 policy-specific PCA
    "MIN_SAMPLES_PER_POLICY": 4,

    "BATCH_SIZE": 4,
    "MAX_LENGTH": 192,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "DTYPE": "auto",

    "RANDOM_SEED": 42,
    "WEIGHT_EPS": 1e-12,

    "WINDOWS": {
        "init_0_6": [0, 1, 2, 3, 4, 5, 6],
        "mid_7_19": list(range(7, 20)),
        "critical_20_22": [20, 21, 22],
        "commit_23_26": [23, 24, 25, 26],
    },

    # 判据阈值为经验启发式，最终以曲线和排序为准
    "ROTATION_HIGH_DEG": 45.0,
    "WEIGHT_SHIFT_HIGH": 0.40,
}


# =========================
# Utility
# =========================

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
        raise ValueError(
            f"Cannot find required column among candidates={candidates}. "
            f"Existing columns={list(df.columns)}"
        )
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[LOAD] model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        **kwargs
    )
    model.eval()
    model.to(device)
    return model, tokenizer


def get_lm_head_weight(model) -> torch.Tensor:
    emb = model.get_output_embeddings()
    if emb is None:
        raise RuntimeError("model.get_output_embeddings() returned None; cannot compute H_l W^T.")
    return emb.weight.detach()


def l2_normalize_np(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def normalized_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    if len(p) <= 1:
        return 0.0
    p = p / (p.sum() + eps)
    h = -(p * np.log(p + eps)).sum()
    return float(h / math.log(len(p)))


def dominance_gap(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64)
    if len(w) == 0:
        return np.nan
    if len(w) == 1:
        return float(w[0])
    s = np.sort(w)
    return float(s[-1] - s[-2])


def principal_angle_metrics(B1: np.ndarray, B2: np.ndarray) -> Dict:
    r = min(B1.shape[0], B2.shape[0])
    if r < 1:
        return {
            "basis_similarity": np.nan,
            "basis_rotation_angle_mean_deg": np.nan,
            "basis_rotation_angle_max_deg": np.nan,
            "basis_rotation_angle_min_deg": np.nan,
        }
    A = B1[:r] @ B2[:r].T
    _, s, _ = np.linalg.svd(A, full_matrices=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(np.abs(s)) * 180.0 / np.pi
    return {
        "basis_similarity": float(np.mean(np.abs(s))),
        "basis_rotation_angle_mean_deg": float(np.mean(angles)),
        "basis_rotation_angle_max_deg": float(np.max(angles)),
        "basis_rotation_angle_min_deg": float(np.min(angles)),
    }


# =========================
# TopK extraction
# =========================

@torch.no_grad()
def extract_topk_centers(
    model,
    tokenizer,
    prompts: List[str],
    layers: List[int],
    topk_list: List[int],
    batch_size: int,
    max_length: int,
    device: str,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    W = get_lm_head_weight(model).to(device)
    W_float = W.float()
    vocab_size, dim = W.shape
    print(f"[INFO] lm_head W shape: vocab={vocab_size}, dim={dim}")

    centers: Dict[str, List[np.ndarray]] = {f"L{l}_K{k}": [] for l in layers for k in topk_list}
    scalar_rows = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
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
                logits = torch.matmul(h, W_float.T)

                for k in topk_list:
                    kk = min(k, logits.numel())
                    vals, ids = torch.topk(logits, k=kk)
                    emb = W_float[ids]
                    center = emb.mean(dim=0)

                    center_norm = F.normalize(center[None, :], dim=-1)[0]
                    emb_norm = F.normalize(emb, dim=-1)
                    cos_to_center = torch.matmul(emb_norm, center_norm)
                    spread = (1.0 - cos_to_center).mean().item()

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
                        "logit_gap_1_mean": (vals_f[0] - vals_f.mean()).item(),
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

    scalar_df = pd.DataFrame(scalar_rows)
    return centers_np, scalar_df


# =========================
# Policy-specific decomposition
# =========================

def fit_policy_layer_basis(X: np.ndarray, basis_dim: int, seed: int) -> Optional[Dict]:
    """
    X: [n_policy_samples, d]
    """
    n, d = X.shape
    r = min(basis_dim, n - 1, d)
    if n < 3 or r < 1:
        return None

    Xn = l2_normalize_np(X.astype(np.float32))
    mean = Xn.mean(axis=0, keepdims=True)
    Xc = Xn - mean

    pca = PCA(n_components=r, random_state=seed)
    scores = pca.fit_transform(Xc)
    basis = pca.components_.astype(np.float32)
    explained = pca.explained_variance_ratio_.astype(np.float32)

    mass = scores ** 2
    weights = mass / (mass.sum(axis=1, keepdims=True) + CONFIG["WEIGHT_EPS"])

    return {
        "mean": mean.astype(np.float32),
        "basis": basis,
        "scores": scores.astype(np.float32),
        "weights": weights.astype(np.float32),
        "explained": explained,
    }


def run_condition_specific_audit(
    meta: pd.DataFrame,
    centers_np: Dict[str, np.ndarray],
    scalar_df: pd.DataFrame,
    layers: List[int],
    topk_list: List[int],
    out_dir: Path,
) -> Dict[str, pd.DataFrame]:
    policy_values = sorted(meta["_policy"].astype(str).unique())

    basis_rows = []
    weight_rows = []
    layer_weight_rows = []
    fit_summary_rows = []

    for topk in topk_list:
        for policy in policy_values:
            idx = np.where(meta["_policy"].astype(str).values == policy)[0]
            if len(idx) < CONFIG["MIN_SAMPLES_PER_POLICY"]:
                print(f"[SKIP] policy={policy}, n={len(idx)} < MIN_SAMPLES")
                continue

            # layer fits for this policy
            fits: Dict[int, Dict] = {}

            for l in layers:
                key = f"L{l}_K{topk}"
                if key not in centers_np:
                    continue
                X_all = centers_np[key]
                Xp = X_all[idx]
                fit = fit_policy_layer_basis(Xp, CONFIG["BASIS_DIM"], CONFIG["RANDOM_SEED"])
                if fit is None:
                    continue
                fits[l] = fit

                # record sample weights for this policy/layer
                weights = fit["weights"]
                for local_i, global_i in enumerate(idx):
                    w = weights[local_i]
                    row = {
                        "row_index": int(global_i),
                        "id": meta.loc[global_i, "_id"],
                        "concept": meta.loc[global_i, "_concept"],
                        "policy_id": policy,
                        "surface_form": meta.loc[global_i, "_surface"],
                        "layer": l,
                        "topk": topk,
                        "basis_dim": int(len(w)),
                        "weight_entropy": normalized_entropy(w),
                        "dominance_gap_w1_w2": dominance_gap(w),
                        "top_weight": float(np.max(w)),
                    }
                    for j, val in enumerate(w):
                        row[f"w{j+1}"] = float(val)
                    layer_weight_rows.append(row)

                fit_summary_rows.append({
                    "policy_id": policy,
                    "layer": l,
                    "topk": topk,
                    "n": int(len(idx)),
                    "basis_dim": int(fit["basis"].shape[0]),
                    "pc1_explained": float(fit["explained"][0]) if len(fit["explained"]) > 0 else np.nan,
                    "pc2_explained": float(fit["explained"][1]) if len(fit["explained"]) > 1 else np.nan,
                    "pc3_explained": float(fit["explained"][2]) if len(fit["explained"]) > 2 else np.nan,
                    "pc_cum_basis_dim": float(np.sum(fit["explained"])),
                    "mean_weight_entropy": float(np.mean([normalized_entropy(w) for w in weights])),
                    "mean_dominance_gap": float(np.mean([dominance_gap(w) for w in weights])),
                    "mean_top_weight": float(np.max(weights, axis=1).mean()),
                })

            sorted_layers = sorted(fits.keys())
            for l1, l2 in zip(sorted_layers[:-1], sorted_layers[1:]):
                f1 = fits[l1]
                f2 = fits[l2]

                # basis rotation for policy
                bm = principal_angle_metrics(f1["basis"], f2["basis"])
                basis_rows.append({
                    "policy_id": policy,
                    "topk": topk,
                    "layer_from": l1,
                    "layer_to": l2,
                    "n": int(len(idx)),
                    **bm,
                })

                # weight shift for same policy samples
                w1 = f1["weights"]
                w2 = f2["weights"]
                r = min(w1.shape[1], w2.shape[1])
                w1r, w2r = w1[:, :r], w2[:, :r]
                shift = np.linalg.norm(w2r - w1r, axis=1)

                gap1 = np.array([dominance_gap(w) for w in w1r])
                gap2 = np.array([dominance_gap(w) for w in w2r])
                ent1 = np.array([normalized_entropy(w) for w in w1r])
                ent2 = np.array([normalized_entropy(w) for w in w2r])

                weight_rows.append({
                    "policy_id": policy,
                    "topk": topk,
                    "layer_from": l1,
                    "layer_to": l2,
                    "n": int(len(idx)),
                    "weight_shift_mean": float(np.mean(shift)),
                    "weight_shift_std": float(np.std(shift)),
                    "weight_entropy_delta_mean": float(np.mean(ent2 - ent1)),
                    "dominance_gap_delta_mean": float(np.mean(gap2 - gap1)),
                    "basis_rotation_angle_mean_deg": bm["basis_rotation_angle_mean_deg"],
                    "basis_similarity": bm["basis_similarity"],
                    "weight_to_basis_ratio": float(np.mean(shift) / (bm["basis_rotation_angle_mean_deg"] + 1e-6)),
                })

    dfs = {
        "basis": pd.DataFrame(basis_rows),
        "weight": pd.DataFrame(weight_rows),
        "layer_weights": pd.DataFrame(layer_weight_rows),
        "fit_summary": pd.DataFrame(fit_summary_rows),
    }

    dfs["basis"].to_csv(out_dir / "dsta1c_policy_basis_rotation_metrics.csv", index=False, encoding="utf-8-sig")
    dfs["weight"].to_csv(out_dir / "dsta1c_policy_weight_shift_metrics.csv", index=False, encoding="utf-8-sig")
    dfs["layer_weights"].to_csv(out_dir / "dsta1c_layer_policy_weights.csv", index=False, encoding="utf-8-sig")
    dfs["fit_summary"].to_csv(out_dir / "dsta1c_policy_layer_fit_summary.csv", index=False, encoding="utf-8-sig")

    return dfs


def summarize_windows(weight_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    if weight_df.empty:
        return pd.DataFrame()

    rows = []
    for (topk, policy), subp in weight_df.groupby(["topk", "policy_id"]):
        for win_name, layers in CONFIG["WINDOWS"].items():
            layer_set = set(layers)
            sub = subp[subp["layer_from"].isin(layer_set)]
            if len(sub) == 0:
                continue
            rows.append({
                "topk": int(topk),
                "policy_id": policy,
                "window": win_name,
                "n_transitions": int(len(sub)),
                "weight_shift_mean": float(sub["weight_shift_mean"].mean()),
                "weight_shift_std_mean": float(sub["weight_shift_std"].mean()),
                "basis_rotation_angle_mean_deg": float(sub["basis_rotation_angle_mean_deg"].mean()),
                "basis_similarity_mean": float(sub["basis_similarity"].mean()),
                "weight_to_basis_ratio_mean": float(sub["weight_to_basis_ratio"].mean()),
                "dominance_gap_delta_mean": float(sub["dominance_gap_delta_mean"].mean()),
                "weight_entropy_delta_mean": float(sub["weight_entropy_delta_mean"].mean()),
            })

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "dsta1c_policy_basis_vs_weight_window_summary.csv", index=False, encoding="utf-8-sig")
    return out


def make_verdict(window_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    if window_df.empty:
        return pd.DataFrame()

    rows = []
    for topk, subk in window_df.groupby("topk"):
        for policy, subp in subk.groupby("policy_id"):
            row = {
                "topk": int(topk),
                "policy_id": policy,
            }

            for win in CONFIG["WINDOWS"].keys():
                sw = subp[subp["window"] == win]
                if len(sw):
                    row[f"{win}_weight_shift"] = float(sw.iloc[0]["weight_shift_mean"])
                    row[f"{win}_basis_rotation"] = float(sw.iloc[0]["basis_rotation_angle_mean_deg"])
                    row[f"{win}_basis_similarity"] = float(sw.iloc[0]["basis_similarity_mean"])
                    row[f"{win}_ratio"] = float(sw.iloc[0]["weight_to_basis_ratio_mean"])
                    row[f"{win}_gap_delta"] = float(sw.iloc[0]["dominance_gap_delta_mean"])

            crit_rot = row.get("critical_20_22_basis_rotation", np.nan)
            mid_rot = row.get("mid_7_19_basis_rotation", np.nan)
            crit_ws = row.get("critical_20_22_weight_shift", np.nan)
            mid_ws = row.get("mid_7_19_weight_shift", np.nan)
            commit_rot = row.get("commit_23_26_basis_rotation", np.nan)

            flags = []
            if np.isfinite(crit_rot) and np.isfinite(mid_rot) and crit_rot > mid_rot:
                flags.append("critical_basis_rotation_increase")
            if np.isfinite(crit_ws) and np.isfinite(mid_ws) and crit_ws > mid_ws:
                flags.append("critical_weight_shift_increase")
            if np.isfinite(crit_rot) and crit_rot >= CONFIG["ROTATION_HIGH_DEG"]:
                flags.append("high_critical_basis_rotation")
            if np.isfinite(crit_ws) and crit_ws >= CONFIG["WEIGHT_SHIFT_HIGH"]:
                flags.append("high_critical_weight_shift")
            if np.isfinite(commit_rot) and np.isfinite(crit_rot) and commit_rot < crit_rot:
                flags.append("basis_stabilizes_after_critical")

            # type classification
            if ("high_critical_basis_rotation" in flags) and ("high_critical_weight_shift" in flags):
                verdict = "BASIS_AND_WEIGHT_ACTIVE"
            elif "high_critical_basis_rotation" in flags:
                verdict = "BASIS_TRANSPORT_ACTIVE"
            elif "high_critical_weight_shift" in flags:
                verdict = "WEIGHT_TRANSPORT_ACTIVE"
            elif "critical_basis_rotation_increase" in flags:
                verdict = "BASIS_ROTATION_RELATIVE_INCREASE"
            elif "critical_weight_shift_increase" in flags:
                verdict = "WEIGHT_SHIFT_RELATIVE_INCREASE"
            else:
                verdict = "NO_CRITICAL_SPIKE"

            row["flags"] = ";".join(flags)
            row["verdict"] = verdict
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "dsta1c_policy_transport_verdict.csv", index=False, encoding="utf-8-sig")
    return out


def main():
    np.random.seed(CONFIG["RANDOM_SEED"])
    torch.manual_seed(CONFIG["RANDOM_SEED"])

    out_dir = ensure_dir(CONFIG["OUTPUT_DIR"])
    with open(out_dir / "dsta1c_config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)

    input_csv = Path(CONFIG["INPUT_CSV"])
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv.resolve()}\n"
            f"当前脚本已硬编码 INPUT_CSV，请确认文件存在：{CONFIG['INPUT_CSV']}"
        )

    df = pd.read_csv(input_csv)

    prompt_col = autodetect_col(df, ["prompt", "text", "input", "query"], required=True)
    concept_col = autodetect_col(df, ["concept", "topic", "entity", "subject"], required=True)
    policy_col = autodetect_col(df, ["policy_id", "policy", "constraint", "condition", "mechanism"], required=True)
    surface_col = autodetect_col(df, ["surface_form", "surface", "style", "template_family", "template", "prompt_style"], required=False)

    meta = df.copy().reset_index(drop=True)
    if "id" in meta.columns:
        meta["_id"] = meta["id"].astype(str)
    else:
        meta["_id"] = [f"row_{i}" for i in range(len(meta))]
    meta["_prompt"] = meta[prompt_col].astype(str)
    meta["_concept"] = meta[concept_col].astype(str)
    meta["_policy"] = meta[policy_col].astype(str)
    if surface_col is not None:
        meta["_surface"] = meta[surface_col].astype(str)
    else:
        meta["_surface"] = "__nosurface__"

    print("[DATA] rows:", len(meta))
    print("[DATA] prompt_col:", prompt_col)
    print("[DATA] concept_col:", concept_col)
    print("[DATA] policy_col:", policy_col)
    print("[DATA] surface_col:", surface_col)
    print("[DATA] n_concepts:", meta["_concept"].nunique())
    print("[DATA] n_policy:", meta["_policy"].nunique())
    print("[DATA] n_surface:", meta["_surface"].nunique())

    policy_counts = meta["_policy"].value_counts()
    print("[DATA] policy counts:")
    print(policy_counts.to_string())

    meta.to_csv(out_dir / "dsta1c_dataset_resolved.csv", index=False, encoding="utf-8-sig")

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
        meta[["_id", "_concept", "_policy", "_surface"]].reset_index().rename(columns={"index": "row_index"}),
        on="row_index",
        how="left"
    )
    scalar_df.to_csv(out_dir / "dsta1c_scalar_topk_features.csv", index=False, encoding="utf-8-sig")

    available_layers = sorted(set(int(k.split("_")[0][1:]) for k in centers_np.keys()))
    layers = [l for l in CONFIG["LAYER_INDICES"] if l in available_layers]
    print("[INFO] available layers:", layers)

    dfs = run_condition_specific_audit(
        meta=meta,
        centers_np=centers_np,
        scalar_df=scalar_df,
        layers=layers,
        topk_list=CONFIG["TOPK_LIST"],
        out_dir=out_dir,
    )

    window_df = summarize_windows(dfs["weight"], out_dir)
    verdict_df = make_verdict(window_df, out_dir)

    print("\n[DONE] Outputs saved to:", out_dir.resolve())

    print("\n[WINDOW SUMMARY: top rows]")
    if len(window_df):
        print(window_df.head(50).to_string(index=False))
    else:
        print("No window summary produced.")

    print("\n[VERDICT]")
    if len(verdict_df):
        print(verdict_df.to_string(index=False))
    else:
        print("No verdict produced.")

    print("\n[KEY FILES]")
    for name in [
        "dsta1c_policy_basis_rotation_metrics.csv",
        "dsta1c_policy_weight_shift_metrics.csv",
        "dsta1c_policy_basis_vs_weight_window_summary.csv",
        "dsta1c_policy_transport_verdict.csv",
        "dsta1c_layer_policy_weights.csv",
        "dsta1c_policy_layer_fit_summary.csv",
    ]:
        print(" -", out_dir / name)


if __name__ == "__main__":
    main()
