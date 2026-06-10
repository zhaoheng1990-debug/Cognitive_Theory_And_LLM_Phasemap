# -*- coding: utf-8 -*-
r"""
GPT_205_DSTA_1A_direction_spectrum_decomposition.py

DSTA-1A: Direction Spectrum Decomposition Audit

目标：
1. 将浅层/中层/后层 TopK/VIM center trajectory 分解为方向谱：
       Sigma_l = (D_l, W_l)
   其中：
       D_l = direction basis / 局部候选解释方向基
       W_l = direction weights / 方向权重分布

2. 直接比较：
       BasisRotation_l_to_l+1
       WeightShift_l_to_l+1

3. 判断层间传播主要是：
       W_l -> W_{l+1} 权重重分配
   还是：
       D_l -> D_{l+1} 方向基运输 / 旋转

4. 输出 condition / policy_id 级别的方向谱 profile，为后续：
       DSTA-2: WeightGap -> DeltaU
       DA-ASA-1: Direction Weight Steering
   做准备。

硬编码路径：
    MODEL_PATH = D:/model/models--Qwen--Qwen2.5-1.5B-Instruct/main
    INPUT_CSV  = C:/Users/ZH/Desktop/AGI/python_script/sem_dsta_prompts.csv

输入 CSV 推荐列：
    id, concept, policy_id, surface_form, prompt

如果列名不同，脚本会自动从以下候选中识别：
    prompt: prompt / text / input / query
    concept: concept / topic / entity / subject
    policy_id: policy_id / policy / constraint / condition / mechanism
    surface_form: surface_form / surface / style / template_family / template / prompt_style

输出目录：
    dsta1a_outputs/

主要输出：
    dsta1a_layer_sample_weights.csv
    dsta1a_layer_condition_weights.csv
    dsta1a_basis_transport_metrics.csv
    dsta1a_weight_transport_metrics.csv
    dsta1a_layer_summary.csv
    dsta1a_verdict_summary.csv
    dsta1a_config.json
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# =========================
# CONFIG: 已按用户本地环境硬编码
# =========================
CONFIG = {
    "MODEL_PATH": r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main",
    "INPUT_CSV": r"C:\Users\ZH\Desktop\AGI\python_script\sem_dsta_prompts.csv",
    "OUTPUT_DIR": r"dsta1a_outputs",

    # Qwen2.5-1.5B-Instruct 通常 hidden_states[0] 是 embedding output,
    # hidden_states[1] 是 layer0 后，直至 hidden_states[28]。
    # 这里自动按模型实际 hidden_states 长度截断。
    "LAYER_INDICES": list(range(0, 29)),

    # 先以 K=500 为主；K=100 用于稳健性对照。
    "TOPK_LIST": [100, 500],

    # PCA basis 维度。样本少时会自动下调。
    "BASIS_DIM": 8,

    "BATCH_SIZE": 4,
    "MAX_LENGTH": 192,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "DTYPE": "auto",  # auto / float16 / bfloat16 / float32

    "RANDOM_SEED": 42,

    # 计算 direction weights 时使用 squared projection mass
    # weights_i = proj_i^2 / sum_j proj_j^2
    "WEIGHT_EPS": 1e-12,

    # 主要判据窗口
    "WINDOWS": {
        "init_0_6": [0, 1, 2, 3, 4, 5, 6],
        "mid_7_19": list(range(7, 20)),
        "critical_20_22": [20, 21, 22],
        "commit_23_26": [23, 24, 25, 26],
    },
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


def entropy_from_probs(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p / (p.sum() + eps)
    return float(-(p * np.log(p + eps)).sum())


def normalized_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    if len(p) <= 1:
        return 0.0
    return entropy_from_probs(p, eps) / math.log(len(p))


def l2_normalize_np(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


# =========================
# Feature extraction
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
    """
    返回：
    centers_np:
        key = f"L{layer}_K{k}"
        value = [n_samples, hidden_dim] 的 TopK embedding center

    scalar_df:
        每个样本每层每K的 spread/entropy/logit gap 等标量
    """
    W = get_lm_head_weight(model).to(device)
    W_float = W.float()
    W_norm = F.normalize(W_float, dim=-1)
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
# Direction spectrum decomposition
# =========================

def fit_layer_basis(X: np.ndarray, basis_dim: int, seed: int) -> Dict:
    """
    对某层 TopK center matrix X [n,d] 做 PCA，得到方向基 D_l。
    返回：
        mean: [d]
        basis: [r,d], PCA components
        scores: [n,r], centered projection
        explained: [r]
    """
    n, d = X.shape
    r = min(basis_dim, n - 1, d)
    if r < 1:
        raise ValueError(f"Too few samples/features for PCA: X.shape={X.shape}")

    # 对 center 做 L2 normalize 后再中心化，弱化 norm 影响
    Xn = l2_normalize_np(X.astype(np.float32))
    mean = Xn.mean(axis=0, keepdims=True)
    Xc = Xn - mean

    pca = PCA(n_components=r, random_state=seed)
    scores = pca.fit_transform(Xc)
    basis = pca.components_.astype(np.float32)
    explained = pca.explained_variance_ratio_.astype(np.float32)

    return {
        "mean": mean.astype(np.float32),
        "basis": basis,
        "scores": scores.astype(np.float32),
        "explained": explained,
    }


def compute_sample_weights(scores: np.ndarray, eps: float) -> np.ndarray:
    """
    用 squared projection mass 估计方向权重。
    """
    mass = scores ** 2
    denom = mass.sum(axis=1, keepdims=True) + eps
    return (mass / denom).astype(np.float32)


def principal_angle_metrics(B1: np.ndarray, B2: np.ndarray) -> Dict:
    """
    B1/B2: [r,d] orthonormal basis rows from PCA.
    返回平均主角、最大主角、subspace similarity。
    """
    r = min(B1.shape[0], B2.shape[0])
    if r < 1:
        return {"basis_similarity": np.nan, "basis_rotation_angle_mean_deg": np.nan, "basis_rotation_angle_max_deg": np.nan}
    A = B1[:r] @ B2[:r].T
    _, s, _ = np.linalg.svd(A, full_matrices=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(np.abs(s)) * 180.0 / np.pi
    return {
        "basis_similarity": float(np.mean(np.abs(s))),
        "basis_rotation_angle_mean_deg": float(np.mean(angles)),
        "basis_rotation_angle_max_deg": float(np.max(angles)),
    }


def align_weight_dims(w1: np.ndarray, w2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r = min(w1.shape[-1], w2.shape[-1])
    return w1[..., :r], w2[..., :r]


def direction_spectrum_audit(
    meta: pd.DataFrame,
    centers_np: Dict[str, np.ndarray],
    scalar_df: pd.DataFrame,
    layers: List[int],
    topk_list: List[int],
    basis_dim: int,
    seed: int,
    eps: float,
    out_dir: Path,
) -> Dict[str, pd.DataFrame]:

    all_sample_weight_rows = []
    all_condition_rows = []
    all_basis_rows = []
    all_weight_transport_rows = []
    all_layer_summary_rows = []

    for k in topk_list:
        # Fit basis per layer
        layer_fits: Dict[int, Dict] = {}

        for l in layers:
            key = f"L{l}_K{k}"
            if key not in centers_np:
                continue
            X = centers_np[key]
            if X.shape[0] != len(meta):
                continue

            fit = fit_layer_basis(X, basis_dim=basis_dim, seed=seed)
            weights = compute_sample_weights(fit["scores"], eps=eps)
            fit["weights"] = weights
            layer_fits[l] = fit

            # sample-level rows
            for i in range(len(meta)):
                w = weights[i]
                row = {
                    "row_index": i,
                    "id": meta.loc[i, "_id"],
                    "concept": meta.loc[i, "_concept"],
                    "policy_id": meta.loc[i, "_policy"],
                    "surface_form": meta.loc[i, "_surface"],
                    "layer": l,
                    "topk": k,
                    "weight_entropy": normalized_entropy(w),
                    "dominance_gap_w1_w2": float(np.sort(w)[-1] - np.sort(w)[-2]) if len(w) >= 2 else float(w[0]),
                    "top_weight": float(np.max(w)),
                    "basis_dim": int(len(w)),
                }
                for j, val in enumerate(w):
                    row[f"w{j+1}"] = float(val)
                all_sample_weight_rows.append(row)

            # condition / policy mean weights
            tmp = pd.DataFrame({
                "concept": meta["_concept"].values,
                "policy_id": meta["_policy"].values,
                "surface_form": meta["_surface"].values,
            })
            for j in range(weights.shape[1]):
                tmp[f"w{j+1}"] = weights[:, j]

            for policy, sub in tmp.groupby("policy_id"):
                wmean = sub[[c for c in tmp.columns if c.startswith("w")]].mean().values.astype(float)
                row = {
                    "policy_id": policy,
                    "layer": l,
                    "topk": k,
                    "n": int(len(sub)),
                    "weight_entropy_mean": normalized_entropy(wmean),
                    "dominance_gap_mean_w1_w2": float(np.sort(wmean)[-1] - np.sort(wmean)[-2]) if len(wmean) >= 2 else float(wmean[0]),
                    "top_weight_mean": float(np.max(wmean)),
                    "basis_dim": int(len(wmean)),
                }
                for j, val in enumerate(wmean):
                    row[f"mean_w{j+1}"] = float(val)
                all_condition_rows.append(row)

            # layer summary
            explained = fit["explained"]
            scalar_sub = scalar_df[(scalar_df["layer"] == l) & (scalar_df["topk"] == k)]
            all_layer_summary_rows.append({
                "layer": l,
                "topk": k,
                "basis_dim": int(fit["basis"].shape[0]),
                "pc1_explained": float(explained[0]) if len(explained) > 0 else np.nan,
                "pc2_explained": float(explained[1]) if len(explained) > 1 else np.nan,
                "pc3_explained": float(explained[2]) if len(explained) > 2 else np.nan,
                "pc_cum_basis_dim": float(np.sum(explained)),
                "mean_weight_entropy": float(np.mean([normalized_entropy(w) for w in weights])),
                "mean_dominance_gap": float(np.mean([
                    np.sort(w)[-1] - np.sort(w)[-2] if len(w) >= 2 else w[0]
                    for w in weights
                ])),
                "mean_spread": float(scalar_sub["spread"].mean()) if len(scalar_sub) else np.nan,
                "mean_entropy_norm": float(scalar_sub["entropy_norm"].mean()) if len(scalar_sub) else np.nan,
                "mean_logit_gap_1_2": float(scalar_sub["logit_gap_1_2"].mean()) if len(scalar_sub) else np.nan,
            })

        # Basis transport and weight transport between adjacent layers
        sorted_layers = sorted(layer_fits.keys())
        for l1, l2 in zip(sorted_layers[:-1], sorted_layers[1:]):
            f1 = layer_fits[l1]
            f2 = layer_fits[l2]

            # basis rotation
            bm = principal_angle_metrics(f1["basis"], f2["basis"])
            all_basis_rows.append({
                "topk": k,
                "layer_from": l1,
                "layer_to": l2,
                **bm,
            })

            # sample-level weight shift
            w1, w2 = align_weight_dims(f1["weights"], f2["weights"])
            weight_l1_l2 = np.linalg.norm(w2 - w1, axis=1)
            l1_entropy = np.array([normalized_entropy(w) for w in w1])
            l2_entropy = np.array([normalized_entropy(w) for w in w2])
            l1_gap = np.array([np.sort(w)[-1] - np.sort(w)[-2] if w.shape[0] >= 2 else w[0] for w in w1])
            l2_gap = np.array([np.sort(w)[-1] - np.sort(w)[-2] if w.shape[0] >= 2 else w[0] for w in w2])

            all_weight_transport_rows.append({
                "topk": k,
                "layer_from": l1,
                "layer_to": l2,
                "weight_shift_mean": float(weight_l1_l2.mean()),
                "weight_shift_std": float(weight_l1_l2.std()),
                "weight_entropy_delta_mean": float((l2_entropy - l1_entropy).mean()),
                "dominance_gap_delta_mean": float((l2_gap - l1_gap).mean()),
                "basis_rotation_angle_mean_deg": bm["basis_rotation_angle_mean_deg"],
                "basis_similarity": bm["basis_similarity"],
                "weight_to_basis_ratio": float(weight_l1_l2.mean() / (bm["basis_rotation_angle_mean_deg"] + 1e-6)),
            })

            # condition-level weight shift
            tmp = pd.DataFrame({
                "policy_id": meta["_policy"].values,
                "concept": meta["_concept"].values,
                "surface_form": meta["_surface"].values,
                "shift": weight_l1_l2,
                "gap_delta": l2_gap - l1_gap,
                "entropy_delta": l2_entropy - l1_entropy,
            })
            for policy, sub in tmp.groupby("policy_id"):
                all_weight_transport_rows.append({
                    "topk": k,
                    "layer_from": l1,
                    "layer_to": l2,
                    "policy_id": policy,
                    "weight_shift_mean": float(sub["shift"].mean()),
                    "weight_shift_std": float(sub["shift"].std(ddof=0)),
                    "weight_entropy_delta_mean": float(sub["entropy_delta"].mean()),
                    "dominance_gap_delta_mean": float(sub["gap_delta"].mean()),
                    "basis_rotation_angle_mean_deg": bm["basis_rotation_angle_mean_deg"],
                    "basis_similarity": bm["basis_similarity"],
                    "weight_to_basis_ratio": float(sub["shift"].mean() / (bm["basis_rotation_angle_mean_deg"] + 1e-6)),
                })

    result = {
        "sample_weights": pd.DataFrame(all_sample_weight_rows),
        "condition_weights": pd.DataFrame(all_condition_rows),
        "basis_transport": pd.DataFrame(all_basis_rows),
        "weight_transport": pd.DataFrame(all_weight_transport_rows),
        "layer_summary": pd.DataFrame(all_layer_summary_rows),
    }

    result["sample_weights"].to_csv(out_dir / "dsta1a_layer_sample_weights.csv", index=False, encoding="utf-8-sig")
    result["condition_weights"].to_csv(out_dir / "dsta1a_layer_condition_weights.csv", index=False, encoding="utf-8-sig")
    result["basis_transport"].to_csv(out_dir / "dsta1a_basis_transport_metrics.csv", index=False, encoding="utf-8-sig")
    result["weight_transport"].to_csv(out_dir / "dsta1a_weight_transport_metrics.csv", index=False, encoding="utf-8-sig")
    result["layer_summary"].to_csv(out_dir / "dsta1a_layer_summary.csv", index=False, encoding="utf-8-sig")

    return result


def summarize_windows(weight_transport: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    if weight_transport.empty:
        return pd.DataFrame()

    # only global rows without policy_id for primary summary
    wt = weight_transport[weight_transport.get("policy_id").isna()] if "policy_id" in weight_transport.columns else weight_transport
    if wt.empty:
        wt = weight_transport

    for topk, subk in wt.groupby("topk"):
        for win_name, layers in CONFIG["WINDOWS"].items():
            layer_set = set(layers)
            sub = subk[subk["layer_from"].isin(layer_set)]
            if len(sub) == 0:
                continue
            rows.append({
                "topk": topk,
                "window": win_name,
                "n_transitions": int(len(sub)),
                "weight_shift_mean": float(sub["weight_shift_mean"].mean()),
                "basis_rotation_angle_mean_deg": float(sub["basis_rotation_angle_mean_deg"].mean()),
                "basis_similarity_mean": float(sub["basis_similarity"].mean()),
                "weight_to_basis_ratio_mean": float(sub["weight_to_basis_ratio"].mean()),
                "dominance_gap_delta_mean": float(sub["dominance_gap_delta_mean"].mean()),
                "weight_entropy_delta_mean": float(sub["weight_entropy_delta_mean"].mean()),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "dsta1a_window_summary.csv", index=False, encoding="utf-8-sig")
    return df


def make_verdict(window_summary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for topk, subk in window_summary.groupby("topk"):
        mid = subk[subk["window"] == "mid_7_19"]
        crit = subk[subk["window"] == "critical_20_22"]
        init = subk[subk["window"] == "init_0_6"]

        if len(mid):
            r = float(mid.iloc[0]["weight_to_basis_ratio_mean"])
            rows.append({
                "topk": topk,
                "claim": "Mid-layer propagation is dominated by weight redistribution if WeightShift >> BasisRotation.",
                "window": "mid_7_19",
                "weight_to_basis_ratio_mean": r,
                "verdict": "PASS_WEIGHT_DOMINANT" if r > 0.02 else "INCONCLUSIVE_OR_BASIS_DOMINANT",
                "note": "ratio uses weight_shift / angle_degree; threshold is heuristic, inspect curves."
            })
        if len(crit) and len(mid):
            crit_rot = float(crit.iloc[0]["basis_rotation_angle_mean_deg"])
            mid_rot = float(mid.iloc[0]["basis_rotation_angle_mean_deg"])
            rows.append({
                "topk": topk,
                "claim": "Critical window has enhanced basis rotation relative to mid-layer.",
                "window": "critical_20_22",
                "critical_rotation": crit_rot,
                "mid_rotation": mid_rot,
                "verdict": "BASIS_ROTATION_INCREASED" if crit_rot > mid_rot else "NO_ROTATION_INCREASE",
                "note": "If increased especially in closure/override subsets, direction-basis transport is implicated."
            })
        if len(init):
            rows.append({
                "topk": topk,
                "claim": "Initialization window direction spectrum is measurable.",
                "window": "init_0_6",
                "weight_shift_mean": float(init.iloc[0]["weight_shift_mean"]),
                "basis_rotation_angle_mean_deg": float(init.iloc[0]["basis_rotation_angle_mean_deg"]),
                "verdict": "MEASURED",
                "note": "Use condition-level file to inspect PolicyID -> Sigma0."
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "dsta1a_verdict_summary.csv", index=False, encoding="utf-8-sig")
    return df


def main():
    np.random.seed(CONFIG["RANDOM_SEED"])
    torch.manual_seed(CONFIG["RANDOM_SEED"])

    out_dir = ensure_dir(CONFIG["OUTPUT_DIR"])
    with open(out_dir / "dsta1a_config.json", "w", encoding="utf-8") as f:
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

    meta.to_csv(out_dir / "dsta1a_dataset_resolved.csv", index=False, encoding="utf-8-sig")

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
    scalar_df.to_csv(out_dir / "dsta1a_scalar_topk_features.csv", index=False, encoding="utf-8-sig")

    # actual layers available
    available_layers = sorted(set(int(k.split("_")[0][1:]) for k in centers_np.keys()))
    layers = [l for l in CONFIG["LAYER_INDICES"] if l in available_layers]
    print("[INFO] available layers:", layers)

    result = direction_spectrum_audit(
        meta=meta,
        centers_np=centers_np,
        scalar_df=scalar_df,
        layers=layers,
        topk_list=CONFIG["TOPK_LIST"],
        basis_dim=CONFIG["BASIS_DIM"],
        seed=CONFIG["RANDOM_SEED"],
        eps=CONFIG["WEIGHT_EPS"],
        out_dir=out_dir,
    )

    window_summary = summarize_windows(result["weight_transport"], out_dir)
    verdict = make_verdict(window_summary, out_dir)

    print("\n[DONE] Outputs saved to:", out_dir.resolve())
    print("\n[WINDOW SUMMARY]")
    if len(window_summary):
        print(window_summary.to_string(index=False))
    else:
        print("No window summary produced.")

    print("\n[VERDICT]")
    if len(verdict):
        print(verdict.to_string(index=False))
    else:
        print("No verdict produced.")

    print("\n[KEY FILES]")
    for name in [
        "dsta1a_layer_sample_weights.csv",
        "dsta1a_layer_condition_weights.csv",
        "dsta1a_basis_transport_metrics.csv",
        "dsta1a_weight_transport_metrics.csv",
        "dsta1a_window_summary.csv",
        "dsta1a_verdict_summary.csv",
    ]:
        print(" -", out_dir / name)


if __name__ == "__main__":
    main()
