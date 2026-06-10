# -*- coding: utf-8 -*-
"""
GPT_200_SEM_DSTA_1_policyid_to_sigma0_audit.py

SEM-DSTA-1: PolicyID -> Sigma0 Initialization Audit

目标：
1. 验证 PolicyID / Constraint 是否能从浅层 TopK/VIM 初始化特征中被读出；
2. 检查 SurfaceForm 是否遮蔽 PolicyID 信号；
3. 比较 raw / concept quotient / style quotient / concept+style additive residual；
4. 检查 PolicyID 低秩结构是否在 quotient 后仍保持；
5. 为后续 DSTA-1A: Sigma_l=(D_l,W_l) 分解提供输入。

默认假设输入 CSV 至少包含：
    prompt
并尽量包含：
    concept / topic / entity
    policy_id / constraint / condition
    surface_form / style / template_family

推荐列名：
    id, concept, policy_id, surface_form, prompt

输出：
    sem_dsta1_outputs/
        sem_dsta1_features_raw.csv
        sem_dsta1_eval_summary.csv
        sem_dsta1_lowrank_summary.csv
        sem_dsta1_config.json
        sem_dsta1_feature_matrix.npz

注意：
- 本脚本不依赖命令行即可运行；如需修改路径，直接改 CONFIG。
- 为避免 sklearn 版本报错，不使用 LogisticRegression(multi_class=...) 参数。
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

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, GroupKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, r2_score
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# =========================
# CONFIG: 按需直接修改
# =========================
CONFIG = {
    # 用户本地 Qwen 默认路径；如不同，直接修改这里
    "MODEL_PATH": r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main",

    # SEM-2A.3 或任意含 prompt/concept/policy/surface 的 CSV
    # 建议把待审计 CSV 放到脚本同目录，并命名为 sem_dsta_prompts.csv
    "INPUT_CSV": r"C:\Users\ZH\Desktop\AGI\python_script\sem_dsta_prompts.csv",

    "OUTPUT_DIR": r"sem_dsta1_outputs",

    # 浅层初始化窗口：hidden_states[0] 是 embedding output；hidden_states[1] 是 block0 后
    # 这里保留 H_emb, H0...H6 的常用读法
    "INIT_HIDDEN_STATE_INDICES": list(range(0, 8)),

    # TopK 列表。500 已经较稳；1000 更重。
    "TOPK_LIST": [100, 500],

    "BATCH_SIZE": 4,
    "MAX_LENGTH": 192,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "DTYPE": "auto",  # auto / float16 / bfloat16 / float32

    # PCA 维度
    "CENTER_PCA_DIM": 32,

    # CV 设置
    "N_SPLITS": 5,
    "RANDOM_SEED": 42,

    # clean policy 名称候选，用于 clean-relative delta；找不到则跳过 clean-delta
    "CLEAN_POLICY_CANDIDATES": [
        "clean", "stable", "stable_clean", "stable_clean_basic",
        "clean_closure", "base", "baseline"
    ],
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
    # fuzzy contains
    for c in df.columns:
        cl = c.lower()
        for name in candidates:
            if name.lower() in cl:
                return c
    if required:
        raise ValueError(f"Cannot find required column among candidates={candidates}. Existing columns={list(df.columns)}")
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
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[LOAD] model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **kwargs)
    model.eval()
    model.to(device)
    return model, tokenizer


def get_lm_head_weight(model) -> torch.Tensor:
    emb = model.get_output_embeddings()
    if emb is None:
        raise RuntimeError("model.get_output_embeddings() returned None; cannot compute H_l W^T.")
    W = emb.weight.detach()
    return W


@torch.no_grad()
def extract_topk_features(
    model,
    tokenizer,
    prompts: List[str],
    layers: List[int],
    topk_list: List[int],
    batch_size: int,
    max_length: int,
    device: str,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    返回：
    - scalar dataframe: 每个样本一行，包含各层各 K 的标量特征
    - center vectors dict: key=f"L{layer}_K{k}", value=[n,d] center matrix
    """
    W = get_lm_head_weight(model).to(device)
    W_norm = F.normalize(W.float(), dim=-1)
    vocab_size, dim = W.shape
    print(f"[INFO] lm_head W shape: vocab={vocab_size}, dim={dim}")

    rows = []
    centers: Dict[str, List[np.ndarray]] = {f"L{l}_K{k}": [] for l in layers for k in topk_list}

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

        for bi in range(input_ids.size(0)):
            sample_row = {"row_index": start + bi}
            pos = int(last_pos[bi].item())

            for layer_idx in layers:
                if layer_idx >= len(hidden_states):
                    continue

                h = hidden_states[layer_idx][bi, pos, :].float()  # [d]
                logits = torch.matmul(h, W.float().T)  # [vocab]

                for k in topk_list:
                    vals, ids = torch.topk(logits, k=min(k, logits.numel()))
                    emb = W.float()[ids]  # [k,d]
                    center = emb.mean(dim=0)
                    center_norm = F.normalize(center[None, :], dim=-1)[0]
                    emb_norm = F.normalize(emb, dim=-1)
                    cos_to_center = torch.matmul(emb_norm, center_norm)
                    spread = (1.0 - cos_to_center).mean().item()

                    probs = torch.softmax(vals.float(), dim=0)
                    entropy = (-(probs * torch.log(probs + 1e-12)).sum()).item()
                    entropy_norm = entropy / math.log(len(vals) + 1e-12)

                    vals_f = vals.float()
                    key_prefix = f"L{layer_idx}_K{k}"
                    sample_row[f"{key_prefix}_logit_mean"] = vals_f.mean().item()
                    sample_row[f"{key_prefix}_logit_std"] = vals_f.std(unbiased=False).item()
                    sample_row[f"{key_prefix}_logit_top1"] = vals_f[0].item()
                    sample_row[f"{key_prefix}_logit_gap_1_2"] = (vals_f[0] - vals_f[1]).item() if len(vals_f) > 1 else 0.0
                    sample_row[f"{key_prefix}_logit_gap_1_mean"] = (vals_f[0] - vals_f.mean()).item()
                    sample_row[f"{key_prefix}_entropy"] = entropy
                    sample_row[f"{key_prefix}_entropy_norm"] = entropy_norm
                    sample_row[f"{key_prefix}_spread"] = spread
                    sample_row[f"{key_prefix}_center_norm"] = center.norm().item()
                    sample_row[f"{key_prefix}_top_ids_head"] = " ".join(map(str, ids[:20].detach().cpu().tolist()))

                    centers[key_prefix].append(center.detach().cpu().numpy().astype(np.float32))

            rows.append(sample_row)

        print(f"[EXTRACT] {min(start + batch_size, len(prompts))}/{len(prompts)}")

    scalar_df = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
    centers_np = {k: np.vstack(v) if len(v) else np.zeros((len(prompts), 1), dtype=np.float32)
                  for k, v in centers.items()}
    return scalar_df, centers_np


def build_feature_matrices(
    meta: pd.DataFrame,
    scalar_df: pd.DataFrame,
    centers_np: Dict[str, np.ndarray],
    concept_col: str,
    policy_col: str,
    surface_col: Optional[str],
    out_dir: Path,
    pca_dim: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    # scalar numeric features
    scalar_cols = [c for c in scalar_df.columns if c not in ["row_index"] and not c.endswith("_top_ids_head")]
    X_scalar = scalar_df[scalar_cols].astype(float).fillna(0.0).values

    # flatten all center vectors and PCA
    center_keys = sorted(centers_np.keys())
    X_centers = np.concatenate([centers_np[k] for k in center_keys], axis=1)
    n_comp = min(pca_dim, X_centers.shape[0] - 1, X_centers.shape[1])
    if n_comp >= 2:
        pca = PCA(n_components=n_comp, random_state=seed)
        X_center_pca = pca.fit_transform(StandardScaler().fit_transform(X_centers))
        pca_var = pca.explained_variance_ratio_
    else:
        X_center_pca = X_centers
        pca_var = np.array([1.0])

    # save PCA variance
    pd.DataFrame({
        "pc": np.arange(1, len(pca_var) + 1),
        "explained_variance_ratio": pca_var,
        "cum_explained": np.cumsum(pca_var),
    }).to_csv(out_dir / "sem_dsta1_center_pca_variance.csv", index=False, encoding="utf-8-sig")

    X_raw = np.concatenate([X_scalar, X_center_pca], axis=1)

    # quotient transforms
    concept = meta[concept_col].astype(str).values
    policy = meta[policy_col].astype(str).values
    if surface_col is not None:
        surface = meta[surface_col].astype(str).values
    else:
        surface = np.array(["__nosurface__"] * len(meta))

    def subtract_group_mean(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
        Xq = X.copy().astype(np.float32)
        global_mean = X.mean(axis=0, keepdims=True)
        for g in np.unique(groups):
            idx = (groups == g)
            Xq[idx] = X[idx] - X[idx].mean(axis=0, keepdims=True)
        # keep centered residual; do not add global mean for classifier
        return Xq

    def additive_residual(X: np.ndarray, g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
        # X - mean(concept) - mean(surface) + global
        Xr = X.copy().astype(np.float32)
        global_mean = X.mean(axis=0, keepdims=True)
        m1 = {}
        m2 = {}
        for g in np.unique(g1):
            m1[g] = X[g1 == g].mean(axis=0, keepdims=True)
        for g in np.unique(g2):
            m2[g] = X[g2 == g].mean(axis=0, keepdims=True)
        for i in range(X.shape[0]):
            Xr[i] = X[i] - m1[g1[i]][0] - m2[g2[i]][0] + global_mean[0]
        return Xr

    matrices = {
        "raw": X_raw.astype(np.float32),
        "scalar_only": X_scalar.astype(np.float32),
        "center_pca_only": X_center_pca.astype(np.float32),
        "concept_quotient": subtract_group_mean(X_raw, concept),
        "style_quotient": subtract_group_mean(X_raw, surface),
        "concept_style_additive_residual": additive_residual(X_raw, concept, surface),
    }

    # Optional clean-relative delta at raw feature level if clean policy exists
    clean_candidates = set([s.lower() for s in CONFIG["CLEAN_POLICY_CANDIDATES"]])
    policy_lower = np.array([p.lower() for p in policy])
    clean_mask = np.array([p in clean_candidates for p in policy_lower])

    if clean_mask.any():
        # baseline by concept+surface; fallback concept; fallback global clean
        X_delta = X_raw.copy().astype(np.float32)
        global_clean = X_raw[clean_mask].mean(axis=0, keepdims=True)
        for i in range(X_raw.shape[0]):
            c = concept[i]
            s = surface[i]
            idx = clean_mask & (concept == c) & (surface == s)
            if idx.sum() == 0:
                idx = clean_mask & (concept == c)
            base = X_raw[idx].mean(axis=0, keepdims=True) if idx.sum() > 0 else global_clean
            X_delta[i] = X_raw[i] - base[0]
        matrices["clean_relative_delta"] = X_delta

    # Save feature matrix
    np.savez_compressed(
        out_dir / "sem_dsta1_feature_matrix.npz",
        **matrices,
        y_policy=policy,
        concept=concept,
        surface=surface,
    )

    # Save feature df raw
    raw_df = meta.copy()
    for j in range(X_raw.shape[1]):
        raw_df[f"feat_{j:04d}"] = X_raw[:, j]
    raw_df.to_csv(out_dir / "sem_dsta1_features_raw.csv", index=False, encoding="utf-8-sig")

    return matrices


def safe_cv_splits(y: np.ndarray, n_splits: int):
    # make sure each class has enough samples
    _, counts = np.unique(y, return_counts=True)
    max_splits = int(counts.min())
    return max(2, min(n_splits, max_splits))


def eval_policy_classification(
    X: np.ndarray,
    y_str: np.ndarray,
    concept: np.ndarray,
    surface: np.ndarray,
    feature_set: str,
    n_splits: int,
    seed: int,
) -> List[Dict]:
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    rows = []

    def fit_predict(train_idx, test_idx):
        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs")
        )
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        return pred

    # Stratified CV
    try:
        k = safe_cv_splits(y, n_splits)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        preds = np.full_like(y, fill_value=-1)
        for tr, te in skf.split(X, y):
            preds[te] = fit_predict(tr, te)
        rows.append(metric_row("stratified_cv", feature_set, y, preds))
    except Exception as e:
        rows.append(error_row("stratified_cv", feature_set, str(e)))

    # Leave-one-concept-out
    if len(np.unique(concept)) >= 2:
        try:
            logo = LeaveOneGroupOut()
            ys, ps = [], []
            for tr, te in logo.split(X, y, groups=concept):
                # skip if train lacks some class? Logistic can still train if >=2 classes
                if len(np.unique(y[tr])) < 2:
                    continue
                pred = fit_predict(tr, te)
                ys.append(y[te])
                ps.append(pred)
            if ys:
                yy = np.concatenate(ys)
                pp = np.concatenate(ps)
                rows.append(metric_row("leave_one_concept_out", feature_set, yy, pp))
        except Exception as e:
            rows.append(error_row("leave_one_concept_out", feature_set, str(e)))

    # Leave-one-surface-out
    if len(np.unique(surface)) >= 2 and not np.all(surface == "__nosurface__"):
        try:
            logo = LeaveOneGroupOut()
            ys, ps = [], []
            for tr, te in logo.split(X, y, groups=surface):
                if len(np.unique(y[tr])) < 2:
                    continue
                pred = fit_predict(tr, te)
                ys.append(y[te])
                ps.append(pred)
            if ys:
                yy = np.concatenate(ys)
                pp = np.concatenate(ps)
                rows.append(metric_row("leave_one_surface_out", feature_set, yy, pp))
        except Exception as e:
            rows.append(error_row("leave_one_surface_out", feature_set, str(e)))

    # GroupKFold by concept
    if len(np.unique(concept)) >= 3:
        try:
            gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(concept))))
            ys, ps = [], []
            for tr, te in gkf.split(X, y, groups=concept):
                if len(np.unique(y[tr])) < 2:
                    continue
                pred = fit_predict(tr, te)
                ys.append(y[te])
                ps.append(pred)
            if ys:
                yy = np.concatenate(ys)
                pp = np.concatenate(ps)
                rows.append(metric_row("groupkfold_concept", feature_set, yy, pp))
        except Exception as e:
            rows.append(error_row("groupkfold_concept", feature_set, str(e)))

    return rows


def metric_row(split_name: str, feature_set: str, y: np.ndarray, pred: np.ndarray) -> Dict:
    return {
        "split": split_name,
        "feature_set": feature_set,
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
        "error": "",
    }


def error_row(split_name: str, feature_set: str, err: str) -> Dict:
    return {
        "split": split_name,
        "feature_set": feature_set,
        "n": 0,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "macro_f1": np.nan,
        "weighted_f1": np.nan,
        "error": err[:500],
    }


def lowrank_audit(matrices: Dict[str, np.ndarray], out_dir: Path) -> pd.DataFrame:
    rows = []
    for name, X in matrices.items():
        Xs = StandardScaler().fit_transform(X)
        n_comp = min(Xs.shape[0] - 1, Xs.shape[1], 128)
        if n_comp < 2:
            continue
        pca = PCA(n_components=n_comp, random_state=CONFIG["RANDOM_SEED"])
        pca.fit(Xs)
        cum = np.cumsum(pca.explained_variance_ratio_)
        rank50 = int(np.searchsorted(cum, 0.50) + 1)
        rank80 = int(np.searchsorted(cum, 0.80) + 1)
        rank90 = int(np.searchsorted(cum, 0.90) + 1)
        rank95 = int(np.searchsorted(cum, 0.95) + 1)
        rows.append({
            "feature_set": name,
            "n_features": int(X.shape[1]),
            "rank50": rank50,
            "rank80": rank80,
            "rank90": rank90,
            "rank95": rank95,
            "pc1": float(pca.explained_variance_ratio_[0]),
            "pc2": float(pca.explained_variance_ratio_[1]) if len(pca.explained_variance_ratio_) > 1 else np.nan,
            "pc3": float(pca.explained_variance_ratio_[2]) if len(pca.explained_variance_ratio_) > 2 else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sem_dsta1_lowrank_summary.csv", index=False, encoding="utf-8-sig")
    return df


def main():
    np.random.seed(CONFIG["RANDOM_SEED"])
    torch.manual_seed(CONFIG["RANDOM_SEED"])

    out_dir = ensure_dir(CONFIG["OUTPUT_DIR"])
    with open(out_dir / "sem_dsta1_config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)

    input_csv = Path(CONFIG["INPUT_CSV"])
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv.resolve()}\n"
            f"请把 SEM-2A.3 / prompts CSV 放到脚本同目录并命名为 sem_dsta_prompts.csv，"
            f"或直接修改 CONFIG['INPUT_CSV']。"
        )

    df = pd.read_csv(input_csv)
    prompt_col = autodetect_col(df, ["prompt", "text", "input", "query"], required=True)
    concept_col = autodetect_col(df, ["concept", "topic", "entity", "subject"], required=True)
    policy_col = autodetect_col(df, ["policy_id", "policy", "constraint", "condition", "mechanism"], required=True)
    surface_col = autodetect_col(df, ["surface_form", "surface", "style", "template_family", "template", "prompt_style"], required=False)

    meta = df.copy().reset_index(drop=True)
    meta["_prompt"] = meta[prompt_col].astype(str)
    meta["_concept"] = meta[concept_col].astype(str)
    meta["_policy"] = meta[policy_col].astype(str)
    if surface_col is not None:
        meta["_surface"] = meta[surface_col].astype(str)
    else:
        meta["_surface"] = "__nosurface__"

    print("[DATA] rows:", len(meta))
    print("[DATA] prompt_col:", prompt_col, "concept_col:", concept_col, "policy_col:", policy_col, "surface_col:", surface_col)
    print("[DATA] n_concepts:", meta["_concept"].nunique(), "n_policy:", meta["_policy"].nunique(), "n_surface:", meta["_surface"].nunique())

    model, tokenizer = load_model_and_tokenizer(CONFIG["MODEL_PATH"], CONFIG["DEVICE"], CONFIG["DTYPE"])

    scalar_df, centers_np = extract_topk_features(
        model=model,
        tokenizer=tokenizer,
        prompts=meta["_prompt"].tolist(),
        layers=CONFIG["INIT_HIDDEN_STATE_INDICES"],
        topk_list=CONFIG["TOPK_LIST"],
        batch_size=CONFIG["BATCH_SIZE"],
        max_length=CONFIG["MAX_LENGTH"],
        device=CONFIG["DEVICE"],
    )

    # Save scalar + metadata
    scalar_out = pd.concat([meta.reset_index(drop=True), scalar_df.reset_index(drop=True)], axis=1)
    scalar_out.to_csv(out_dir / "sem_dsta1_scalar_topk_features.csv", index=False, encoding="utf-8-sig")

    matrices = build_feature_matrices(
        meta=meta,
        scalar_df=scalar_df,
        centers_np=centers_np,
        concept_col="_concept",
        policy_col="_policy",
        surface_col="_surface",
        out_dir=out_dir,
        pca_dim=CONFIG["CENTER_PCA_DIM"],
        seed=CONFIG["RANDOM_SEED"],
    )

    # Low-rank audit
    lowrank_df = lowrank_audit(matrices, out_dir)

    # PolicyID classification audit
    eval_rows = []
    y_policy = meta["_policy"].astype(str).values
    concept = meta["_concept"].astype(str).values
    surface = meta["_surface"].astype(str).values

    for name, X in matrices.items():
        print(f"[EVAL] feature_set={name}, shape={X.shape}")
        eval_rows.extend(eval_policy_classification(
            X=X,
            y_str=y_policy,
            concept=concept,
            surface=surface,
            feature_set=name,
            n_splits=CONFIG["N_SPLITS"],
            seed=CONFIG["RANDOM_SEED"],
        ))

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(out_dir / "sem_dsta1_eval_summary.csv", index=False, encoding="utf-8-sig")

    # Simple verdict
    verdict = []
    def get_metric(split, feature, metric):
        sub = eval_df[(eval_df["split"] == split) & (eval_df["feature_set"] == feature)]
        if len(sub) == 0:
            return np.nan
        return float(sub.iloc[0][metric])

    raw_surface_f1 = get_metric("leave_one_surface_out", "raw", "macro_f1")
    styleq_surface_f1 = get_metric("leave_one_surface_out", "style_quotient", "macro_f1")
    resid_surface_f1 = get_metric("leave_one_surface_out", "concept_style_additive_residual", "macro_f1")

    verdict.append({
        "question": "Does style_quotient improve leave-one-surface PolicyID recovery over raw?",
        "raw_macro_f1": raw_surface_f1,
        "style_quotient_macro_f1": styleq_surface_f1,
        "improvement": styleq_surface_f1 - raw_surface_f1 if np.isfinite(raw_surface_f1) and np.isfinite(styleq_surface_f1) else np.nan,
        "pass": bool(np.isfinite(raw_surface_f1) and np.isfinite(styleq_surface_f1) and styleq_surface_f1 > raw_surface_f1),
    })
    verdict.append({
        "question": "Does concept_style_additive_residual improve leave-one-surface PolicyID recovery over raw?",
        "raw_macro_f1": raw_surface_f1,
        "residual_macro_f1": resid_surface_f1,
        "improvement": resid_surface_f1 - raw_surface_f1 if np.isfinite(raw_surface_f1) and np.isfinite(resid_surface_f1) else np.nan,
        "pass": bool(np.isfinite(raw_surface_f1) and np.isfinite(resid_surface_f1) and resid_surface_f1 > raw_surface_f1),
    })

    verdict_df = pd.DataFrame(verdict)
    verdict_df.to_csv(out_dir / "sem_dsta1_verdict_summary.csv", index=False, encoding="utf-8-sig")

    print("\n[DONE] Outputs saved to:", out_dir.resolve())
    print("\n[EVAL SUMMARY]")
    print(eval_df.sort_values(["split", "macro_f1"], ascending=[True, False]).to_string(index=False))
    print("\n[LOWRANK SUMMARY]")
    print(lowrank_df.to_string(index=False))
    print("\n[VERDICT]")
    print(verdict_df.to_string(index=False))


if __name__ == "__main__":
    main()
