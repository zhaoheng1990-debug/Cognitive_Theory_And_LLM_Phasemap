# -*- coding: utf-8 -*-
r"""
GPT_210_DA_ASA_1_attractor_steering_audit.py

DA-ASA-1: Attractor Steering Audit

目标：
DSTA-2B 证明 closure_override / exception / hallucination-like 不是简单方向反向，
而是落入不同 drift attractor。DA-ASA-1 进一步做因果验证：

    能否把样本从 wrong-closure / override-like attractor
    拉回 stable / update safe attractor？

核心假设：
    A_l^{wrong} -> A_l^{safe}

而不是简单：
    w_wrong down

实验策略：
1. 自动构造 closure/update/override/exception/hallucination-like prompts；
2. 基线 forward，收集 target layers 的 hidden state；
3. 构造 attractor steering directions：
       v_to_stable_l = mean_H(stable_clean)_l - mean_H(wrong_target)_l
       v_to_update_l = mean_H(update_safe)_l  - mean_H(wrong_target)_l
4. 在 L15-L19 对 last-token hidden state 加 hook：
       h_l' = h_l + alpha * scale * normalize(v_l)
5. 观测 final C/E logit margin 是否向 clean label 回拉：
       R_final = logit(C_clean_first_token)-logit(E_conflict_first_token)
6. 比较 target vs non-target specificity。

注意：
- 这是 trajectory-level / margin-level steering，不宣称 generation-level 完全控制。
- 当前为了稳健和简单，C/E 使用 first-token proxy；脚本会输出 candidate tokenization 供审计。
- 自动数据使用简单颜色 label，尽量降低多 token label 噪声。

硬编码模型路径：
    D:/model/models--Qwen--Qwen2.5-1.5B-Instruct/main

输出目录：
    da_asa1_outputs/

关键输出：
    da_asa1_generated_prompts.csv
    da_asa1_candidate_tokenization.csv
    da_asa1_baseline_scores.csv
    da_asa1_steering_results.csv
    da_asa1_specificity_summary.csv
    da_asa1_verdict_summary.csv
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

warnings.filterwarnings("ignore")


CONFIG = {
    "MODEL_PATH": r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main",
    "MODEL_PATH_FALLBACK": r"D:\model\models--Qwen--Qwen2.5B-Instruct\main",

    "OUTPUT_DIR": r"da_asa1_outputs",

    # Qwen decoder layer indices, 0-based
    "STEER_LAYERS": [15, 16, 17, 18, 19],
    "COLLECT_LAYERS": [15, 16, 17, 18, 19],

    "ALPHAS": [0.0, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30],

    # steering modes
    "STEERING_MODES": ["to_stable", "to_update"],

    # target conditions for wrong attractor
    "TARGET_CONDITIONS": ["closure_override", "closure_exception", "closure_negation", "hallucination_like"],
    "WRONG_ATTRACTOR_CONDITIONS": ["closure_override", "hallucination_like"],
    "SAFE_STABLE_CONDITIONS": ["stable_clean", "stable_redundant"],
    "SAFE_UPDATE_CONDITIONS": ["closure_update", "closure_temporal", "closure_authority"],

    # non-target controls
    "NONTARGET_CONDITIONS": ["stable_clean", "stable_redundant", "weak_distractor", "competition_direct"],

    "BATCH_SIZE": 4,
    "MAX_LENGTH": 224,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "DTYPE": "auto",
    "RANDOM_SEED": 42,

    # intervention scaling
    # actual addition: alpha * mean_hidden_norm[layer] * unit_direction[layer]
    "USE_MEAN_HIDDEN_NORM_SCALE": True,
}


# =========================
# Dataset
# =========================

def build_prompt_dataset() -> pd.DataFrame:
    """
    构造类似 DSTA-2A 的 closure 数据，但 C/E 使用简单颜色标签，降低 tokenization 噪声。
    """
    pairs = [
        ("ItemA", "GroupA", "Red", "Blue"),
        ("ItemB", "GroupB", "Green", "Yellow"),
        ("ItemC", "GroupC", "North", "South"),
        ("ItemD", "GroupD", "East", "West"),
        ("ItemE", "GroupE", "Silver", "Gold"),
        ("ItemF", "GroupF", "Circle", "Square"),
        ("ItemG", "GroupG", "Spring", "Winter"),
        ("ItemH", "GroupH", "Ocean", "Desert"),
        ("ItemI", "GroupI", "Copper", "Iron"),
        ("ItemJ", "GroupJ", "Forest", "Mountain"),
        ("ItemK", "GroupK", "River", "Island"),
        ("ItemL", "GroupL", "Cloud", "Stone"),
    ]

    templates = []
    def add(condition, mechanism, risk, template):
        templates.append({"condition": condition, "mechanism": mechanism, "risk_regime": risk, "template": template})

    add("stable_clean", "stable", "clean_basin",
        "{A} belongs to {B}. {B} is associated with {C}. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")
    add("stable_redundant", "stable", "clean_basin",
        "{A} belongs to {B}. {B} is associated with {C}. Members of {B} normally map to {C}. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")
    add("weak_distractor", "stable", "low_risk_shift",
        "{A} belongs to {B}. {B} is associated with {C}. Unrelated note: another item is associated with {E}. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")

    add("competition_direct", "competition", "competition_low_risk",
        "{A} belongs to {B}. {B} is associated with {C}. However, one note says {A} is associated with {E}. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")
    add("competition_equal_evidence", "competition", "competition_high_risk",
        "Evidence 1: {A} belongs to {B}, and {B} is associated with {C}. Evidence 2: {A} is associated with {E}. Both evidence sources are equally reliable. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")
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

    rows = []
    sid = 0
    for A, B, C, E in pairs:
        for t in templates:
            prompt = t["template"].format(A=A, B=B, C=C, E=E)
            rows.append({
                "id": f"daasa1_{sid:05d}",
                "graph_id": f"{A}_{B}_{C}_{E}",
                "A": A, "B": B, "C_clean": C, "E_conflict": E,
                "concept": A,
                "condition": t["condition"],
                "mechanism": t["mechanism"],
                "risk_regime": t["risk_regime"],
                "prompt": prompt,
            })
            sid += 1

    return pd.DataFrame(rows)


# =========================
# Utilities
# =========================

def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


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


def get_decoder_layers(model):
    """
    Qwen/Llama/Gemma common locations.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise AttributeError("Cannot locate decoder layers. Please adapt get_decoder_layers().")


def first_token_id(tokenizer, text: str) -> int:
    """
    Use first token of candidate label as proxy.
    Try with leading space first, then without.
    """
    ids = tokenizer.encode(" " + text, add_special_tokens=False)
    if len(ids) == 0:
        ids = tokenizer.encode(text, add_special_tokens=False)
    return int(ids[0])


def tokenize_candidates(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    rows = []
    for _, r in df.drop_duplicates(["C_clean", "E_conflict"]).iterrows():
        C, E = str(r["C_clean"]), str(r["E_conflict"])
        c_ids_space = tokenizer.encode(" " + C, add_special_tokens=False)
        e_ids_space = tokenizer.encode(" " + E, add_special_tokens=False)
        rows.append({
            "C_clean": C,
            "E_conflict": E,
            "C_first_token_id": first_token_id(tokenizer, C),
            "E_first_token_id": first_token_id(tokenizer, E),
            "C_ids_with_space": " ".join(map(str, c_ids_space)),
            "E_ids_with_space": " ".join(map(str, e_ids_space)),
            "C_n_tokens": len(c_ids_space),
            "E_n_tokens": len(e_ids_space),
        })
    return pd.DataFrame(rows)


# =========================
# Baseline collection
# =========================

@torch.no_grad()
def forward_collect_hidden_and_scores(model, tokenizer, df: pd.DataFrame, layers: List[int], batch_size: int, max_length: int, device: str):
    """
    Collect hidden vectors at decoder layers and final first-token C/E margin.
    hidden layer index here is decoder layer index, so hidden_states[layer+1].
    """
    all_hidden = {l: [] for l in layers}
    score_rows = []

    prompts = df["prompt"].astype(str).tolist()

    for start in range(0, len(prompts), batch_size):
        batch = df.iloc[start:start+batch_size].reset_index(drop=True)
        enc = tokenizer(
            batch["prompt"].astype(str).tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        last_pos = attn.sum(dim=1) - 1

        out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
        logits = out.logits  # [b,seq,v]
        hidden_states = out.hidden_states

        for bi in range(input_ids.size(0)):
            pos = int(last_pos[bi].item())
            C = str(batch.loc[bi, "C_clean"])
            E = str(batch.loc[bi, "E_conflict"])
            c_id = first_token_id(tokenizer, C)
            e_id = first_token_id(tokenizer, E)

            final_logits = logits[bi, pos, :].float()
            c_logit = float(final_logits[c_id].item())
            e_logit = float(final_logits[e_id].item())
            R = c_logit - e_logit

            score_rows.append({
                "row_index": start + bi,
                "id": batch.loc[bi, "id"],
                "graph_id": batch.loc[bi, "graph_id"],
                "condition": batch.loc[bi, "condition"],
                "mechanism": batch.loc[bi, "mechanism"],
                "risk_regime": batch.loc[bi, "risk_regime"],
                "C_clean": C,
                "E_conflict": E,
                "C_first_token_id": c_id,
                "E_first_token_id": e_id,
                "baseline_C_logit": c_logit,
                "baseline_E_logit": e_logit,
                "baseline_R_final": R,
                "baseline_pred_clean": int(R > 0),
            })

            for l in layers:
                hs_idx = l + 1
                if hs_idx < len(hidden_states):
                    h = hidden_states[hs_idx][bi, pos, :].detach().float().cpu().numpy().astype(np.float32)
                    all_hidden[l].append(h)

        print(f"[COLLECT] {min(start+batch_size, len(prompts))}/{len(prompts)}")

    hidden_np = {l: np.vstack(v) for l, v in all_hidden.items() if len(v) == len(df)}
    scores = pd.DataFrame(score_rows)
    return hidden_np, scores


def build_steering_vectors(df: pd.DataFrame, hidden_np: Dict[int, np.ndarray], out_dir: Path):
    """
    Build per-layer steering vectors:
      to_stable = mean(stable safe) - mean(wrong attractor)
      to_update = mean(update safe) - mean(wrong attractor)
    """
    cond = df["condition"].values
    wrong_idx = np.where(np.isin(cond, CONFIG["WRONG_ATTRACTOR_CONDITIONS"]))[0]
    stable_idx = np.where(np.isin(cond, CONFIG["SAFE_STABLE_CONDITIONS"]))[0]
    update_idx = np.where(np.isin(cond, CONFIG["SAFE_UPDATE_CONDITIONS"]))[0]

    if len(wrong_idx) == 0 or len(stable_idx) == 0 or len(update_idx) == 0:
        raise ValueError("Not enough conditions to build steering vectors.")

    vectors = {mode: {} for mode in CONFIG["STEERING_MODES"]}
    rows = []
    for l, H in hidden_np.items():
        wrong_mean = H[wrong_idx].mean(axis=0)
        stable_mean = H[stable_idx].mean(axis=0)
        update_mean = H[update_idx].mean(axis=0)

        v_stable = stable_mean - wrong_mean
        v_update = update_mean - wrong_mean

        hidden_norm_mean = float(np.linalg.norm(H, axis=1).mean())

        for mode, v in [("to_stable", v_stable), ("to_update", v_update)]:
            norm = float(np.linalg.norm(v))
            unit = v / (norm + 1e-12)
            vectors[mode][l] = {
                "raw": v.astype(np.float32),
                "unit": unit.astype(np.float32),
                "norm": norm,
                "hidden_norm_mean": hidden_norm_mean,
            }
            rows.append({
                "layer": l,
                "mode": mode,
                "vector_norm": norm,
                "hidden_norm_mean": hidden_norm_mean,
                "scale_norm_ratio": norm / (hidden_norm_mean + 1e-12),
            })

    pd.DataFrame(rows).to_csv(out_dir / "da_asa1_steering_vector_summary.csv", index=False, encoding="utf-8-sig")
    return vectors


# =========================
# Intervention
# =========================

def make_layer_hook(layer_idx: int, direction_tensor: torch.Tensor, alpha: float, scale: float, last_pos_tensor: torch.Tensor):
    """
    Modify last-token hidden state at a decoder layer output.
    Handles output tensor or tuple output.
    """
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        # avoid in-place on views
        h2 = hidden.clone()
        bsz = h2.shape[0]
        add_vec = (alpha * scale * direction_tensor).to(h2.device, dtype=h2.dtype)

        for bi in range(bsz):
            pos = int(last_pos_tensor[bi].item())
            h2[bi, pos, :] = h2[bi, pos, :] + add_vec

        if rest is not None:
            return (h2,) + rest
        return h2
    return hook


@torch.no_grad()
def run_intervention_eval(model, tokenizer, df: pd.DataFrame, vectors: Dict, out_dir: Path):
    layers_module = get_decoder_layers(model)
    prompts = df["prompt"].astype(str).tolist()
    rows = []

    for mode in CONFIG["STEERING_MODES"]:
        for alpha in CONFIG["ALPHAS"]:
            print(f"[EVAL] mode={mode}, alpha={alpha}")

            for start in range(0, len(prompts), CONFIG["BATCH_SIZE"]):
                batch = df.iloc[start:start+CONFIG["BATCH_SIZE"]].reset_index(drop=True)
                enc = tokenizer(
                    batch["prompt"].astype(str).tolist(),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=CONFIG["MAX_LENGTH"],
                )
                input_ids = enc["input_ids"].to(CONFIG["DEVICE"])
                attn = enc["attention_mask"].to(CONFIG["DEVICE"])
                last_pos = attn.sum(dim=1) - 1

                handles = []
                if alpha != 0.0:
                    for l in CONFIG["STEER_LAYERS"]:
                        if l >= len(layers_module) or l not in vectors[mode]:
                            continue
                        direction = torch.tensor(vectors[mode][l]["unit"], device=CONFIG["DEVICE"])
                        scale = vectors[mode][l]["hidden_norm_mean"] if CONFIG["USE_MEAN_HIDDEN_NORM_SCALE"] else 1.0
                        h = layers_module[l].register_forward_hook(
                            make_layer_hook(l, direction, alpha, scale, last_pos)
                        )
                        handles.append(h)

                out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
                logits = out.logits

                for h in handles:
                    h.remove()

                for bi in range(input_ids.size(0)):
                    pos = int(last_pos[bi].item())
                    C = str(batch.loc[bi, "C_clean"])
                    E = str(batch.loc[bi, "E_conflict"])
                    c_id = first_token_id(tokenizer, C)
                    e_id = first_token_id(tokenizer, E)

                    final_logits = logits[bi, pos, :].float()
                    c_logit = float(final_logits[c_id].item())
                    e_logit = float(final_logits[e_id].item())
                    R = c_logit - e_logit

                    rows.append({
                        "row_index": start + bi,
                        "id": batch.loc[bi, "id"],
                        "graph_id": batch.loc[bi, "graph_id"],
                        "condition": batch.loc[bi, "condition"],
                        "mechanism": batch.loc[bi, "mechanism"],
                        "risk_regime": batch.loc[bi, "risk_regime"],
                        "mode": mode,
                        "alpha": alpha,
                        "C_clean": C,
                        "E_conflict": E,
                        "C_logit": c_logit,
                        "E_logit": e_logit,
                        "R_final": R,
                        "pred_clean": int(R > 0),
                    })

                print(f"[EVAL] mode={mode}, alpha={alpha}, {min(start+CONFIG['BATCH_SIZE'], len(prompts))}/{len(prompts)}")

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "da_asa1_steering_results.csv", index=False, encoding="utf-8-sig")
    return res


def summarize_specificity(results: pd.DataFrame, baseline_scores: pd.DataFrame, out_dir: Path):
    base = baseline_scores[["id", "baseline_R_final", "baseline_pred_clean"]].copy()
    df = results.merge(base, on="id", how="left")
    df["delta_R"] = df["R_final"] - df["baseline_R_final"]

    target_mask = df["condition"].isin(CONFIG["TARGET_CONDITIONS"])
    nontarget_mask = df["condition"].isin(CONFIG["NONTARGET_CONDITIONS"])

    rows = []
    for (mode, alpha), sub in df.groupby(["mode", "alpha"]):
        target = sub[target_mask.loc[sub.index] if hasattr(target_mask, "loc") else sub["condition"].isin(CONFIG["TARGET_CONDITIONS"])]
        nontarget = sub[nontarget_mask.loc[sub.index] if hasattr(nontarget_mask, "loc") else sub["condition"].isin(CONFIG["NONTARGET_CONDITIONS"])]

        row = {
            "mode": mode,
            "alpha": alpha,
            "n_total": int(len(sub)),
            "mean_delta_R_all": float(sub["delta_R"].mean()),
            "clean_rate_all": float(sub["pred_clean"].mean()),
        }

        if len(target):
            row.update({
                "n_target": int(len(target)),
                "mean_delta_R_target": float(target["delta_R"].mean()),
                "clean_rate_target": float(target["pred_clean"].mean()),
                "baseline_clean_rate_target": float(target["baseline_pred_clean"].mean()),
            })
        else:
            row.update({"n_target": 0, "mean_delta_R_target": np.nan, "clean_rate_target": np.nan, "baseline_clean_rate_target": np.nan})

        if len(nontarget):
            row.update({
                "n_nontarget": int(len(nontarget)),
                "mean_delta_R_nontarget": float(nontarget["delta_R"].mean()),
                "clean_rate_nontarget": float(nontarget["pred_clean"].mean()),
                "baseline_clean_rate_nontarget": float(nontarget["baseline_pred_clean"].mean()),
            })
        else:
            row.update({"n_nontarget": 0, "mean_delta_R_nontarget": np.nan, "clean_rate_nontarget": np.nan, "baseline_clean_rate_nontarget": np.nan})

        row["specificity_delta_R"] = row["mean_delta_R_target"] - row["mean_delta_R_nontarget"]
        row["target_clean_rate_gain"] = row["clean_rate_target"] - row["baseline_clean_rate_target"]
        row["nontarget_clean_rate_gain"] = row["clean_rate_nontarget"] - row["baseline_clean_rate_nontarget"]
        row["specificity_clean_gain"] = row["target_clean_rate_gain"] - row["nontarget_clean_rate_gain"]
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["mode", "alpha"])
    summary.to_csv(out_dir / "da_asa1_specificity_summary.csv", index=False, encoding="utf-8-sig")

    # condition summary
    cond_rows = []
    for (mode, alpha, cond), sub in df.groupby(["mode", "alpha", "condition"]):
        cond_rows.append({
            "mode": mode,
            "alpha": alpha,
            "condition": cond,
            "n": int(len(sub)),
            "mean_delta_R": float(sub["delta_R"].mean()),
            "mean_R_final": float(sub["R_final"].mean()),
            "clean_rate": float(sub["pred_clean"].mean()),
            "baseline_clean_rate": float(sub["baseline_pred_clean"].mean()),
            "clean_rate_gain": float(sub["pred_clean"].mean() - sub["baseline_pred_clean"].mean()),
        })
    cond_summary = pd.DataFrame(cond_rows)
    cond_summary.to_csv(out_dir / "da_asa1_condition_summary.csv", index=False, encoding="utf-8-sig")

    # verdict
    verdict_rows = []
    best = summary[summary["alpha"] > 0].copy()
    if len(best):
        best_spec = best.sort_values("specificity_delta_R", ascending=False).iloc[0]
        best_clean = best.sort_values("specificity_clean_gain", ascending=False).iloc[0]
        verdict_rows.append({
            "claim": "Attractor steering improves target R margin more than non-target.",
            "best_mode": best_spec["mode"],
            "best_alpha": float(best_spec["alpha"]),
            "specificity_delta_R": float(best_spec["specificity_delta_R"]),
            "mean_delta_R_target": float(best_spec["mean_delta_R_target"]),
            "mean_delta_R_nontarget": float(best_spec["mean_delta_R_nontarget"]),
            "verdict": "PASS_MARGIN_SPECIFICITY" if best_spec["specificity_delta_R"] > 0 else "FAIL_MARGIN_SPECIFICITY",
        })
        verdict_rows.append({
            "claim": "Attractor steering improves target clean-rate more than non-target.",
            "best_mode": best_clean["mode"],
            "best_alpha": float(best_clean["alpha"]),
            "specificity_clean_gain": float(best_clean["specificity_clean_gain"]),
            "target_clean_rate_gain": float(best_clean["target_clean_rate_gain"]),
            "nontarget_clean_rate_gain": float(best_clean["nontarget_clean_rate_gain"]),
            "verdict": "PASS_CLEAN_RATE_SPECIFICITY" if best_clean["specificity_clean_gain"] > 0 else "FAIL_CLEAN_RATE_SPECIFICITY",
        })

    verdict = pd.DataFrame(verdict_rows)
    verdict.to_csv(out_dir / "da_asa1_verdict_summary.csv", index=False, encoding="utf-8-sig")
    return summary, cond_summary, verdict


def main():
    np.random.seed(CONFIG["RANDOM_SEED"])
    torch.manual_seed(CONFIG["RANDOM_SEED"])

    out_dir = ensure_dir(CONFIG["OUTPUT_DIR"])
    with open(out_dir / "da_asa1_config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)

    df = build_prompt_dataset()
    df.to_csv(out_dir / "da_asa1_generated_prompts.csv", index=False, encoding="utf-8-sig")

    model, tokenizer = load_model_and_tokenizer(CONFIG["MODEL_PATH"], CONFIG["DEVICE"], CONFIG["DTYPE"])

    tok_df = tokenize_candidates(df, tokenizer)
    tok_df.to_csv(out_dir / "da_asa1_candidate_tokenization.csv", index=False, encoding="utf-8-sig")

    hidden_np, baseline_scores = forward_collect_hidden_and_scores(
        model=model,
        tokenizer=tokenizer,
        df=df,
        layers=CONFIG["COLLECT_LAYERS"],
        batch_size=CONFIG["BATCH_SIZE"],
        max_length=CONFIG["MAX_LENGTH"],
        device=CONFIG["DEVICE"],
    )
    baseline_scores.to_csv(out_dir / "da_asa1_baseline_scores.csv", index=False, encoding="utf-8-sig")

    vectors = build_steering_vectors(df, hidden_np, out_dir)

    results = run_intervention_eval(model, tokenizer, df, vectors, out_dir)
    summary, cond_summary, verdict = summarize_specificity(results, baseline_scores, out_dir)

    print("\n[DONE] Outputs saved to:", out_dir.resolve())
    print("\n[SUMMARY]")
    print(summary.to_string(index=False))
    print("\n[VERDICT]")
    print(verdict.to_string(index=False))

    print("\n[KEY FILES]")
    for name in [
        "da_asa1_generated_prompts.csv",
        "da_asa1_candidate_tokenization.csv",
        "da_asa1_baseline_scores.csv",
        "da_asa1_steering_vector_summary.csv",
        "da_asa1_steering_results.csv",
        "da_asa1_specificity_summary.csv",
        "da_asa1_condition_summary.csv",
        "da_asa1_verdict_summary.csv",
    ]:
        print(" -", out_dir / name)


if __name__ == "__main__":
    main()
