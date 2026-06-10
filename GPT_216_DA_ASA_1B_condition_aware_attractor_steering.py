# -*- coding: utf-8 -*-
r"""
GPT_211_DA_ASA_1B_condition_aware_attractor_steering.py

DA-ASA-1B: Condition-Aware Attractor Steering

背景：
DA-ASA-1 证明：
1. to_update 对 closure_exception / closure_negation / closure_override 有显著拉回作用；
2. closure_exception / closure_negation 最佳 alpha 约 0.15-0.20；
3. closure_override 需要更强 alpha 约 0.20-0.30；
4. hallucination_like 被 to_update 明显恶化；
5. to_stable 副作用大，不适合作为通用 safe attractor。

本实验目标：
构造 condition-aware policy：

    closure_exception  -> to_update alpha=0.15
    closure_negation   -> to_update alpha=0.20
    closure_override   -> to_update alpha=0.25 / 0.30
    hallucination_like -> no to_update; test to_stable / suppress_wrong_chain
    stable / weak / competition -> no intervention

并与以下基线比较：
    no_intervention
    fixed_to_update_alpha0.15
    fixed_to_update_alpha0.20
    fixed_to_stable_alpha0.30
    condition_aware_policy_v1
    condition_aware_policy_v2

新增 suppress_wrong_chain：
    对 hallucination_like 构造：
        v_suppress_wrong = mean_H(stable_clean) - mean_H(hallucination_like)
    只对 hallucination_like 施加，避免把它推向 update attractor。

输出：
    da_asa1b_outputs/
        da_asa1b_generated_prompts.csv
        da_asa1b_baseline_scores.csv
        da_asa1b_policy_results.csv
        da_asa1b_policy_summary.csv
        da_asa1b_condition_summary.csv
        da_asa1b_verdict_summary.csv
"""

import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")


CONFIG = {
    "MODEL_PATH": r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main",
    "MODEL_PATH_FALLBACK": r"D:\model\models--Qwen--Qwen2.5B-Instruct\main",

    "OUTPUT_DIR": r"da_asa1b_outputs",

    "STEER_LAYERS": [15, 16, 17, 18, 19],
    "COLLECT_LAYERS": [15, 16, 17, 18, 19],

    "BATCH_SIZE": 4,
    "MAX_LENGTH": 224,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "DTYPE": "auto",
    "RANDOM_SEED": 42,

    "USE_MEAN_HIDDEN_NORM_SCALE": True,

    "TARGET_CONDITIONS": ["closure_override", "closure_exception", "closure_negation", "hallucination_like"],
    "NONTARGET_CONDITIONS": ["stable_clean", "stable_redundant", "weak_distractor", "competition_direct"],

    "WRONG_ATTRACTOR_CONDITIONS": ["closure_override", "hallucination_like"],
    "SAFE_STABLE_CONDITIONS": ["stable_clean", "stable_redundant"],
    "SAFE_UPDATE_CONDITIONS": ["closure_update", "closure_temporal", "closure_authority"],

    # policies. Each policy maps condition -> (mode, alpha)
    # mode can be: none, to_update, to_stable, suppress_wrong
    "POLICIES": {
        "no_intervention": {
            "*": ("none", 0.0),
        },
        "fixed_to_update_a015": {
            "*": ("to_update", 0.15),
        },
        "fixed_to_update_a020": {
            "*": ("to_update", 0.20),
        },
        "fixed_to_stable_a030": {
            "*": ("to_stable", 0.30),
        },
        "condition_aware_v1": {
            "closure_exception": ("to_update", 0.15),
            "closure_negation": ("to_update", 0.20),
            "closure_override": ("to_update", 0.25),
            "hallucination_like": ("suppress_wrong", 0.15),
            "*": ("none", 0.0),
        },
        "condition_aware_v2": {
            "closure_exception": ("to_update", 0.15),
            "closure_negation": ("to_update", 0.15),
            "closure_override": ("to_update", 0.30),
            "hallucination_like": ("to_stable", 0.15),
            "*": ("none", 0.0),
        },
        "condition_aware_v3_conservative": {
            "closure_exception": ("to_update", 0.10),
            "closure_negation": ("to_update", 0.15),
            "closure_override": ("to_update", 0.20),
            "hallucination_like": ("none", 0.0),
            "*": ("none", 0.0),
        },
    },
}


def build_prompt_dataset() -> pd.DataFrame:
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

    rows, sid = [], 0
    for A, B, C, E in pairs:
        for t in templates:
            rows.append({
                "id": f"daasa1b_{sid:05d}",
                "graph_id": f"{A}_{B}_{C}_{E}",
                "A": A,
                "B": B,
                "C_clean": C,
                "E_conflict": E,
                "concept": A,
                "condition": t["condition"],
                "mechanism": t["mechanism"],
                "risk_regime": t["risk_regime"],
                "prompt": t["template"].format(A=A, B=B, C=C, E=E),
            })
            sid += 1

    return pd.DataFrame(rows)


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
    kwargs = {"local_files_only": True, "output_hidden_states": True}
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
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Cannot locate decoder layers.")


def first_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(" " + text, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(text, add_special_tokens=False)
    return int(ids[0])


def tokenize_candidates(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    rows = []
    for _, r in df.drop_duplicates(["C_clean", "E_conflict"]).iterrows():
        C, E = str(r["C_clean"]), str(r["E_conflict"])
        c_ids = tokenizer.encode(" " + C, add_special_tokens=False)
        e_ids = tokenizer.encode(" " + E, add_special_tokens=False)
        rows.append({
            "C_clean": C,
            "E_conflict": E,
            "C_first_token_id": first_token_id(tokenizer, C),
            "E_first_token_id": first_token_id(tokenizer, E),
            "C_ids_with_space": " ".join(map(str, c_ids)),
            "E_ids_with_space": " ".join(map(str, e_ids)),
            "C_n_tokens": len(c_ids),
            "E_n_tokens": len(e_ids),
        })
    return pd.DataFrame(rows)


@torch.no_grad()
def forward_collect_hidden_and_scores(model, tokenizer, df, layers, batch_size, max_length, device):
    all_hidden = {l: [] for l in layers}
    rows = []
    prompts = df["prompt"].astype(str).tolist()

    for start in range(0, len(prompts), batch_size):
        batch = df.iloc[start:start+batch_size].reset_index(drop=True)
        enc = tokenizer(batch["prompt"].tolist(), return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        last_pos = attn.sum(dim=1) - 1

        out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
        logits = out.logits
        hidden_states = out.hidden_states

        for bi in range(input_ids.shape[0]):
            pos = int(last_pos[bi].item())
            C, E = str(batch.loc[bi, "C_clean"]), str(batch.loc[bi, "E_conflict"])
            c_id, e_id = first_token_id(tokenizer, C), first_token_id(tokenizer, E)
            fl = logits[bi, pos, :].float()
            R = float(fl[c_id].item() - fl[e_id].item())

            rows.append({
                "row_index": start + bi,
                "id": batch.loc[bi, "id"],
                "graph_id": batch.loc[bi, "graph_id"],
                "condition": batch.loc[bi, "condition"],
                "mechanism": batch.loc[bi, "mechanism"],
                "risk_regime": batch.loc[bi, "risk_regime"],
                "C_clean": C,
                "E_conflict": E,
                "baseline_C_logit": float(fl[c_id].item()),
                "baseline_E_logit": float(fl[e_id].item()),
                "baseline_R_final": R,
                "baseline_pred_clean": int(R > 0),
            })

            for l in layers:
                hs_idx = l + 1
                if hs_idx < len(hidden_states):
                    all_hidden[l].append(hidden_states[hs_idx][bi, pos, :].detach().float().cpu().numpy().astype(np.float32))

        print(f"[COLLECT] {min(start+batch_size, len(prompts))}/{len(prompts)}")

    hidden_np = {l: np.vstack(v) for l, v in all_hidden.items() if len(v) == len(df)}
    return hidden_np, pd.DataFrame(rows)


def build_steering_vectors(df, hidden_np, out_dir):
    cond = df["condition"].values
    wrong_idx = np.where(np.isin(cond, CONFIG["WRONG_ATTRACTOR_CONDITIONS"]))[0]
    stable_idx = np.where(np.isin(cond, CONFIG["SAFE_STABLE_CONDITIONS"]))[0]
    update_idx = np.where(np.isin(cond, CONFIG["SAFE_UPDATE_CONDITIONS"]))[0]
    halluc_idx = np.where(cond == "hallucination_like")[0]

    vectors = {"to_stable": {}, "to_update": {}, "suppress_wrong": {}}
    rows = []

    for l, H in hidden_np.items():
        wrong_mean = H[wrong_idx].mean(axis=0)
        stable_mean = H[stable_idx].mean(axis=0)
        update_mean = H[update_idx].mean(axis=0)
        halluc_mean = H[halluc_idx].mean(axis=0) if len(halluc_idx) else wrong_mean
        hidden_norm_mean = float(np.linalg.norm(H, axis=1).mean())

        mode_vecs = {
            "to_stable": stable_mean - wrong_mean,
            "to_update": update_mean - wrong_mean,
            "suppress_wrong": stable_mean - halluc_mean,
        }

        for mode, v in mode_vecs.items():
            norm = float(np.linalg.norm(v))
            unit = v / (norm + 1e-12)
            vectors[mode][l] = {
                "unit": unit.astype(np.float32),
                "raw": v.astype(np.float32),
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

    pd.DataFrame(rows).to_csv(out_dir / "da_asa1b_steering_vector_summary.csv", index=False, encoding="utf-8-sig")
    return vectors


def make_layer_hook(direction_tensor, alpha, scale, last_pos_tensor):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        h2 = hidden.clone()
        add_vec = (alpha * scale * direction_tensor).to(h2.device, dtype=h2.dtype)
        for bi in range(h2.shape[0]):
            pos = int(last_pos_tensor[bi].item())
            h2[bi, pos, :] = h2[bi, pos, :] + add_vec

        if rest is not None:
            return (h2,) + rest
        return h2
    return hook


def policy_action(policy_name: str, condition: str):
    pol = CONFIG["POLICIES"][policy_name]
    if condition in pol:
        return pol[condition]
    return pol.get("*", ("none", 0.0))


@torch.no_grad()
def run_policy_eval(model, tokenizer, df, vectors, out_dir):
    layers_module = get_decoder_layers(model)
    prompts = df["prompt"].astype(str).tolist()
    rows = []

    for policy_name in CONFIG["POLICIES"].keys():
        print(f"[POLICY] {policy_name}")

        for start in range(0, len(prompts), CONFIG["BATCH_SIZE"]):
            batch = df.iloc[start:start+CONFIG["BATCH_SIZE"]].reset_index(drop=True)
            enc = tokenizer(batch["prompt"].tolist(), return_tensors="pt", padding=True, truncation=True, max_length=CONFIG["MAX_LENGTH"])
            input_ids = enc["input_ids"].to(CONFIG["DEVICE"])
            attn = enc["attention_mask"].to(CONFIG["DEVICE"])
            last_pos = attn.sum(dim=1) - 1

            # For mixed per-sample policies within a batch, we need a custom hook that applies different vector/alpha per sample.
            handles = []

            def make_mixed_hook(layer_idx):
                def hook(module, inputs, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                        rest = output[1:]
                    else:
                        hidden = output
                        rest = None

                    h2 = hidden.clone()
                    for bi in range(h2.shape[0]):
                        cond = batch.loc[bi, "condition"]
                        mode, alpha = policy_action(policy_name, cond)
                        if mode == "none" or alpha == 0.0:
                            continue
                        if layer_idx not in vectors[mode]:
                            continue
                        direction = torch.tensor(vectors[mode][layer_idx]["unit"], device=h2.device, dtype=h2.dtype)
                        scale = vectors[mode][layer_idx]["hidden_norm_mean"] if CONFIG["USE_MEAN_HIDDEN_NORM_SCALE"] else 1.0
                        pos = int(last_pos[bi].item())
                        h2[bi, pos, :] = h2[bi, pos, :] + alpha * scale * direction

                    if rest is not None:
                        return (h2,) + rest
                    return h2
                return hook

            for l in CONFIG["STEER_LAYERS"]:
                if l < len(layers_module):
                    handles.append(layers_module[l].register_forward_hook(make_mixed_hook(l)))

            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            logits = out.logits

            for h in handles:
                h.remove()

            for bi in range(input_ids.shape[0]):
                pos = int(last_pos[bi].item())
                C, E = str(batch.loc[bi, "C_clean"]), str(batch.loc[bi, "E_conflict"])
                c_id, e_id = first_token_id(tokenizer, C), first_token_id(tokenizer, E)
                fl = logits[bi, pos, :].float()
                R = float(fl[c_id].item() - fl[e_id].item())
                mode, alpha = policy_action(policy_name, batch.loc[bi, "condition"])

                rows.append({
                    "row_index": start + bi,
                    "id": batch.loc[bi, "id"],
                    "graph_id": batch.loc[bi, "graph_id"],
                    "condition": batch.loc[bi, "condition"],
                    "mechanism": batch.loc[bi, "mechanism"],
                    "risk_regime": batch.loc[bi, "risk_regime"],
                    "policy": policy_name,
                    "applied_mode": mode,
                    "applied_alpha": alpha,
                    "C_clean": C,
                    "E_conflict": E,
                    "C_logit": float(fl[c_id].item()),
                    "E_logit": float(fl[e_id].item()),
                    "R_final": R,
                    "pred_clean": int(R > 0),
                })

            print(f"[POLICY] {policy_name}: {min(start+CONFIG['BATCH_SIZE'], len(prompts))}/{len(prompts)}")

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "da_asa1b_policy_results.csv", index=False, encoding="utf-8-sig")
    return res


def summarize(results, baseline_scores, out_dir):
    base = baseline_scores[["id", "baseline_R_final", "baseline_pred_clean"]].copy()
    df = results.merge(base, on="id", how="left")
    df["delta_R"] = df["R_final"] - df["baseline_R_final"]

    target_cond = set(CONFIG["TARGET_CONDITIONS"])
    nontarget_cond = set(CONFIG["NONTARGET_CONDITIONS"])

    rows = []
    for policy, sub in df.groupby("policy"):
        target = sub[sub["condition"].isin(target_cond)]
        nontarget = sub[sub["condition"].isin(nontarget_cond)]

        row = {
            "policy": policy,
            "n_total": int(len(sub)),
            "mean_delta_R_all": float(sub["delta_R"].mean()),
            "clean_rate_all": float(sub["pred_clean"].mean()),
        }
        for name, part in [("target", target), ("nontarget", nontarget)]:
            if len(part):
                row[f"n_{name}"] = int(len(part))
                row[f"mean_delta_R_{name}"] = float(part["delta_R"].mean())
                row[f"clean_rate_{name}"] = float(part["pred_clean"].mean())
                row[f"baseline_clean_rate_{name}"] = float(part["baseline_pred_clean"].mean())
                row[f"clean_rate_gain_{name}"] = row[f"clean_rate_{name}"] - row[f"baseline_clean_rate_{name}"]
            else:
                row[f"n_{name}"] = 0
                row[f"mean_delta_R_{name}"] = np.nan
                row[f"clean_rate_{name}"] = np.nan
                row[f"baseline_clean_rate_{name}"] = np.nan
                row[f"clean_rate_gain_{name}"] = np.nan

        row["specificity_delta_R"] = row["mean_delta_R_target"] - row["mean_delta_R_nontarget"]
        row["specificity_clean_gain"] = row["clean_rate_gain_target"] - row["clean_rate_gain_nontarget"]
        rows.append(row)

    policy_summary = pd.DataFrame(rows).sort_values("specificity_delta_R", ascending=False)
    policy_summary.to_csv(out_dir / "da_asa1b_policy_summary.csv", index=False, encoding="utf-8-sig")

    cond_rows = []
    for (policy, cond), sub in df.groupby(["policy", "condition"]):
        cond_rows.append({
            "policy": policy,
            "condition": cond,
            "n": int(len(sub)),
            "applied_mode": sub["applied_mode"].iloc[0],
            "applied_alpha": float(sub["applied_alpha"].iloc[0]),
            "mean_delta_R": float(sub["delta_R"].mean()),
            "mean_R_final": float(sub["R_final"].mean()),
            "clean_rate": float(sub["pred_clean"].mean()),
            "baseline_clean_rate": float(sub["baseline_pred_clean"].mean()),
            "clean_rate_gain": float(sub["pred_clean"].mean() - sub["baseline_pred_clean"].mean()),
        })
    cond_summary = pd.DataFrame(cond_rows)
    cond_summary.to_csv(out_dir / "da_asa1b_condition_summary.csv", index=False, encoding="utf-8-sig")

    verdict_rows = []
    if len(policy_summary):
        best_margin = policy_summary.iloc[0]
        best_clean = policy_summary.sort_values("specificity_clean_gain", ascending=False).iloc[0]

        verdict_rows.append({
            "claim": "Condition-aware policy improves target R margin more than non-target.",
            "best_policy": best_margin["policy"],
            "specificity_delta_R": float(best_margin["specificity_delta_R"]),
            "mean_delta_R_target": float(best_margin["mean_delta_R_target"]),
            "mean_delta_R_nontarget": float(best_margin["mean_delta_R_nontarget"]),
            "verdict": "PASS_POLICY_MARGIN_SPECIFICITY" if best_margin["specificity_delta_R"] > 0 else "FAIL_POLICY_MARGIN_SPECIFICITY",
        })
        verdict_rows.append({
            "claim": "Condition-aware policy improves target clean-rate more than non-target.",
            "best_policy": best_clean["policy"],
            "specificity_clean_gain": float(best_clean["specificity_clean_gain"]),
            "clean_rate_gain_target": float(best_clean["clean_rate_gain_target"]),
            "clean_rate_gain_nontarget": float(best_clean["clean_rate_gain_nontarget"]),
            "verdict": "PASS_POLICY_CLEAN_RATE_SPECIFICITY" if best_clean["specificity_clean_gain"] > 0 else "FAIL_POLICY_CLEAN_RATE_SPECIFICITY",
        })

        # Compare best condition-aware vs fixed baselines
        fixed = policy_summary[policy_summary["policy"].str.startswith("fixed_")]
        aware = policy_summary[policy_summary["policy"].str.startswith("condition_aware")]
        if len(fixed) and len(aware):
            best_fixed = fixed.sort_values("specificity_delta_R", ascending=False).iloc[0]
            best_aware = aware.sort_values("specificity_delta_R", ascending=False).iloc[0]
            verdict_rows.append({
                "claim": "Best condition-aware policy beats best fixed policy on margin specificity.",
                "best_aware_policy": best_aware["policy"],
                "best_fixed_policy": best_fixed["policy"],
                "aware_specificity_delta_R": float(best_aware["specificity_delta_R"]),
                "fixed_specificity_delta_R": float(best_fixed["specificity_delta_R"]),
                "gain": float(best_aware["specificity_delta_R"] - best_fixed["specificity_delta_R"]),
                "verdict": "PASS_AWARE_BEATS_FIXED" if best_aware["specificity_delta_R"] > best_fixed["specificity_delta_R"] else "FAIL_AWARE_NOT_BEAT_FIXED",
            })

    verdict = pd.DataFrame(verdict_rows)
    verdict.to_csv(out_dir / "da_asa1b_verdict_summary.csv", index=False, encoding="utf-8-sig")
    return policy_summary, cond_summary, verdict


def main():
    np.random.seed(CONFIG["RANDOM_SEED"])
    torch.manual_seed(CONFIG["RANDOM_SEED"])

    out_dir = ensure_dir(CONFIG["OUTPUT_DIR"])
    with open(out_dir / "da_asa1b_config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)

    df = build_prompt_dataset()
    df.to_csv(out_dir / "da_asa1b_generated_prompts.csv", index=False, encoding="utf-8-sig")

    model, tokenizer = load_model_and_tokenizer(CONFIG["MODEL_PATH"], CONFIG["DEVICE"], CONFIG["DTYPE"])

    tok_df = tokenize_candidates(df, tokenizer)
    tok_df.to_csv(out_dir / "da_asa1b_candidate_tokenization.csv", index=False, encoding="utf-8-sig")

    hidden_np, baseline = forward_collect_hidden_and_scores(
        model=model,
        tokenizer=tokenizer,
        df=df,
        layers=CONFIG["COLLECT_LAYERS"],
        batch_size=CONFIG["BATCH_SIZE"],
        max_length=CONFIG["MAX_LENGTH"],
        device=CONFIG["DEVICE"],
    )
    baseline.to_csv(out_dir / "da_asa1b_baseline_scores.csv", index=False, encoding="utf-8-sig")

    vectors = build_steering_vectors(df, hidden_np, out_dir)
    results = run_policy_eval(model, tokenizer, df, vectors, out_dir)
    policy_summary, cond_summary, verdict = summarize(results, baseline, out_dir)

    print("\n[DONE] Outputs saved to:", out_dir.resolve())
    print("\n[POLICY SUMMARY]")
    print(policy_summary.to_string(index=False))
    print("\n[VERDICT]")
    print(verdict.to_string(index=False))

    print("\n[KEY FILES]")
    for name in [
        "da_asa1b_generated_prompts.csv",
        "da_asa1b_candidate_tokenization.csv",
        "da_asa1b_baseline_scores.csv",
        "da_asa1b_steering_vector_summary.csv",
        "da_asa1b_policy_results.csv",
        "da_asa1b_policy_summary.csv",
        "da_asa1b_condition_summary.csv",
        "da_asa1b_verdict_summary.csv",
    ]:
        print(" -", out_dir / name)


if __name__ == "__main__":
    main()
