# -*- coding: utf-8 -*-
r"""
GPT_214_DA_ASA_2A_generalized_three_stage_attractor_policy.py

DA-ASA-2A: Generalized Three-Stage Attractor Policy

背景：
DA-ASA-1D 已证明 closure_override 可通过三阶段控制完全推过边界：
    Stage 1: L15-L17 to_update
    Stage 2: L18-L19 suppress_override
    Stage 3: L23-L25 order/boundary commit correction

DA-ASA-2A 目标：
把三阶段 attractor control 从单一 closure_override 推广到：
    closure_exception
    closure_negation
    closure_override
    hallucination_like
    competition_source_claim
    competition_equal_evidence

核心问题：
1. 三阶段策略是否能泛化到多种 wrong-closure / binding / competition 样本？
2. 哪些条件需要三阶段，哪些只需要 to_update？
3. hallucination_like 是否仍需要特殊控制器？
4. 手工 condition-aware policy 是否优于 fixed policy？
5. 是否可以形成一个初步 generalized policy template？

策略类型：
- no_intervention
- fixed_to_update
- fixed_three_stage
- condition_aware_v1
- condition_aware_v2
- generalized_three_stage_v1
- generalized_three_stage_v2

输出：
    da_asa2a_outputs/
        da_asa2a_policy_summary.csv
        da_asa2a_condition_summary.csv
        da_asa2a_verdict_summary.csv
"""

import json
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

    "OUTPUT_DIR": r"da_asa2a_outputs",

    "COLLECT_LAYERS": list(range(15, 26)),
    "STAGE1_LAYERS": [15, 16, 17],
    "STAGE2_LAYERS": [18, 19],
    "COMMIT_20_22": [20, 21, 22],
    "COMMIT_23_25": [23, 24, 25],
    "ALL_STEER_LAYERS": list(range(15, 26)),

    "BATCH_SIZE": 4,
    "MAX_LENGTH": 224,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "DTYPE": "auto",
    "RANDOM_SEED": 42,
    "USE_MEAN_HIDDEN_NORM_SCALE": True,

    # Generalized target family
    "TARGET_CONDITIONS": [
        "closure_override",
        "closure_exception",
        "closure_negation",
        "hallucination_like",
        "competition_source_claim",
        "competition_equal_evidence",
    ],
    "CORE_CLOSURE_TARGETS": ["closure_override", "closure_exception", "closure_negation"],
    "COMPETITION_BINDING_TARGETS": ["competition_source_claim", "competition_equal_evidence"],
    "HALLUCINATION_CONDITIONS": ["hallucination_like"],
    "NONTARGET_CONDITIONS": ["stable_clean", "stable_redundant", "weak_distractor", "competition_direct"],

    "WRONG_ATTRACTOR_CONDITIONS": ["closure_override", "hallucination_like"],
    "OVERRIDE_ATTRACTOR_CONDITIONS": ["closure_override"],
    "HALLUCINATION_ATTRACTOR_CONDITIONS": ["hallucination_like"],
    "SAFE_STABLE_CONDITIONS": ["stable_clean", "stable_redundant"],
    "SAFE_UPDATE_CONDITIONS": ["closure_update", "closure_temporal", "closure_authority"],

    "STAGE1_ALPHA": 0.20,
    "STAGE2_ALPHA": 0.15,
    "COMMIT_ALPHA": 0.20,
}


# =========================
# Dataset
# =========================

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
                "id": f"daasa2a_{sid:05d}",
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


# =========================
# Model utils
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
    kwargs = {"local_files_only": True, "output_hidden_states": True}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    elif device == "cuda":
        kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    for start in range(0, len(df), batch_size):
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
                "baseline_R_final": R,
                "baseline_pred_clean": int(R > 0),
            })

            for l in layers:
                hs_idx = l + 1
                if hs_idx < len(hidden_states):
                    all_hidden[l].append(hidden_states[hs_idx][bi, pos, :].detach().float().cpu().numpy().astype(np.float32))

        print(f"[COLLECT] {min(start+batch_size, len(df))}/{len(df)}")
    hidden_np = {l: np.vstack(v) for l, v in all_hidden.items() if len(v) == len(df)}
    return hidden_np, pd.DataFrame(rows)


# =========================
# Vectors and policies
# =========================

def build_vectors(df, hidden_np, baseline, out_dir):
    cond = df["condition"].values

    stable_idx = np.where(np.isin(cond, CONFIG["SAFE_STABLE_CONDITIONS"]))[0]
    update_idx = np.where(np.isin(cond, CONFIG["SAFE_UPDATE_CONDITIONS"]))[0]
    override_idx = np.where(cond == "closure_override")[0]
    halluc_idx = np.where(cond == "hallucination_like")[0]
    wrong_idx = np.where(np.isin(cond, CONFIG["WRONG_ATTRACTOR_CONDITIONS"]))[0]

    clean_pred_idx = baseline[baseline["baseline_pred_clean"] == 1]["row_index"].values
    conflict_pred_idx = baseline[baseline["baseline_pred_clean"] == 0]["row_index"].values
    if len(clean_pred_idx) == 0 or len(conflict_pred_idx) == 0:
        clean_pred_idx = stable_idx
        conflict_pred_idx = override_idx

    vectors = {m: {} for m in [
        "to_update",
        "to_stable",
        "suppress_override",
        "suppress_wrong",
        "order_precursor",
        "boundary_push",
        "commit_stable",
        "commit_update",
    ]}
    rows = []
    for l, H in hidden_np.items():
        stable_mean = H[stable_idx].mean(axis=0)
        update_mean = H[update_idx].mean(axis=0)
        override_mean = H[override_idx].mean(axis=0)
        halluc_mean = H[halluc_idx].mean(axis=0)
        wrong_mean = H[wrong_idx].mean(axis=0)
        clean_mean = H[clean_pred_idx].mean(axis=0)
        conflict_mean = H[conflict_pred_idx].mean(axis=0)
        hidden_norm_mean = float(np.linalg.norm(H, axis=1).mean())

        raw = {
            "to_update": update_mean - wrong_mean,
            "to_stable": stable_mean - wrong_mean,
            "suppress_override": stable_mean - override_mean,
            "suppress_wrong": stable_mean - halluc_mean,
            "order_precursor": clean_mean - conflict_mean,
            "boundary_push": clean_mean - conflict_mean,
            "commit_stable": stable_mean - override_mean,
            "commit_update": update_mean - override_mean,
        }

        for mode, v in raw.items():
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
    pd.DataFrame(rows).to_csv(out_dir / "da_asa2a_steering_vector_summary.csv", index=False, encoding="utf-8-sig")
    return vectors


def layers_for_group(group):
    if group == "stage1":
        return set(CONFIG["STAGE1_LAYERS"])
    if group == "stage2":
        return set(CONFIG["STAGE2_LAYERS"])
    if group == "commit23":
        return set(CONFIG["COMMIT_23_25"])
    if group == "commit20":
        return set(CONFIG["COMMIT_20_22"])
    if group == "all":
        return set(CONFIG["ALL_STEER_LAYERS"])
    return set()


def generate_policies():
    """
    Returns policy dict:
      policy[condition] = [(layer_group, mode, alpha), ...]
      "*" default
    """
    a1 = CONFIG["STAGE1_ALPHA"]
    a2 = CONFIG["STAGE2_ALPHA"]
    ac = CONFIG["COMMIT_ALPHA"]

    three_stage_override = [("stage1", "to_update", a1), ("stage2", "suppress_override", a2), ("commit23", "order_precursor", ac)]
    update_moderate = [("all", "to_update", 0.15)]
    update_strong = [("all", "to_update", 0.20)]
    halluc_safe = [("all", "to_stable", 0.15)]
    suppress_halluc = [("all", "suppress_wrong", 0.15)]
    competition_binding = [("stage1", "to_update", 0.10), ("commit23", "order_precursor", 0.10)]

    policies = {
        "no_intervention": {"*": []},
        "fixed_to_update_a015": {"*": [("all", "to_update", 0.15)]},
        "fixed_to_update_a020": {"*": [("all", "to_update", 0.20)]},
        "fixed_three_stage_all": {"*": three_stage_override},

        # 1B-like
        "condition_aware_v1": {
            "closure_exception": update_moderate,
            "closure_negation": update_strong,
            "closure_override": [("all", "to_update", 0.30)],
            "hallucination_like": halluc_safe,
            "*": [],
        },

        # 1D-specialized override
        "condition_aware_v2_override_3stage": {
            "closure_exception": update_moderate,
            "closure_negation": update_moderate,
            "closure_override": three_stage_override,
            "hallucination_like": halluc_safe,
            "*": [],
        },

        # Generalized: also treats source/equal evidence as binding targets
        "generalized_three_stage_v1": {
            "closure_exception": update_moderate,
            "closure_negation": update_moderate,
            "closure_override": three_stage_override,
            "hallucination_like": halluc_safe,
            "competition_source_claim": competition_binding,
            "competition_equal_evidence": competition_binding,
            "*": [],
        },

        # More aggressive for binding/competition
        "generalized_three_stage_v2": {
            "closure_exception": update_moderate,
            "closure_negation": update_strong,
            "closure_override": three_stage_override,
            "hallucination_like": suppress_halluc,
            "competition_source_claim": [("stage1", "to_update", 0.15), ("commit23", "order_precursor", 0.15)],
            "competition_equal_evidence": [("stage1", "to_update", 0.15), ("commit23", "order_precursor", 0.15)],
            "*": [],
        },

        # Conservative: only intervene proven closure family; leave hallucination/binding alone.
        "generalized_conservative_v3": {
            "closure_exception": update_moderate,
            "closure_negation": update_moderate,
            "closure_override": three_stage_override,
            "*": [],
        },
    }
    return policies


def policy_action(policies, policy_name, condition):
    pol = policies[policy_name]
    if condition in pol:
        return pol[condition]
    return pol.get("*", [])


# =========================
# Intervention
# =========================

@torch.no_grad()
def run_policy_eval(model, tokenizer, df, vectors, policies, out_dir):
    layers_module = get_decoder_layers(model)
    rows = []

    for policy_name in policies.keys():
        print(f"[POLICY] {policy_name}")

        for start in range(0, len(df), CONFIG["BATCH_SIZE"]):
            batch = df.iloc[start:start+CONFIG["BATCH_SIZE"]].reset_index(drop=True)
            enc = tokenizer(batch["prompt"].tolist(), return_tensors="pt", padding=True, truncation=True, max_length=CONFIG["MAX_LENGTH"])
            input_ids = enc["input_ids"].to(CONFIG["DEVICE"])
            attn = enc["attention_mask"].to(CONFIG["DEVICE"])
            last_pos = attn.sum(dim=1) - 1

            def make_hook(layer_idx):
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
                        actions = policy_action(policies, policy_name, cond)
                        for group, mode, alpha in actions:
                            if alpha == 0.0 or mode == "none":
                                continue
                            if layer_idx not in layers_for_group(group):
                                continue
                            if mode not in vectors or layer_idx not in vectors[mode]:
                                continue
                            direction = torch.tensor(vectors[mode][layer_idx]["unit"], device=h2.device, dtype=h2.dtype)
                            scale = vectors[mode][layer_idx]["hidden_norm_mean"] if CONFIG["USE_MEAN_HIDDEN_NORM_SCALE"] else 1.0
                            pos = int(last_pos[bi].item())
                            h2[bi, pos, :] = h2[bi, pos, :] + alpha * scale * direction

                    if rest is not None:
                        return (h2,) + rest
                    return h2
                return hook

            handles = []
            for l in CONFIG["ALL_STEER_LAYERS"]:
                if l < len(layers_module):
                    handles.append(layers_module[l].register_forward_hook(make_hook(l)))

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
                actions = policy_action(policies, policy_name, batch.loc[bi, "condition"])
                rows.append({
                    "row_index": start + bi,
                    "id": batch.loc[bi, "id"],
                    "graph_id": batch.loc[bi, "graph_id"],
                    "condition": batch.loc[bi, "condition"],
                    "mechanism": batch.loc[bi, "mechanism"],
                    "risk_regime": batch.loc[bi, "risk_regime"],
                    "policy": policy_name,
                    "applied_actions": json.dumps(actions),
                    "C_clean": C,
                    "E_conflict": E,
                    "R_final": R,
                    "pred_clean": int(R > 0),
                })

            print(f"[POLICY] {policy_name}: {min(start+CONFIG['BATCH_SIZE'], len(df))}/{len(df)}")

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "da_asa2a_policy_results.csv", index=False, encoding="utf-8-sig")
    return res


# =========================
# Summary
# =========================

def summarize(results, baseline, out_dir):
    df = results.merge(baseline[["id", "baseline_R_final", "baseline_pred_clean"]], on="id", how="left")
    df["delta_R"] = df["R_final"] - df["baseline_R_final"]

    groups = {
        "target_all": set(CONFIG["TARGET_CONDITIONS"]),
        "core_closure": set(CONFIG["CORE_CLOSURE_TARGETS"]),
        "binding_competition": set(CONFIG["COMPETITION_BINDING_TARGETS"]),
        "hallucination": set(CONFIG["HALLUCINATION_CONDITIONS"]),
        "nontarget": set(CONFIG["NONTARGET_CONDITIONS"]),
    }

    rows = []
    for policy, sub in df.groupby("policy"):
        row = {
            "policy": policy,
            "n_total": int(len(sub)),
            "mean_delta_R_all": float(sub["delta_R"].mean()),
            "clean_rate_all": float(sub["pred_clean"].mean()),
        }
        for name, condset in groups.items():
            part = sub[sub["condition"].isin(condset)]
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

        row["target_specificity_delta_R"] = row["mean_delta_R_target_all"] - row["mean_delta_R_nontarget"]
        row["target_specificity_clean_gain"] = row["clean_rate_gain_target_all"] - row["clean_rate_gain_nontarget"]
        rows.append(row)

    policy_summary = pd.DataFrame(rows).sort_values("target_specificity_delta_R", ascending=False)
    policy_summary.to_csv(out_dir / "da_asa2a_policy_summary.csv", index=False, encoding="utf-8-sig")

    cond_rows = []
    for (policy, cond), sub in df.groupby(["policy", "condition"]):
        cond_rows.append({
            "policy": policy,
            "condition": cond,
            "n": int(len(sub)),
            "mean_delta_R": float(sub["delta_R"].mean()),
            "mean_R_final": float(sub["R_final"].mean()),
            "clean_rate": float(sub["pred_clean"].mean()),
            "baseline_clean_rate": float(sub["baseline_pred_clean"].mean()),
            "clean_rate_gain": float(sub["pred_clean"].mean() - sub["baseline_pred_clean"].mean()),
        })
    condition_summary = pd.DataFrame(cond_rows)
    condition_summary.to_csv(out_dir / "da_asa2a_condition_summary.csv", index=False, encoding="utf-8-sig")

    # verdict
    verdict_rows = []
    fixed = policy_summary[policy_summary["policy"].str.startswith("fixed_")]
    generalized = policy_summary[policy_summary["policy"].str.contains("generalized|condition_aware", regex=True)]

    if len(policy_summary):
        best = policy_summary.iloc[0]
        verdict_rows.append({
            "claim": "Best generalized/condition-aware policy improves target margin more than non-target.",
            "best_policy": best["policy"],
            "target_specificity_delta_R": float(best["target_specificity_delta_R"]),
            "mean_delta_R_target_all": float(best["mean_delta_R_target_all"]),
            "mean_delta_R_nontarget": float(best["mean_delta_R_nontarget"]),
            "clean_rate_target_all": float(best["clean_rate_target_all"]),
            "clean_rate_nontarget": float(best["clean_rate_nontarget"]),
            "verdict": "PASS_TARGET_MARGIN_SPECIFICITY" if best["target_specificity_delta_R"] > 0 else "FAIL_TARGET_MARGIN_SPECIFICITY",
        })

    if len(fixed) and len(generalized):
        best_fixed = fixed.sort_values("target_specificity_delta_R", ascending=False).iloc[0]
        best_gen = generalized.sort_values("target_specificity_delta_R", ascending=False).iloc[0]
        verdict_rows.append({
            "claim": "Best generalized policy beats best fixed policy on target margin specificity.",
            "best_generalized_policy": best_gen["policy"],
            "best_fixed_policy": best_fixed["policy"],
            "generalized_specificity_delta_R": float(best_gen["target_specificity_delta_R"]),
            "fixed_specificity_delta_R": float(best_fixed["target_specificity_delta_R"]),
            "gain": float(best_gen["target_specificity_delta_R"] - best_fixed["target_specificity_delta_R"]),
            "verdict": "PASS_GENERALIZED_BEATS_FIXED" if best_gen["target_specificity_delta_R"] > best_fixed["target_specificity_delta_R"] else "FAIL_GENERALIZED_NOT_BEAT_FIXED",
        })

    # Check hallucination protection
    if len(generalized):
        best_h = generalized.sort_values("clean_rate_gain_hallucination", ascending=False).iloc[0]
        verdict_rows.append({
            "claim": "Generalized policy avoids hallucination_like degradation.",
            "best_policy_for_hallucination": best_h["policy"],
            "clean_rate_gain_hallucination": float(best_h["clean_rate_gain_hallucination"]),
            "mean_delta_R_hallucination": float(best_h["mean_delta_R_hallucination"]),
            "verdict": "PASS_HALLUCINATION_NOT_DEGRADED" if best_h["clean_rate_gain_hallucination"] >= 0 else "FAIL_HALLUCINATION_DEGRADED",
        })

    verdict = pd.DataFrame(verdict_rows)
    verdict.to_csv(out_dir / "da_asa2a_verdict_summary.csv", index=False, encoding="utf-8-sig")
    return policy_summary, condition_summary, verdict


def main():
    np.random.seed(CONFIG["RANDOM_SEED"])
    torch.manual_seed(CONFIG["RANDOM_SEED"])

    out_dir = ensure_dir(CONFIG["OUTPUT_DIR"])
    with open(out_dir / "da_asa2a_config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)

    df = build_prompt_dataset()
    df.to_csv(out_dir / "da_asa2a_generated_prompts.csv", index=False, encoding="utf-8-sig")

    model, tokenizer = load_model_and_tokenizer(CONFIG["MODEL_PATH"], CONFIG["DEVICE"], CONFIG["DTYPE"])

    tokenize_candidates(df, tokenizer).to_csv(out_dir / "da_asa2a_candidate_tokenization.csv", index=False, encoding="utf-8-sig")

    hidden, baseline = forward_collect_hidden_and_scores(
        model=model,
        tokenizer=tokenizer,
        df=df,
        layers=CONFIG["COLLECT_LAYERS"],
        batch_size=CONFIG["BATCH_SIZE"],
        max_length=CONFIG["MAX_LENGTH"],
        device=CONFIG["DEVICE"],
    )
    baseline.to_csv(out_dir / "da_asa2a_baseline_scores.csv", index=False, encoding="utf-8-sig")

    vectors = build_vectors(df, hidden, baseline, out_dir)
    policies = generate_policies()
    with open(out_dir / "da_asa2a_policies.json", "w", encoding="utf-8") as f:
        json.dump(policies, f, ensure_ascii=False, indent=2)

    results = run_policy_eval(model, tokenizer, df, vectors, policies, out_dir)
    policy_summary, condition_summary, verdict = summarize(results, baseline, out_dir)

    print("\n[DONE] Outputs saved to:", out_dir.resolve())
    print("\n[POLICY SUMMARY]")
    print(policy_summary.to_string(index=False))
    print("\n[VERDICT]")
    print(verdict.to_string(index=False))

    print("\n[KEY FILES]")
    for name in [
        "da_asa2a_generated_prompts.csv",
        "da_asa2a_candidate_tokenization.csv",
        "da_asa2a_baseline_scores.csv",
        "da_asa2a_steering_vector_summary.csv",
        "da_asa2a_policies.json",
        "da_asa2a_policy_results.csv",
        "da_asa2a_policy_summary.csv",
        "da_asa2a_condition_summary.csv",
        "da_asa2a_verdict_summary.csv",
    ]:
        print(" -", out_dir / name)


if __name__ == "__main__":
    main()
