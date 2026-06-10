# -*- coding: utf-8 -*-
r"""
GPT_213_DA_ASA_1D_commit_stage_override_correction.py

DA-ASA-1D: Commit-Stage Override Correction

背景：
DA-ASA-1C 证明：
- multi-stage steering 在 closure_override margin specificity 上超过 single-stage；
- 最佳策略 override_only_update020_then_suppress015 能把 override 从深 wrong attractor 拉到边界附近；
- 但 clean-rate 仍停在 0.5，说明还缺 final / commit-stage correction。

本实验目标：
在 1C 最佳前两段基础上增加第三段：
    Stage 1: L15-L17 to_update
    Stage 2: L18-L19 suppress_override
    Stage 3: L20-L22 或 L23-L25 commit correction

核心问题：
    能否把 closure_override 从 R≈边界附近推过 R=0，
    完成 clean commitment？

候选 Stage 3：
1. commit_to_stable:
       mean_H(stable_clean, L20-L25) - mean_H(override, L20-L25)
2. commit_to_update:
       mean_H(update_safe, L20-L25) - mean_H(override, L20-L25)
3. order_precursor:
       用 clean-vs-conflict final margin 的 ridge-like hidden direction
       简化为 mean_H(clean-pred samples) - mean_H(conflict-pred samples)
4. boundary_push:
       用 R_final>0 与 R_final<0 的均值差，直接推过 C/E 边界

比较策略：
    no_intervention
    1C_best_update020_suppress015
    1D_commit_stable_20_22
    1D_commit_stable_23_25
    1D_commit_update_20_22
    1D_commit_update_23_25
    1D_order_precursor_20_22
    1D_order_precursor_23_25
    1D_boundary_push_20_22
    1D_boundary_push_23_25

输出：
    da_asa1d_outputs/
        da_asa1d_policy_summary.csv
        da_asa1d_condition_summary.csv
        da_asa1d_verdict_summary.csv
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

    "OUTPUT_DIR": r"da_asa1d_outputs",

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

    "TARGET_CONDITIONS": ["closure_override"],
    "SECONDARY_TARGET_CONDITIONS": ["closure_exception", "closure_negation"],
    "NONTARGET_CONDITIONS": ["stable_clean", "stable_redundant", "weak_distractor", "competition_direct"],
    "HALLUCINATION_CONDITIONS": ["hallucination_like"],

    "WRONG_ATTRACTOR_CONDITIONS": ["closure_override", "hallucination_like"],
    "OVERRIDE_ATTRACTOR_CONDITIONS": ["closure_override"],
    "SAFE_STABLE_CONDITIONS": ["stable_clean", "stable_redundant"],
    "SAFE_UPDATE_CONDITIONS": ["closure_update", "closure_temporal", "closure_authority"],

    # Stage strengths. Conservative because commit-stage interventions can be risky.
    "STAGE1_ALPHA": 0.20,       # early to_update
    "STAGE2_ALPHA": 0.15,       # late suppress_override
    "COMMIT_ALPHAS": [0.05, 0.10, 0.15, 0.20],

    # Policies are generated programmatically.
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
                "id": f"daasa1d_{sid:05d}",
                "graph_id": f"{A}_{B}_{C}_{E}",
                "A": A, "B": B, "C_clean": C, "E_conflict": E,
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


def tokenize_candidates(df, tokenizer):
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


def build_vectors(df, hidden_np, baseline, out_dir):
    cond = df["condition"].values
    override_idx = np.where(cond == "closure_override")[0]
    wrong_idx = np.where(np.isin(cond, CONFIG["WRONG_ATTRACTOR_CONDITIONS"]))[0]
    stable_idx = np.where(np.isin(cond, CONFIG["SAFE_STABLE_CONDITIONS"]))[0]
    update_idx = np.where(np.isin(cond, CONFIG["SAFE_UPDATE_CONDITIONS"]))[0]

    clean_pred_idx = baseline[baseline["baseline_pred_clean"] == 1]["row_index"].values
    conflict_pred_idx = baseline[baseline["baseline_pred_clean"] == 0]["row_index"].values
    if len(clean_pred_idx) == 0 or len(conflict_pred_idx) == 0:
        # fallback
        clean_pred_idx = stable_idx
        conflict_pred_idx = override_idx

    vectors = {m: {} for m in ["to_update", "to_stable", "suppress_override", "commit_stable", "commit_update", "order_precursor", "boundary_push"]}
    rows = []
    for l, H in hidden_np.items():
        override_mean = H[override_idx].mean(axis=0)
        wrong_mean = H[wrong_idx].mean(axis=0)
        stable_mean = H[stable_idx].mean(axis=0)
        update_mean = H[update_idx].mean(axis=0)
        clean_mean = H[clean_pred_idx].mean(axis=0)
        conflict_mean = H[conflict_pred_idx].mean(axis=0)
        hidden_norm_mean = float(np.linalg.norm(H, axis=1).mean())

        raw_vectors = {
            "to_update": update_mean - wrong_mean,
            "to_stable": stable_mean - wrong_mean,
            "suppress_override": stable_mean - override_mean,
            "commit_stable": stable_mean - override_mean,
            "commit_update": update_mean - override_mean,
            "order_precursor": clean_mean - conflict_mean,
            "boundary_push": clean_mean - conflict_mean,
        }
        for mode, v in raw_vectors.items():
            norm = float(np.linalg.norm(v))
            unit = v / (norm + 1e-12)
            vectors[mode][l] = {"unit": unit.astype(np.float32), "raw": v.astype(np.float32), "norm": norm, "hidden_norm_mean": hidden_norm_mean}
            rows.append({"layer": l, "mode": mode, "vector_norm": norm, "hidden_norm_mean": hidden_norm_mean, "scale_norm_ratio": norm/(hidden_norm_mean+1e-12)})
    pd.DataFrame(rows).to_csv(out_dir / "da_asa1d_steering_vector_summary.csv", index=False, encoding="utf-8-sig")
    return vectors


def generate_policies():
    policies = {
        "no_intervention": {"*": []},
        "baseline_1c_best_update020_suppress015": {
            "closure_override": [("stage1", "to_update", 0.20), ("stage2", "suppress_override", 0.15)],
            "*": []
        },
        "single_to_update_a020": {"*": [("all", "to_update", 0.20)]},
        "single_to_update_a030": {"*": [("all", "to_update", 0.30)]},
    }
    commit_specs = [
        ("commit_stable", "commit20", "COMMIT_20_22"),
        ("commit_stable", "commit23", "COMMIT_23_25"),
        ("commit_update", "commit20", "COMMIT_20_22"),
        ("commit_update", "commit23", "COMMIT_23_25"),
        ("order_precursor", "commit20", "COMMIT_20_22"),
        ("order_precursor", "commit23", "COMMIT_23_25"),
        ("boundary_push", "commit20", "COMMIT_20_22"),
        ("boundary_push", "commit23", "COMMIT_23_25"),
    ]
    for mode, label, group in commit_specs:
        for a in CONFIG["COMMIT_ALPHAS"]:
            pname = f"1d_update020_suppress015_{mode}_{label}_a{str(a).replace('.','')}"
            policies[pname] = {
                "closure_override": [
                    ("stage1", "to_update", 0.20),
                    ("stage2", "suppress_override", 0.15),
                    (group, mode, a),
                ],
                "*": []
            }
    # full condition-aware with best commit candidates, included as candidates
    for mode in ["commit_stable", "commit_update", "order_precursor", "boundary_push"]:
        pname = f"1d_condition_aware_plus_{mode}_23_a010"
        policies[pname] = {
            "closure_exception": [("all", "to_update", 0.15)],
            "closure_negation": [("all", "to_update", 0.15)],
            "closure_override": [
                ("stage1", "to_update", 0.20),
                ("stage2", "suppress_override", 0.15),
                ("COMMIT_23_25", mode, 0.10),
            ],
            "hallucination_like": [("all", "to_stable", 0.15)],
            "*": []
        }
    return policies


def layers_for_group(group):
    if group == "stage1":
        return set(CONFIG["STAGE1_LAYERS"])
    if group == "stage2":
        return set(CONFIG["STAGE2_LAYERS"])
    if group == "COMMIT_20_22":
        return set(CONFIG["COMMIT_20_22"])
    if group == "COMMIT_23_25":
        return set(CONFIG["COMMIT_23_25"])
    if group == "all":
        return set(CONFIG["ALL_STEER_LAYERS"])
    return set()


def policy_action(policies, policy_name, condition):
    pol = policies[policy_name]
    if condition in pol:
        return pol[condition]
    return pol.get("*", [])


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
    res.to_csv(out_dir / "da_asa1d_policy_results.csv", index=False, encoding="utf-8-sig")
    return res


def summarize(results, baseline, out_dir):
    df = results.merge(baseline[["id", "baseline_R_final", "baseline_pred_clean"]], on="id", how="left")
    df["delta_R"] = df["R_final"] - df["baseline_R_final"]

    groups = {
        "override": set(CONFIG["TARGET_CONDITIONS"]),
        "secondary": set(CONFIG["SECONDARY_TARGET_CONDITIONS"]),
        "hallucination": set(CONFIG["HALLUCINATION_CONDITIONS"]),
        "nontarget": set(CONFIG["NONTARGET_CONDITIONS"]),
    }
    rows = []
    for policy, sub in df.groupby("policy"):
        row = {"policy": policy, "n_total": int(len(sub)), "mean_delta_R_all": float(sub["delta_R"].mean()), "clean_rate_all": float(sub["pred_clean"].mean())}
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
        row["override_specificity_vs_nontarget_delta_R"] = row["mean_delta_R_override"] - row["mean_delta_R_nontarget"]
        row["override_specificity_vs_nontarget_clean_gain"] = row["clean_rate_gain_override"] - row["clean_rate_gain_nontarget"]
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("override_specificity_vs_nontarget_delta_R", ascending=False)
    summary.to_csv(out_dir / "da_asa1d_policy_summary.csv", index=False, encoding="utf-8-sig")

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
    cond_summary = pd.DataFrame(cond_rows)
    cond_summary.to_csv(out_dir / "da_asa1d_condition_summary.csv", index=False, encoding="utf-8-sig")

    fixed = summary[summary["policy"].isin(["single_to_update_a020", "single_to_update_a030", "baseline_1c_best_update020_suppress015"])]
    commit = summary[~summary["policy"].isin(["no_intervention", "single_to_update_a020", "single_to_update_a030", "baseline_1c_best_update020_suppress015"])]
    verdict_rows = []

    if len(commit) and len(fixed):
        best_commit = commit.sort_values("override_specificity_vs_nontarget_delta_R", ascending=False).iloc[0]
        best_fixed = fixed.sort_values("override_specificity_vs_nontarget_delta_R", ascending=False).iloc[0]
        verdict_rows.append({
            "claim": "Commit-stage correction beats best pre-commit/mid-stage policy on override margin specificity.",
            "best_commit_policy": best_commit["policy"],
            "best_fixed_policy": best_fixed["policy"],
            "commit_specificity_delta_R": float(best_commit["override_specificity_vs_nontarget_delta_R"]),
            "fixed_specificity_delta_R": float(best_fixed["override_specificity_vs_nontarget_delta_R"]),
            "gain": float(best_commit["override_specificity_vs_nontarget_delta_R"] - best_fixed["override_specificity_vs_nontarget_delta_R"]),
            "commit_clean_rate_override": float(best_commit["clean_rate_override"]),
            "fixed_clean_rate_override": float(best_fixed["clean_rate_override"]),
            "verdict": "PASS_COMMIT_BEATS_FIXED" if best_commit["override_specificity_vs_nontarget_delta_R"] > best_fixed["override_specificity_vs_nontarget_delta_R"] else "FAIL_COMMIT_NOT_BEAT_FIXED",
        })

        best_clean = commit.sort_values("clean_rate_override", ascending=False).iloc[0]
        verdict_rows.append({
            "claim": "Commit-stage correction improves override clean-rate beyond 0.5 barrier.",
            "best_policy": best_clean["policy"],
            "clean_rate_override": float(best_clean["clean_rate_override"]),
            "baseline_clean_rate_override": float(best_clean["baseline_clean_rate_override"]),
            "clean_rate_gain_override": float(best_clean["clean_rate_gain_override"]),
            "verdict": "PASS_CLEAN_RATE_BREAKS_05" if best_clean["clean_rate_override"] > 0.5 else "FAIL_CLEAN_RATE_STILL_05_OR_LESS",
        })
    verdict = pd.DataFrame(verdict_rows)
    verdict.to_csv(out_dir / "da_asa1d_verdict_summary.csv", index=False, encoding="utf-8-sig")
    return summary, cond_summary, verdict


def main():
    np.random.seed(CONFIG["RANDOM_SEED"])
    torch.manual_seed(CONFIG["RANDOM_SEED"])

    out_dir = ensure_dir(CONFIG["OUTPUT_DIR"])
    with open(out_dir / "da_asa1d_config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)

    df = build_prompt_dataset()
    df.to_csv(out_dir / "da_asa1d_generated_prompts.csv", index=False, encoding="utf-8-sig")

    model, tokenizer = load_model_and_tokenizer(CONFIG["MODEL_PATH"], CONFIG["DEVICE"], CONFIG["DTYPE"])

    tokenize_candidates(df, tokenizer).to_csv(out_dir / "da_asa1d_candidate_tokenization.csv", index=False, encoding="utf-8-sig")

    hidden, baseline = forward_collect_hidden_and_scores(
        model, tokenizer, df, CONFIG["COLLECT_LAYERS"], CONFIG["BATCH_SIZE"], CONFIG["MAX_LENGTH"], CONFIG["DEVICE"]
    )
    baseline.to_csv(out_dir / "da_asa1d_baseline_scores.csv", index=False, encoding="utf-8-sig")

    vectors = build_vectors(df, hidden, baseline, out_dir)
    policies = generate_policies()
    with open(out_dir / "da_asa1d_policies.json", "w", encoding="utf-8") as f:
        json.dump(policies, f, ensure_ascii=False, indent=2)

    results = run_policy_eval(model, tokenizer, df, vectors, policies, out_dir)
    summary, cond_summary, verdict = summarize(results, baseline, out_dir)

    print("\n[DONE] Outputs saved to:", out_dir.resolve())
    print("\n[POLICY SUMMARY]")
    print(summary.head(40).to_string(index=False))
    print("\n[VERDICT]")
    print(verdict.to_string(index=False))

    print("\n[KEY FILES]")
    for name in [
        "da_asa1d_generated_prompts.csv",
        "da_asa1d_candidate_tokenization.csv",
        "da_asa1d_baseline_scores.csv",
        "da_asa1d_steering_vector_summary.csv",
        "da_asa1d_policy_results.csv",
        "da_asa1d_policy_summary.csv",
        "da_asa1d_condition_summary.csv",
        "da_asa1d_verdict_summary.csv",
        "da_asa1d_policies.json",
    ]:
        print(" -", out_dir / name)


if __name__ == "__main__":
    main()
