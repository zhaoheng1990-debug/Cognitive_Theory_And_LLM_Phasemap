# -*- coding: utf-8 -*-
r"""
GPT_216_DA_ASA_2C_policy_selection_rule_audit.py

DA-ASA-2C: Policy Selection Rule Audit

背景：
DA-ASA-2B 已经说明：
1. core_closure family 可控：
   - closure_exception / closure_negation: moderate to_update
   - closure_override: three-stage correction
2. competition_equal_evidence 可控：
   - early to_update + commit order_precursor
3. competition_source_claim 最好不干预：
   - fixed_to_update 会破坏 source_claim
4. hallucination_like 当前最好不干预：
   - to_update / suppress_wrong / to_stable 都可能无益或有害

因此 2C 不再增加新 steering，而是验证：
    Condition / mechanism / attractor family -> policy selection

核心策略：
    if condition in core_closure:
        closure_exception -> to_update 0.15
        closure_negation  -> to_update 0.15
        closure_override  -> three-stage:
            L15-L17 to_update 0.20
            L18-L19 suppress_override 0.15
            L23-L25 order_precursor 0.20

    elif condition == competition_equal_evidence:
        early to_update 0.15 + commit order_precursor 0.15

    elif condition == competition_source_claim:
        no intervention

    elif condition == hallucination_like:
        no intervention

    else:
        no intervention

对照：
    no_intervention
    fixed_to_update_a015
    fixed_to_update_a020
    fixed_three_stage_all
    naive_generalized_all
    policy_selection_rule_v1
    policy_selection_rule_v2_conservative
    policy_selection_rule_v3_aggressive_equal

输出：
    da_asa2c_outputs/
        da_asa2c_policy_summary.csv
        da_asa2c_condition_summary.csv
        da_asa2c_verdict_summary.csv
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")


CONFIG = {
    "MODEL_PATH": r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main",
    "MODEL_PATH_FALLBACK": r"D:\model\models--Qwen--Qwen2.5B-Instruct\main",

    "OUTPUT_DIR": r"da_asa2c_outputs",

    "COLLECT_LAYERS": list(range(15, 26)),
    "EARLY_LAYERS": [15, 16, 17],
    "MID_LAYERS": [18, 19],
    "COMMIT_20_22": [20, 21, 22],
    "COMMIT_23_25": [23, 24, 25],
    "ALL_STEER_LAYERS": list(range(15, 26)),

    "BATCH_SIZE": 4,
    "MAX_LENGTH": 224,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "DTYPE": "auto",
    "RANDOM_SEED": 42,
    "USE_MEAN_HIDDEN_NORM_SCALE": True,

    "CORE_CLOSURE_TARGETS": ["closure_override", "closure_exception", "closure_negation"],
    "EQUAL_TARGET": ["competition_equal_evidence"],
    "SOURCE_TARGET": ["competition_source_claim"],
    "HALLUCINATION_TARGET": ["hallucination_like"],

    "TARGET_CONDITIONS": [
        "closure_override", "closure_exception", "closure_negation",
        "competition_equal_evidence", "competition_source_claim",
        "hallucination_like"
    ],
    "NONTARGET_CONDITIONS": ["stable_clean", "stable_redundant", "weak_distractor", "competition_direct"],

    "SAFE_STABLE_CONDITIONS": ["stable_clean", "stable_redundant"],
    "SAFE_UPDATE_CONDITIONS": ["closure_update", "closure_temporal", "closure_authority"],
    "WRONG_ATTRACTOR_CONDITIONS": ["closure_override", "hallucination_like"],
    "OVERRIDE_ATTRACTOR_CONDITIONS": ["closure_override"],
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
    add("competition_equal_evidence", "competition_equal_evidence", "competition_high_risk",
        "Evidence 1: {A} belongs to {B}, and {B} is associated with {C}. Evidence 2: {A} is associated with {E}. Both evidence sources are equally reliable. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")
    add("competition_source_claim", "competition_source_claim", "source_binding",
        "A source claims that {A} is associated with {E}. Another source says {A} belongs to {B}, and {B} is associated with {C}. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")

    add("closure_update", "closure", "closure_collapse",
        "Old rule: {B} is associated with {C}. New rule: in this graph, {B} is associated with {E}. {A} belongs to {B}. Use the new rule. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")
    add("closure_temporal", "closure", "closure_collapse",
        "Before 2020, {B} was associated with {C}. After 2020, {B} is associated with {E}. The current year is 2024. {A} belongs to {B}. Question: Which label is {A} associated with now, {C} or {E}? Answer with only one label.")
    add("closure_authority", "closure", "closure_collapse",
        "General note: {B} is associated with {C}. Verified registry: {B} is associated with {E}. {A} belongs to {B}. Follow the verified registry. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")

    add("closure_override", "closure_override", "closure_rule_override",
        "General rule: items belonging to {B} are associated with {C}. Override rule: for this task, items belonging to {B} are associated with {E}. {A} belongs to {B}. Use the override rule. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")
    add("closure_exception", "closure_exception", "exception_binding",
        "General rule: items belonging to {B} are associated with {C}. Exception: {A} is a special case associated with {E}. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")
    add("closure_negation", "closure_negation", "closure_collapse",
        "{A} belongs to {B}. It is false that {B} is associated with {C}. Instead, {B} is associated with {E}. Question: Which label is {A} associated with, {C} or {E}? Answer with only one label.")
    add("hallucination_like", "hallucination_like", "wrong_closure",
        "A misleading summary says: {A} belongs to {B}; {B} is connected to {E}; therefore {A} is associated with {E}. The original background sometimes says {B} is associated with {C}. Question: Which label is {A} associated with in the misleading summary, {C} or {E}? Answer with only one label.")

    rows, sid = [], 0
    for A, B, C, E in pairs:
        for t in templates:
            rows.append({
                "id": f"daasa2c_{sid:05d}",
                "graph_id": f"{A}_{B}_{C}_{E}",
                "A": A, "B": B, "C_clean": C, "E_conflict": E,
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

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_torch_dtype(name):
    if name == "float16": return torch.float16
    if name == "bfloat16": return torch.bfloat16
    if name == "float32": return torch.float32
    return None


def load_model_and_tokenizer(model_path, device, dtype_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if not Path(model_path).exists():
        model_path = CONFIG["MODEL_PATH_FALLBACK"]

    torch_dtype = get_torch_dtype(dtype_name)
    kwargs = {"local_files_only": True, "output_hidden_states": True}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    elif device == "cuda":
        kwargs["torch_dtype"] = torch.float16

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **kwargs)
    model.eval()
    model.to(device)
    return model, tok


def get_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Cannot locate decoder layers.")


def first_token_id(tok, text):
    ids = tok.encode(" " + str(text), add_special_tokens=False)
    if not ids:
        ids = tok.encode(str(text), add_special_tokens=False)
    return int(ids[0])


def tokenize_candidates(df, tok):
    rows = []
    for _, r in df.drop_duplicates(["C_clean", "E_conflict"]).iterrows():
        C, E = str(r.C_clean), str(r.E_conflict)
        c_ids = tok.encode(" " + C, add_special_tokens=False)
        e_ids = tok.encode(" " + E, add_special_tokens=False)
        rows.append({
            "C_clean": C, "E_conflict": E,
            "C_first_token_id": first_token_id(tok, C),
            "E_first_token_id": first_token_id(tok, E),
            "C_ids_with_space": " ".join(map(str, c_ids)),
            "E_ids_with_space": " ".join(map(str, e_ids)),
            "C_n_tokens": len(c_ids), "E_n_tokens": len(e_ids),
        })
    return pd.DataFrame(rows)


@torch.no_grad()
def collect(model, tok, df):
    all_hidden = {l: [] for l in CONFIG["COLLECT_LAYERS"]}
    rows = []
    for start in range(0, len(df), CONFIG["BATCH_SIZE"]):
        batch = df.iloc[start:start+CONFIG["BATCH_SIZE"]].reset_index(drop=True)
        enc = tok(batch.prompt.tolist(), return_tensors="pt", padding=True, truncation=True, max_length=CONFIG["MAX_LENGTH"])
        input_ids = enc.input_ids.to(CONFIG["DEVICE"])
        attn = enc.attention_mask.to(CONFIG["DEVICE"])
        last_pos = attn.sum(dim=1) - 1

        out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        logits = out.logits

        for bi in range(input_ids.shape[0]):
            pos = int(last_pos[bi].item())
            C, E = str(batch.loc[bi, "C_clean"]), str(batch.loc[bi, "E_conflict"])
            c_id, e_id = first_token_id(tok, C), first_token_id(tok, E)
            fl = logits[bi, pos, :].float()
            R = float(fl[c_id].item() - fl[e_id].item())
            rows.append({
                "row_index": start+bi,
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
            for l in CONFIG["COLLECT_LAYERS"]:
                idx = l + 1
                if idx < len(hs):
                    all_hidden[l].append(hs[idx][bi, pos, :].detach().float().cpu().numpy().astype(np.float32))
        print(f"[COLLECT] {min(start+CONFIG['BATCH_SIZE'], len(df))}/{len(df)}")
    hidden = {l: np.vstack(v) for l, v in all_hidden.items()}
    return hidden, pd.DataFrame(rows)


# =========================
# Vectors and policies
# =========================

def build_vectors(df, hidden, baseline, out_dir):
    cond = df.condition.values
    stable_idx = np.where(np.isin(cond, CONFIG["SAFE_STABLE_CONDITIONS"]))[0]
    update_idx = np.where(np.isin(cond, ["closure_update", "closure_temporal", "closure_authority"]))[0]
    wrong_idx = np.where(np.isin(cond, ["closure_override", "hallucination_like"]))[0]
    override_idx = np.where(cond == "closure_override")[0]
    halluc_idx = np.where(cond == "hallucination_like")[0]
    source_idx = np.where(cond == "competition_source_claim")[0]
    equal_idx = np.where(cond == "competition_equal_evidence")[0]

    clean_pred_idx = baseline[baseline.baseline_pred_clean == 1].row_index.values
    conflict_pred_idx = baseline[baseline.baseline_pred_clean == 0].row_index.values
    if len(clean_pred_idx) == 0 or len(conflict_pred_idx) == 0:
        clean_pred_idx = stable_idx
        conflict_pred_idx = override_idx

    modes = [
        "to_update", "to_stable", "suppress_override", "suppress_wrong",
        "order_precursor", "boundary_push",
        "source_to_stable", "equal_to_update"
    ]
    vectors = {m: {} for m in modes}
    rows = []
    for l, H in hidden.items():
        stable = H[stable_idx].mean(axis=0)
        update = H[update_idx].mean(axis=0)
        wrong = H[wrong_idx].mean(axis=0)
        override = H[override_idx].mean(axis=0)
        halluc = H[halluc_idx].mean(axis=0)
        source = H[source_idx].mean(axis=0)
        equal = H[equal_idx].mean(axis=0)
        clean = H[clean_pred_idx].mean(axis=0)
        conflict = H[conflict_pred_idx].mean(axis=0)
        scale = float(np.linalg.norm(H, axis=1).mean())

        raw = {
            "to_update": update - wrong,
            "to_stable": stable - wrong,
            "suppress_override": stable - override,
            "suppress_wrong": stable - halluc,
            "order_precursor": clean - conflict,
            "boundary_push": clean - conflict,
            "source_to_stable": stable - source,
            "equal_to_update": update - equal,
        }
        for mode, v in raw.items():
            norm = float(np.linalg.norm(v))
            unit = v / (norm + 1e-12)
            vectors[mode][l] = {"unit": unit.astype(np.float32), "norm": norm, "hidden_norm_mean": scale}
            rows.append({"layer": l, "mode": mode, "vector_norm": norm, "hidden_norm_mean": scale, "ratio": norm/(scale+1e-12)})
    pd.DataFrame(rows).to_csv(out_dir/"da_asa2c_steering_vector_summary.csv", index=False, encoding="utf-8-sig")
    return vectors


def layers_for_group(g):
    if g == "early": return set(CONFIG["EARLY_LAYERS"])
    if g == "mid": return set(CONFIG["MID_LAYERS"])
    if g == "commit23": return set(CONFIG["COMMIT_23_25"])
    if g == "commit20": return set(CONFIG["COMMIT_20_22"])
    if g == "all": return set(CONFIG["ALL_STEER_LAYERS"])
    return set()


def generate_policies():
    core_closure = {
        "closure_exception": [("all", "to_update", 0.15)],
        "closure_negation": [("all", "to_update", 0.15)],
        "closure_override": [("early", "to_update", 0.20), ("mid", "suppress_override", 0.15), ("commit23", "order_precursor", 0.20)],
    }
    equal_light = [("early", "to_update", 0.10), ("commit23", "order_precursor", 0.10)]
    equal_strong = [("early", "to_update", 0.15), ("commit23", "order_precursor", 0.15)]

    policies = {
        "no_intervention": {"*": []},
        "fixed_to_update_a015": {"*": [("all", "to_update", 0.15)]},
        "fixed_to_update_a020": {"*": [("all", "to_update", 0.20)]},
        "fixed_three_stage_all": {"*": [("early", "to_update", 0.20), ("mid", "suppress_override", 0.15), ("commit23", "order_precursor", 0.20)]},

        "policy_selection_rule_v1": {
            **core_closure,
            "competition_equal_evidence": equal_light,
            "competition_source_claim": [],
            "hallucination_like": [],
            "*": []
        },
        "policy_selection_rule_v2_strong_equal": {
            **core_closure,
            "competition_equal_evidence": equal_strong,
            "competition_source_claim": [],
            "hallucination_like": [],
            "*": []
        },
        "policy_selection_rule_v3_conservative": {
            **core_closure,
            "competition_equal_evidence": [],
            "competition_source_claim": [],
            "hallucination_like": [],
            "*": []
        },
        "policy_selection_rule_v4_halluc_stable": {
            **core_closure,
            "competition_equal_evidence": equal_light,
            "competition_source_claim": [],
            "hallucination_like": [("all", "to_stable", 0.15)],
            "*": []
        },
        "policy_selection_rule_v5_source_weak": {
            **core_closure,
            "competition_equal_evidence": equal_light,
            "competition_source_claim": [("early", "source_to_stable", 0.05)],
            "hallucination_like": [],
            "*": []
        },
    }
    return policies


def policy_action(policies, pname, condition):
    pol = policies[pname]
    if condition in pol: return pol[condition]
    return pol.get("*", [])


@torch.no_grad()
def run_policy_eval(model, tok, df, vectors, policies, out_dir):
    layers_module = get_decoder_layers(model)
    rows = []
    for pname in policies.keys():
        print(f"[POLICY] {pname}")
        for start in range(0, len(df), CONFIG["BATCH_SIZE"]):
            batch = df.iloc[start:start+CONFIG["BATCH_SIZE"]].reset_index(drop=True)
            enc = tok(batch.prompt.tolist(), return_tensors="pt", padding=True, truncation=True, max_length=CONFIG["MAX_LENGTH"])
            input_ids = enc.input_ids.to(CONFIG["DEVICE"])
            attn = enc.attention_mask.to(CONFIG["DEVICE"])
            last_pos = attn.sum(dim=1)-1

            def make_hook(layer_idx):
                def hook(module, inputs, output):
                    if isinstance(output, tuple):
                        hidden = output[0]; rest = output[1:]
                    else:
                        hidden = output; rest = None
                    h2 = hidden.clone()
                    for bi in range(h2.shape[0]):
                        cond = batch.loc[bi, "condition"]
                        for group, mode, alpha in policy_action(policies, pname, cond):
                            if alpha == 0 or mode == "none": continue
                            if layer_idx not in layers_for_group(group): continue
                            if mode not in vectors or layer_idx not in vectors[mode]: continue
                            direction = torch.tensor(vectors[mode][layer_idx]["unit"], device=h2.device, dtype=h2.dtype)
                            scale = vectors[mode][layer_idx]["hidden_norm_mean"] if CONFIG["USE_MEAN_HIDDEN_NORM_SCALE"] else 1.0
                            pos = int(last_pos[bi].item())
                            h2[bi, pos, :] = h2[bi, pos, :] + alpha * scale * direction
                    if rest is not None: return (h2,) + rest
                    return h2
                return hook

            handles = []
            for l in CONFIG["ALL_STEER_LAYERS"]:
                if l < len(layers_module):
                    handles.append(layers_module[l].register_forward_hook(make_hook(l)))

            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            logits = out.logits
            for h in handles: h.remove()

            for bi in range(input_ids.shape[0]):
                pos = int(last_pos[bi].item())
                C, E = str(batch.loc[bi, "C_clean"]), str(batch.loc[bi, "E_conflict"])
                c_id, e_id = first_token_id(tok, C), first_token_id(tok, E)
                fl = logits[bi, pos, :].float()
                R = float(fl[c_id].item() - fl[e_id].item())
                rows.append({
                    "row_index": start+bi,
                    "id": batch.loc[bi, "id"],
                    "graph_id": batch.loc[bi, "graph_id"],
                    "condition": batch.loc[bi, "condition"],
                    "mechanism": batch.loc[bi, "mechanism"],
                    "risk_regime": batch.loc[bi, "risk_regime"],
                    "policy": pname,
                    "applied_actions": json.dumps(policy_action(policies, pname, batch.loc[bi, "condition"])),
                    "R_final": R,
                    "pred_clean": int(R > 0),
                })
            print(f"[POLICY] {pname}: {min(start+CONFIG['BATCH_SIZE'], len(df))}/{len(df)}")
    res = pd.DataFrame(rows)
    res.to_csv(out_dir/"da_asa2c_policy_results.csv", index=False, encoding="utf-8-sig")
    return res


def summarize(results, baseline, out_dir):
    df = results.merge(baseline[["id", "baseline_R_final", "baseline_pred_clean"]], on="id", how="left")
    df["delta_R"] = df.R_final - df.baseline_R_final

    groups = {
        "target_all": set(CONFIG["TARGET_CONDITIONS"]),
        "core_closure": set(CONFIG["CORE_CLOSURE_TARGETS"]),
        "equal": set(CONFIG["EQUAL_TARGET"]),
        "source": set(CONFIG["SOURCE_TARGET"]),
        "hallucination": set(CONFIG["HALLUCINATION_TARGET"]),
        "nontarget": set(CONFIG["NONTARGET_CONDITIONS"]),
    }
    rows = []
    for pol, sub in df.groupby("policy"):
        row = {"policy": pol, "n_total": int(len(sub)), "mean_delta_R_all": float(sub.delta_R.mean()), "clean_rate_all": float(sub.pred_clean.mean())}
        for name, condset in groups.items():
            part = sub[sub.condition.isin(condset)]
            if len(part):
                row[f"n_{name}"] = int(len(part))
                row[f"mean_delta_R_{name}"] = float(part.delta_R.mean())
                row[f"clean_rate_{name}"] = float(part.pred_clean.mean())
                row[f"baseline_clean_rate_{name}"] = float(part.baseline_pred_clean.mean())
                row[f"clean_rate_gain_{name}"] = row[f"clean_rate_{name}"] - row[f"baseline_clean_rate_{name}"]
            else:
                row[f"n_{name}"] = 0
                row[f"mean_delta_R_{name}"] = np.nan
                row[f"clean_rate_{name}"] = np.nan
                row[f"baseline_clean_rate_{name}"] = np.nan
                row[f"clean_rate_gain_{name}"] = np.nan
        row["target_specificity_delta_R"] = row["mean_delta_R_target_all"] - row["mean_delta_R_nontarget"]
        row["target_specificity_clean_gain"] = row["clean_rate_gain_target_all"] - row["clean_rate_gain_nontarget"]
        # safety score: target improvement minus penalties for nontarget/source/hallucination degradation
        source_penalty = max(0.0, -row["clean_rate_gain_source"])
        halluc_penalty = max(0.0, -row["clean_rate_gain_hallucination"])
        nontarget_penalty = max(0.0, -row["clean_rate_gain_nontarget"])
        row["safety_adjusted_gain"] = row["clean_rate_gain_core_closure"] + row["clean_rate_gain_equal"] - source_penalty - halluc_penalty - nontarget_penalty
        rows.append(row)
    policy_summary = pd.DataFrame(rows).sort_values("safety_adjusted_gain", ascending=False)
    policy_summary.to_csv(out_dir/"da_asa2c_policy_summary.csv", index=False, encoding="utf-8-sig")

    cond_rows = []
    for (pol, cond), sub in df.groupby(["policy", "condition"]):
        cond_rows.append({
            "policy": pol,
            "condition": cond,
            "n": int(len(sub)),
            "mean_delta_R": float(sub.delta_R.mean()),
            "mean_R_final": float(sub.R_final.mean()),
            "clean_rate": float(sub.pred_clean.mean()),
            "baseline_clean_rate": float(sub.baseline_pred_clean.mean()),
            "clean_rate_gain": float(sub.pred_clean.mean() - sub.baseline_pred_clean.mean()),
        })
    condition_summary = pd.DataFrame(cond_rows)
    condition_summary.to_csv(out_dir/"da_asa2c_condition_summary.csv", index=False, encoding="utf-8-sig")

    verdict_rows = []
    best = policy_summary.iloc[0]
    fixed = policy_summary[policy_summary.policy.str.startswith("fixed_")]
    rule = policy_summary[policy_summary.policy.str.startswith("policy_selection")]
    if len(fixed) and len(rule):
        best_fixed = fixed.sort_values("safety_adjusted_gain", ascending=False).iloc[0]
        best_rule = rule.sort_values("safety_adjusted_gain", ascending=False).iloc[0]
        verdict_rows.append({
            "claim": "Best policy-selection rule beats best fixed policy by safety-adjusted gain.",
            "best_rule": best_rule.policy,
            "best_fixed": best_fixed.policy,
            "rule_safety_adjusted_gain": float(best_rule.safety_adjusted_gain),
            "fixed_safety_adjusted_gain": float(best_fixed.safety_adjusted_gain),
            "gain": float(best_rule.safety_adjusted_gain - best_fixed.safety_adjusted_gain),
            "verdict": "PASS_RULE_BEATS_FIXED" if best_rule.safety_adjusted_gain > best_fixed.safety_adjusted_gain else "FAIL_RULE_NOT_BEAT_FIXED",
        })
    verdict_rows.append({
        "claim": "Best policy protects source_claim and hallucination_like while improving core closure.",
        "best_policy": best.policy,
        "clean_rate_gain_core_closure": float(best.clean_rate_gain_core_closure),
        "clean_rate_gain_equal": float(best.clean_rate_gain_equal),
        "clean_rate_gain_source": float(best.clean_rate_gain_source),
        "clean_rate_gain_hallucination": float(best.clean_rate_gain_hallucination),
        "clean_rate_gain_nontarget": float(best.clean_rate_gain_nontarget),
        "safety_adjusted_gain": float(best.safety_adjusted_gain),
        "verdict": "PASS_SAFE_POLICY_SELECTION" if (best.clean_rate_gain_core_closure > 0 and best.clean_rate_gain_source >= 0 and best.clean_rate_gain_hallucination >= 0 and best.clean_rate_gain_nontarget >= 0) else "FAIL_SAFE_POLICY_SELECTION",
    })
    verdict = pd.DataFrame(verdict_rows)
    verdict.to_csv(out_dir/"da_asa2c_verdict_summary.csv", index=False, encoding="utf-8-sig")
    return policy_summary, condition_summary, verdict


def main():
    np.random.seed(CONFIG["RANDOM_SEED"])
    torch.manual_seed(CONFIG["RANDOM_SEED"])

    out_dir = ensure_dir(CONFIG["OUTPUT_DIR"])
    with open(out_dir/"da_asa2c_config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)

    df = build_prompt_dataset()
    df.to_csv(out_dir/"da_asa2c_generated_prompts.csv", index=False, encoding="utf-8-sig")

    model, tok = load_model_and_tokenizer(CONFIG["MODEL_PATH"], CONFIG["DEVICE"], CONFIG["DTYPE"])
    tokenize_candidates(df, tok).to_csv(out_dir/"da_asa2c_candidate_tokenization.csv", index=False, encoding="utf-8-sig")

    hidden, baseline = collect(model, tok, df)
    baseline.to_csv(out_dir/"da_asa2c_baseline_scores.csv", index=False, encoding="utf-8-sig")

    vectors = build_vectors(df, hidden, baseline, out_dir)
    policies = generate_policies()
    with open(out_dir/"da_asa2c_policies.json", "w", encoding="utf-8") as f:
        json.dump(policies, f, ensure_ascii=False, indent=2)

    results = run_policy_eval(model, tok, df, vectors, policies, out_dir)
    ps, cs, vd = summarize(results, baseline, out_dir)

    print("\n[DONE] Outputs saved to:", out_dir.resolve())
    print("\n[POLICY SUMMARY]")
    print(ps.to_string(index=False))
    print("\n[VERDICT]")
    print(vd.to_string(index=False))


if __name__ == "__main__":
    main()
