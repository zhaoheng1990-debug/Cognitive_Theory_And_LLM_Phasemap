# ============================================================
# ASA-8: Closed-Loop / Inertia-Aware Control Audit
#
# Purpose
#   After ASA-7B validated fixed same_combo control, ASA-8 tests whether
#   adaptive gains can improve or safely match the fixed combo.
#
#   Fixed combo:
#       stable mechanism operator A_S(H_l) + position_ridge precursor
#       alpha=0.30, beta=0.30
#
#   Closed-loop combo:
#       alpha_l, beta_l are chosen at hook time from lightweight state probes:
#       DeltaU_hat_l, D_B_hat_l, and velocity/inertia proxy.
#
# Outputs in ./asa8_outputs:
#   asa8_summary.json
#   asa8_verdict.csv
#   asa8_specificity_summary.csv
#   asa8_gain_profile.csv
#   asa8_control_trace.csv
#   asa8_steering_results.csv
#   asa8_operator_audit.csv
#   asa8_probe_audit.csv
# ============================================================

import gc
import json
import random
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

QWEN_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
LLAMA_PATH = r"D:\model\Llama-3.2-1B-Instruct"
GEMMA_PATH = r"D:\model\gemma-2-2b-it"
MODEL_PATHS = {"qwen": QWEN_PATH, "llama": LLAMA_PATH, "gemma": GEMMA_PATH}


@dataclass
class Config:
    model_key: str = "qwen"
    model_path: str = "auto"
    save_dir: str = "./asa8_outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float16"
    seed: int = 42
    n_graphs: int = 96
    max_len: int = 260
    batch_size: int = 4
    test_size: float = 0.30

    operator_layers: Tuple[int, ...] = tuple(range(15, 20))
    precursor_layers: Tuple[int, ...] = (17, 18, 19)
    decision_layers: Tuple[int, ...] = tuple(range(20, 26))

    pca_dim: int = 128
    operator_ridge_alpha: float = 10.0
    probe_ridge_alpha: float = 10.0
    precursor_ridge_alpha: float = 10.0
    min_dir_norm: float = 1e-4

    fixed_alpha: float = 0.30
    fixed_beta: float = 0.30
    alpha_max: float = 0.35
    beta_max: float = 0.35
    safe_gain_cap: float = 0.25

    db_temp: float = 0.40
    du_temp: float = 0.50
    inertia_low_q: float = 0.50
    inertia_temp: float = 0.50
    include_answer_control: bool = True

CFG = Config()

LABEL_CANDIDATES = [
    "Red", "Blue", "Green", "Yellow", "North", "South", "East", "West",
    "Copper", "Silver", "Gold", "Iron", "Circle", "Square", "Triangle", "Star",
    "River", "Mountain", "Forest", "Ocean", "Sun", "Moon", "Cloud", "Stone",
    "Alpha", "Beta", "Gamma", "Delta", "Apple", "Orange", "Lemon", "Pear",
]
ENTITIES = [
    ("Ava", "Bela", "Cora"), ("Darin", "Elo", "Faye"), ("Galen", "Hera", "Ivo"),
    ("Juno", "Kira", "Lio"), ("Mira", "Nero", "Orin"), ("Pia", "Quin", "Rhea"),
    ("Sola", "Taro", "Una"), ("Vera", "Wen", "Xio"), ("Yara", "Zeno", "Nia"),
    ("Orla", "Pavel", "Rin"), ("Nora", "Silas", "Tess"), ("Uma", "Vito", "Willa"),
    ("Xena", "Yuri", "Zara"), ("Iris", "Kai", "Lena"), ("Omar", "Priya", "Quill"),
    ("Ravi", "Sara", "Theo"),
]
REL_WORDS = [
    ("belongs to", "is located at"), ("is assigned to", "maps to"),
    ("is part of", "points to"), ("is grouped under", "has label"),
    ("routes through", "ends at"), ("is linked with", "resolves to"),
]
STABLE_CONDS = ["clean", "redundant", "irrelevant", "paraphrase"]
COMPETITION_CONDS = ["weak_distractor", "competition_balanced", "direct_conflict"]
CLOSURE_CONDS = ["closure_update", "closure_override", "exception_override"]
ALL_CONDS = STABLE_CONDS + COMPETITION_CONDS + CLOSURE_CONDS


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def dtype_from_cfg(cfg):
    if cfg.dtype == "float16" and cfg.device == "cuda": return torch.float16
    if cfg.dtype == "bfloat16" and cfg.device == "cuda": return torch.bfloat16
    return torch.float32


def resolve_model_path(cfg):
    if cfg.model_path == "auto" or "YOUR_SNAPSHOT" in str(cfg.model_path):
        return MODEL_PATHS[cfg.model_key]
    return cfg.model_path


def mechanism_of(cond: str) -> str:
    if cond in STABLE_CONDS: return "stable"
    if cond in COMPETITION_CONDS: return "competition"
    if cond in CLOSURE_CONDS: return "closure"
    raise ValueError(cond)


def continuation_ids(tokenizer, text):
    return tokenizer(" " + text, add_special_tokens=False)["input_ids"]


def normalize_np(v, eps=1e-8):
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < eps: return None, float(n)
    return (v / n).astype(np.float32), float(n)


def safe_corr(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12: return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def select_single_token_labels(tokenizer, save_dir):
    rows, chosen = [], []
    for lab in LABEL_CANDIDATES:
        ids = continuation_ids(tokenizer, lab)
        rows.append({"label": lab, "ids": str(ids), "len": len(ids), "single": int(len(ids) == 1)})
        if len(ids) == 1: chosen.append(lab)
    pd.DataFrame(rows).to_csv(save_dir / "asa8_label_tokenization_audit.csv", index=False, encoding="utf-8-sig")
    if len(chosen) < 12: raise RuntimeError(f"Need >=12 single-token labels; got {chosen}")
    return chosen[:12]


def make_prompt(cond, a, b, d, clean_label, conflict_label, aux_label, relation1, relation2):
    if cond == "clean":
        lines = ["You are given a small relation graph.", f"Fact 1: {a} {relation1} {b}.", f"Fact 2: {b} {relation2} {clean_label}.", f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?", "Answer with exactly one word:"]
    elif cond == "redundant":
        lines = ["You are given a small relation graph.", f"Fact 1: {a} {relation1} {b}.", f"Fact 2: {b} {relation2} {clean_label}.", f"Confirmation: {a} still goes through {b}.", f"Confirmation: {b} still points to {clean_label}.", f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?", "Answer with exactly one word:"]
    elif cond == "irrelevant":
        lines = ["You are given a small relation graph.", f"Fact 1: {a} {relation1} {b}.", f"Fact 2: {b} {relation2} {clean_label}.", f"Irrelevant fact: {d} is associated with {aux_label}.", f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?", "Answer with exactly one word:"]
    elif cond == "paraphrase":
        lines = [f"In this example, {a} reaches {b}.", f"The destination connected to {b} is {clean_label}.", f"Based on those two links, select the label for {a}: {clean_label} or {conflict_label}.", "Answer:"]
    elif cond == "weak_distractor":
        lines = ["You are given a small relation graph.", f"Fact 1: {a} {relation1} {b}.", f"Fact 2: {b} {relation2} {clean_label}.", f"Weak note: some unrelated source mentions {conflict_label}, but does not update the graph.", f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?", "Answer with exactly one word:"]
    elif cond == "competition_balanced":
        lines = ["You are given a small relation graph.", f"Fact 1: {a} {relation1} {b}.", f"Fact 2: {b} {relation2} {clean_label}.", f"Competing fact: {a} is also associated with {conflict_label}.", f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?", "Answer with exactly one word:"]
    elif cond == "direct_conflict":
        lines = ["You are given a small relation graph.", f"Fact 1: {a} {relation1} {b}.", f"Fact 2: {b} {relation2} {clean_label}.", f"Direct conflicting fact: {a} {relation2} {conflict_label}.", f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?", "Answer with exactly one word:"]
    elif cond == "closure_update":
        lines = ["You are given a small relation graph.", f"Fact 1: {a} {relation1} {b}.", f"Old record: {b} {relation2} {clean_label}.", f"Updated record: {b} {relation2} {conflict_label}, not {clean_label}.", f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?", "Answer with exactly one word:"]
    elif cond == "closure_override":
        lines = ["You are given a rule system.", f"General rule: items that {relation1} {b} receive label {clean_label}.", f"Override rule: in this case, items that {relation1} {b} receive label {conflict_label}.", f"Fact: {a} {relation1} {b}.", f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?", "Answer with exactly one word:"]
    elif cond == "exception_override":
        lines = ["You are given a rule system.", f"General rule: items connected to {b} use label {clean_label}.", f"Exception: {a} is a special case and uses label {conflict_label}.", f"Fact: {a} is connected to {b}.", f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?", "Answer with exactly one word:"]
    else:
        raise ValueError(cond)
    return "\n".join(lines)


def build_dataset(tokenizer, cfg, save_dir):
    labels = select_single_token_labels(tokenizer, save_dir)
    rows, graph_count = [], 0
    n_labels = len(labels)
    for ent_i, (a, b, d) in enumerate(ENTITIES):
        for rel_i, (r1, r2) in enumerate(REL_WORDS):
            if graph_count >= cfg.n_graphs: break
            clean_label = labels[(2 * graph_count) % n_labels]
            conflict_label = labels[(2 * graph_count + 1) % n_labels]
            aux_label = labels[(2 * graph_count + 2) % n_labels]
            clean_id = continuation_ids(tokenizer, clean_label)[0]
            conflict_id = continuation_ids(tokenizer, conflict_label)[0]
            for cond in ALL_CONDS:
                rows.append({
                    "prompt_id": f"g{graph_count:03d}_{cond}", "graph_id": graph_count,
                    "entity_id": ent_i, "relation_id": rel_i, "condition": cond,
                    "mechanism": mechanism_of(cond), "clean_label": clean_label,
                    "conflict_label": conflict_label, "clean_token_id": clean_id,
                    "conflict_token_id": conflict_id,
                    "text": make_prompt(cond, a, b, d, clean_label, conflict_label, aux_label, r1, r2),
                })
            graph_count += 1
        if graph_count >= cfg.n_graphs: break
    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "asa8_dataset.csv", index=False, encoding="utf-8-sig")
    return df


def load_model_and_tokenizer(cfg):
    cfg.model_path = resolve_model_path(cfg)
    print(f"[MODEL] model_key={cfg.model_key} resolved_path={cfg.model_path}")
    tok = AutoTokenizer.from_pretrained(cfg.model_path, local_files_only=True, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, local_files_only=True, trust_remote_code=True,
        torch_dtype=dtype_from_cfg(cfg), device_map="auto" if cfg.device == "cuda" else None,
    )
    if cfg.device == "cpu": model.to(cfg.device)
    model.eval()
    return model, tok


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"): return model.model.layers
    if hasattr(model, "layers"): return model.layers
    raise RuntimeError("Cannot locate transformer layers")


def get_lm_head_weight(model):
    if hasattr(model, "lm_head"): return model.lm_head.weight.detach()
    raise RuntimeError("Cannot locate lm_head")


def last_positions(attention_mask):
    return torch.full((attention_mask.shape[0],), attention_mask.shape[1] - 1, dtype=torch.long, device=attention_mask.device)


def extract_baseline(model, tokenizer, df, cfg, save_dir):
    print("[EXTRACT] baseline hidden states and targets")
    W = get_lm_head_weight(model).detach().float(); W_device = W.to(model.device)
    n = len(df)
    needed_layers = sorted(set(list(cfg.operator_layers) + [l+1 for l in cfg.operator_layers] + list(cfg.precursor_layers) + list(cfg.decision_layers)))
    h_by_layer = {l: None for l in needed_layers}
    R_by_layer = {l: np.zeros(n, dtype=np.float32) for l in cfg.decision_layers}
    final_clean_logits = np.zeros(n, dtype=np.float32); final_conflict_logits = np.zeros(n, dtype=np.float32)
    texts = df["text"].tolist(); clean_ids_all = df["clean_token_id"].values.astype(np.int64); conflict_ids_all = df["conflict_token_id"].values.astype(np.int64)
    with torch.no_grad():
        for start in range(0, n, cfg.batch_size):
            end = min(n, start + cfg.batch_size)
            inputs = tokenizer(texts[start:end], return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_len).to(model.device)
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            hstates = outputs.hidden_states; pos = last_positions(inputs["attention_mask"]); bsz = end - start
            idx_t = torch.arange(bsz, device=model.device)
            cids = torch.tensor(clean_ids_all[start:end], dtype=torch.long, device=model.device)
            eids = torch.tensor(conflict_ids_all[start:end], dtype=torch.long, device=model.device)
            for l in needed_layers:
                h = hstates[l + 1][idx_t, pos, :].detach().float().cpu().numpy().astype(np.float32)
                if h_by_layer[l] is None: h_by_layer[l] = np.zeros((n, h.shape[1]), dtype=np.float32)
                h_by_layer[l][start:end] = h
            for l in cfg.decision_layers:
                ht = hstates[l + 1][idx_t, pos, :].detach().float()
                clean_logits = torch.sum(ht * W_device[cids].float(), dim=1)
                conflict_logits = torch.sum(ht * W_device[eids].float(), dim=1)
                R_by_layer[l][start:end] = (clean_logits - conflict_logits).detach().cpu().numpy().astype(np.float32)
            final_logits = outputs.logits[idx_t, pos, :].detach().float()
            final_clean_logits[start:end] = final_logits[idx_t, cids].detach().cpu().numpy().astype(np.float32)
            final_conflict_logits[start:end] = final_logits[idx_t, eids].detach().cpu().numpy().astype(np.float32)
            print(f"  extracted {end}/{n}")
            del outputs, hstates, inputs
            gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    out = df.copy()
    for l in cfg.decision_layers: out[f"R_L{l}"] = R_by_layer[l]
    out["final_clean_logit"] = final_clean_logits; out["final_conflict_logit"] = final_conflict_logits
    out["pair_choice"] = np.where(final_clean_logits >= final_conflict_logits, "clean", "conflict")
    clean_anchor = out[out["condition"] == "clean"].set_index("graph_id")
    dR_cols = []
    for l in cfg.decision_layers:
        vals = [row[f"R_L{l}"] - clean_anchor.loc[row["graph_id"], f"R_L{l}"] for _, row in out.iterrows()]
        col = f"dR_L{l}"; out[col] = np.asarray(vals, dtype=np.float32); dR_cols.append(col)
    X_dec = out[dR_cols].values.astype(np.float32)
    pca = PCA(n_components=1); du = pca.fit_transform(X_dec)[:, 0]
    if np.mean(du[out["mechanism"] == "stable"]) < np.mean(du[out["mechanism"] == "closure"]):
        du = -du; pca.components_[0] *= -1
    out["DeltaU"] = du.astype(np.float32)
    out["D_B_proxy"] = np.min(np.abs(X_dec), axis=1).astype(np.float32)
    out["R_final_margin"] = final_clean_logits - final_conflict_logits
    out.to_csv(save_dir / "asa8_baseline_features.csv", index=False, encoding="utf-8-sig")
    print("[PCA] DeltaU explained variance:", float(pca.explained_variance_ratio_[0]))
    return out, h_by_layer, pca


def make_split(df, cfg, save_dir):
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
    tr, te = next(gss.split(df, groups=df["graph_id"].values))
    test_set = set(te.tolist())
    split_df = pd.DataFrame({"prompt_id": df["prompt_id"], "graph_id": df["graph_id"], "condition": df["condition"], "mechanism": df["mechanism"], "split": ["test" if i in test_set else "train" for i in range(len(df))]})
    split_df.to_csv(save_dir / "asa8_split.csv", index=False, encoding="utf-8-sig")
    return np.asarray(tr), np.asarray(te), split_df


class LowDimOperator:
    def __init__(self, pca, ridge, layer): self.pca = pca; self.ridge = ridge; self.layer = layer
    def predict_torch(self, h_in: torch.Tensor) -> torch.Tensor:
        device = h_in.device; dtype = h_in.dtype
        mean = torch.tensor(self.pca.mean_, device=device, dtype=dtype)
        comps = torch.tensor(self.pca.components_, device=device, dtype=dtype)
        coef = torch.tensor(self.ridge.coef_, device=device, dtype=dtype)
        intercept = torch.tensor(self.ridge.intercept_, device=device, dtype=dtype)
        z = (h_in - mean) @ comps.T
        z_next = z @ coef.T + intercept
        return z_next @ comps + mean


def fit_stable_operators(base_df, h_by_layer, train_idx, cfg, save_dir):
    operators, audits = {}, []
    for l in cfg.operator_layers:
        H0, H1 = h_by_layer[l], h_by_layer[l + 1]
        H_comb = np.concatenate([H0[train_idx], H1[train_idx]], axis=0)
        dim = min(cfg.pca_dim, H_comb.shape[0] - 1, H_comb.shape[1])
        pca = PCA(n_components=dim, random_state=cfg.seed); pca.fit(H_comb)
        z0_all, z1_all = pca.transform(H0), pca.transform(H1)
        operators[l] = {}
        for mech in ["stable", "closure"]:
            idx = np.asarray([i for i in train_idx if base_df.iloc[i]["mechanism"] == mech], dtype=int)
            reg = Ridge(alpha=cfg.operator_ridge_alpha); reg.fit(z0_all[idx], z1_all[idx])
            pred = reg.predict(z0_all[idx])
            audits.append({"layer": l, "mechanism": mech, "n_train": int(len(idx)), "pca_dim": int(dim), "pca_var_sum": float(np.sum(pca.explained_variance_ratio_)), "train_r2_znext": float(r2_score(z1_all[idx], pred)), "train_flat_corr_znext": safe_corr(z1_all[idx].ravel(), pred.ravel())})
            operators[l][mech] = LowDimOperator(pca, reg, l)
    pd.DataFrame(audits).to_csv(save_dir / "asa8_operator_audit.csv", index=False, encoding="utf-8-sig")
    return operators


def fit_ridge_vector(X_train, y_train, alpha, min_norm):
    X = X_train.astype(np.float32); y = y_train.astype(np.float32)
    x_mean = X.mean(axis=0, keepdims=True); y_mean = float(y.mean())
    X0 = X - x_mean; y0 = y - y_mean
    reg = Ridge(alpha=alpha); reg.fit(X0, y0)
    coef = reg.coef_.astype(np.float32); unit, norm = normalize_np(coef, eps=min_norm)
    pred = reg.predict(X0)
    return {"coef": coef, "unit": unit, "norm": norm, "x_mean": x_mean.reshape(-1).astype(np.float32), "y_mean": y_mean, "train_r2": float(r2_score(y0, pred)), "train_corr": safe_corr(y0, pred)}


def fit_precursor_and_probes(base_df, h_by_layer, train_idx, cfg, save_dir):
    y_du = base_df["DeltaU"].values.astype(np.float32); y_db = base_df["D_B_proxy"].values.astype(np.float32)
    dirs, probes, audits = {}, {}, []
    for l in sorted(set(cfg.operator_layers).union(set(cfg.precursor_layers))):
        H = h_by_layer[l].astype(np.float32)
        du_fit = fit_ridge_vector(H[train_idx], y_du[train_idx], cfg.probe_ridge_alpha, cfg.min_dir_norm)
        db_fit = fit_ridge_vector(H[train_idx], y_db[train_idx], cfg.probe_ridge_alpha, cfg.min_dir_norm)
        probes[(l, "DeltaU")] = du_fit; probes[(l, "D_B")] = db_fit
        if l in cfg.precursor_layers: dirs[l] = du_fit["unit"]
        audits.append({"layer": l, "target": "DeltaU", "coef_norm": du_fit["norm"], "valid": bool(du_fit["unit"] is not None), "train_r2": du_fit["train_r2"], "train_corr": du_fit["train_corr"]})
        audits.append({"layer": l, "target": "D_B", "coef_norm": db_fit["norm"], "valid": bool(db_fit["unit"] is not None), "train_r2": db_fit["train_r2"], "train_corr": db_fit["train_corr"]})
    vel_stats = {}
    for l in cfg.operator_layers:
        V = h_by_layer[l + 1] - h_by_layer[l]
        vn = np.linalg.norm(V[train_idx], axis=1)
        vel_stats[l] = {"mean": float(np.mean(vn)), "std": float(np.std(vn)+1e-8), "q_low": float(np.quantile(vn, cfg.inertia_low_q)), "q_mid": float(np.quantile(vn, 0.50))}
    pd.DataFrame(audits).to_csv(save_dir / "asa8_probe_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"layer": l, "direction": "position_ridge_DeltaU", "valid": bool(dirs.get(l) is not None), "norm": float(np.linalg.norm(dirs[l])) if dirs.get(l) is not None else np.nan} for l in cfg.precursor_layers]).to_csv(save_dir / "asa8_direction_audit.csv", index=False, encoding="utf-8-sig")
    return dirs, probes, vel_stats


def make_probe_tensors(probes, device):
    out = {}
    for key, fit in probes.items():
        out[key] = {"coef": torch.tensor(fit["coef"], device=device, dtype=torch.float32), "x_mean": torch.tensor(fit["x_mean"], device=device, dtype=torch.float32), "y_mean": torch.tensor([fit["y_mean"]], device=device, dtype=torch.float32)}
    return out


def estimate_probe(h, fit):
    return ((h.float() - fit["x_mean"][None, :]) @ fit["coef"][:, None] + fit["y_mean"][None, :]).float()


def make_hook_factory(cfg, stable_ops, precursor_dirs, probe_tensors, vel_stats, control_mode, W_device, trace_buffer, clean_ids_batch, conflict_ids_batch):
    pre_tensors = {int(l): torch.tensor(v, dtype=torch.float32, device=W_device.device) for l, v in precursor_dirs.items() if v is not None}
    def hook_factory(l):
        def hook(module, inputs, output):
            if control_mode == "none": return output
            if isinstance(output, tuple): h_out, rest = output[0], output[1:]
            else: h_out, rest = output, None
            B, S, _ = h_out.shape; pos = torch.full((B,), S-1, dtype=torch.long, device=h_out.device); idx = torch.arange(B, device=h_out.device)
            h_in = inputs[0]; h_in_last = h_in[idx, pos, :]; h_real_last = h_out[idx, pos, :]; h_new = h_real_last
            alpha = torch.full((B,1), cfg.fixed_alpha, device=h_out.device, dtype=h_out.dtype)
            beta = torch.full((B,1), cfg.fixed_beta, device=h_out.device, dtype=h_out.dtype)
            du_hat = estimate_probe(h_in_last, probe_tensors[(l,"DeltaU")]) if (l,"DeltaU") in probe_tensors else torch.zeros((B,1), device=h_out.device)
            db_hat = estimate_probe(h_in_last, probe_tensors[(l,"D_B")]) if (l,"D_B") in probe_tensors else torch.ones((B,1), device=h_out.device)
            vel_norm = torch.norm(h_real_last.float() - h_in_last.float(), dim=1, keepdim=True)
            vq = torch.tensor([[vel_stats.get(l, {}).get("q_low", 1.0)]], device=h_out.device, dtype=torch.float32)
            boundary_gate = torch.sigmoid((torch.tensor([[0.75]], device=h_out.device) - db_hat.float()) / max(cfg.db_temp, 1e-6))
            du_gate = torch.sigmoid((torch.tensor([[1.00]], device=h_out.device) - torch.abs(du_hat.float())) / max(cfg.du_temp, 1e-6))
            inertia_gate = torch.sigmoid((vq - vel_norm.float()) / max(cfg.inertia_temp, 1e-6))
            if control_mode == "operator_only_fixed": alpha, beta = alpha, torch.zeros_like(beta)
            elif control_mode == "precursor_only_fixed": alpha, beta = torch.zeros_like(alpha), beta
            elif control_mode == "same_combo_fixed": pass
            elif control_mode == "closed_loop_boundary":
                g = boundary_gate; alpha = (cfg.alpha_max*g).to(h_out.dtype); beta = (cfg.beta_max*g).to(h_out.dtype)
            elif control_mode == "closed_loop_inertia":
                g = inertia_gate; alpha = (cfg.alpha_max*g).to(h_out.dtype); beta = (cfg.beta_max*g).to(h_out.dtype)
            elif control_mode == "closed_loop_full":
                g = (0.50*boundary_gate + 0.30*du_gate + 0.20*inertia_gate).clamp(0,1); alpha = (cfg.alpha_max*g).to(h_out.dtype); beta = (cfg.beta_max*g).to(h_out.dtype)
            elif control_mode == "closed_loop_full_safe":
                g = (0.50*boundary_gate + 0.30*du_gate + 0.20*inertia_gate).clamp(0,1); cap = min(cfg.safe_gain_cap, cfg.alpha_max); alpha = (cap*g).to(h_out.dtype); beta = (cap*g).to(h_out.dtype)
            elif control_mode == "random_gain_combo":
                g = torch.full_like(boundary_gate, ((l*37)%100)/100.0); alpha = (cfg.alpha_max*g).to(h_out.dtype); beta = (cfg.beta_max*g).to(h_out.dtype)
            elif control_mode == "answer": alpha, beta = torch.zeros_like(alpha), beta
            def apply_operator(h_base):
                if l not in stable_ops or torch.max(alpha).item() == 0.0: return h_base
                h_target = stable_ops[l]["stable"].predict_torch(h_in_last.float()).to(h_base.dtype)
                return (1.0-alpha)*h_base + alpha*h_target
            def apply_precursor(h_base):
                if l not in pre_tensors or torch.max(beta).item() == 0.0: return h_base
                d = pre_tensors[l].to(h_base.device, h_base.dtype); scale = h_base.float().std(dim=-1, keepdim=True).to(h_base.dtype)
                return h_base + beta*scale*d[None,:].expand(B,-1)
            def apply_answer(h_base):
                ans = W_device[clean_ids_batch].float() - W_device[conflict_ids_batch].float(); ans = ans/(torch.norm(ans, dim=1, keepdim=True)+1e-8)
                scale = h_base.float().std(dim=-1, keepdim=True).to(h_base.dtype); return h_base + beta*scale*ans.to(h_base.dtype)
            h_new = apply_answer(h_new) if control_mode == "answer" else apply_precursor(apply_operator(h_new))
            trace_buffer.append({"control": control_mode, "layer": int(l), "mean_alpha": float(alpha.detach().float().mean().cpu()), "mean_beta": float(beta.detach().float().mean().cpu()), "mean_du_hat": float(du_hat.detach().float().mean().cpu()), "mean_db_hat": float(db_hat.detach().float().mean().cpu()), "mean_vel_norm": float(vel_norm.detach().float().mean().cpu()), "mean_boundary_gate": float(boundary_gate.detach().float().mean().cpu()), "mean_du_gate": float(du_gate.detach().float().mean().cpu()), "mean_inertia_gate": float(inertia_gate.detach().float().mean().cpu()), "batch_size": int(B)})
            if torch.allclose(h_new, h_real_last): return output
            h2 = h_out.clone(); h2[idx, pos, :] = h_new
            return (h2,) + rest if rest is not None else h2
        return hook
    return hook_factory


def forward_eval(model, tokenizer, df_subset, cfg, pca_decision, clean_anchor_R, stable_ops, precursor_dirs, probes, vel_stats, control_mode):
    W = get_lm_head_weight(model).detach().float(); W_device = W.to(model.device)
    probe_tensors = make_probe_tensors(probes, W_device.device); rows, trace_buffer = [], []
    texts = df_subset["text"].tolist(); clean_ids_all = df_subset["clean_token_id"].values.astype(np.int64); conflict_ids_all = df_subset["conflict_token_id"].values.astype(np.int64)
    layer_modules = get_layers(model); hook_layers = sorted(set(list(cfg.operator_layers)+list(cfg.precursor_layers)))
    with torch.no_grad():
        for start in range(0, len(df_subset), cfg.batch_size):
            end = min(len(df_subset), start+cfg.batch_size); batch = df_subset.iloc[start:end].reset_index(drop=True)
            inputs = tokenizer(texts[start:end], return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_len).to(model.device)
            cids = torch.tensor(clean_ids_all[start:end], dtype=torch.long, device=model.device); eids = torch.tensor(conflict_ids_all[start:end], dtype=torch.long, device=model.device)
            handles = []
            if control_mode != "none":
                hf = make_hook_factory(cfg, stable_ops, precursor_dirs, probe_tensors, vel_stats, control_mode, W_device, trace_buffer, cids, eids)
                for l in hook_layers:
                    if l < len(layer_modules): handles.append(layer_modules[l].register_forward_hook(hf(l)))
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            for h in handles: h.remove()
            hstates = outputs.hidden_states; pos = last_positions(inputs["attention_mask"]); B = end-start; idx_t = torch.arange(B, device=model.device)
            dR_mat = []
            for l in cfg.decision_layers:
                h = hstates[l+1][idx_t, pos, :].detach().float(); clean_logits = torch.sum(h * W_device[cids].float(), dim=1); conflict_logits = torch.sum(h * W_device[eids].float(), dim=1)
                R = (clean_logits - conflict_logits).detach().cpu().numpy().astype(np.float32)
                dR_mat.append([R[bi] - clean_anchor_R[int(row["graph_id"])][l] for bi, (_, row) in enumerate(batch.iterrows())])
            dR_mat = np.asarray(dR_mat, dtype=np.float32).T; du = pca_decision.transform(dR_mat)[:,0].astype(np.float32)
            final_logits = outputs.logits[idx_t, pos, :].detach().float(); final_clean = final_logits[idx_t, cids].detach().cpu().numpy(); final_conflict = final_logits[idx_t, eids].detach().cpu().numpy(); pair_choice = np.where(final_clean >= final_conflict, "clean", "conflict")
            for bi, (_, row) in enumerate(batch.iterrows()):
                rows.append({"prompt_id": row["prompt_id"], "graph_id": int(row["graph_id"]), "condition": row["condition"], "mechanism": row["mechanism"], "clean_label": row["clean_label"], "conflict_label": row["conflict_label"], "DeltaU": float(du[bi]), "final_clean_logit": float(final_clean[bi]), "final_conflict_logit": float(final_conflict[bi]), "pair_choice": str(pair_choice[bi])})
            del outputs, hstates, inputs
            gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return pd.DataFrame(rows), pd.DataFrame(trace_buffer)


def summarize_specificity(results_df, base_test):
    merged = results_df.merge(base_test[["prompt_id", "DeltaU", "pair_choice"]], on="prompt_id", suffixes=("", "_base"))
    merged["shift"] = merged["DeltaU"] - merged["DeltaU_base"]; merged["abs_shift"] = merged["shift"].abs(); merged["target_hit"] = (merged["pair_choice"] == "clean").astype(int)
    rows = []
    for control, g in merged.groupby("control"):
        target = g[g["mechanism"] == "closure"]; nontarget = g[g["mechanism"] != "closure"]
        rows.append({"control": control, "target_abs_shift": float(target["abs_shift"].mean()) if len(target) else np.nan, "nontarget_abs_shift": float(nontarget["abs_shift"].mean()) if len(nontarget) else np.nan, "specificity": float(target["abs_shift"].mean()-nontarget["abs_shift"].mean()) if len(target) and len(nontarget) else np.nan, "target_signed_shift": float(target["shift"].mean()) if len(target) else np.nan, "nontarget_signed_shift": float(nontarget["shift"].mean()) if len(nontarget) else np.nan, "target_hit_rate": float(target["target_hit"].mean()) if len(target) else np.nan, "nontarget_hit_rate": float(nontarget["target_hit"].mean()) if len(nontarget) else np.nan, "n": int(len(g))})
    return pd.DataFrame(rows), merged


def summarize_gain_trace(trace_df):
    if len(trace_df) == 0: return pd.DataFrame()
    rows = []
    for (control, layer), g in trace_df.groupby(["control", "layer"]):
        w = g["batch_size"]
        rows.append({"control": control, "layer": int(layer), "mean_alpha": float(np.average(g["mean_alpha"], weights=w)), "mean_beta": float(np.average(g["mean_beta"], weights=w)), "mean_du_hat": float(np.average(g["mean_du_hat"], weights=w)), "mean_db_hat": float(np.average(g["mean_db_hat"], weights=w)), "mean_vel_norm": float(np.average(g["mean_vel_norm"], weights=w)), "mean_boundary_gate": float(np.average(g["mean_boundary_gate"], weights=w)), "mean_du_gate": float(np.average(g["mean_du_gate"], weights=w)), "mean_inertia_gate": float(np.average(g["mean_inertia_gate"], weights=w)), "n_batches": int(len(g)), "n_samples_weighted": int(g["batch_size"].sum())})
    return pd.DataFrame(rows)


def main(cfg):
    set_seed(cfg.seed)
    save_dir = Path(cfg.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "asa8_config.json", "w", encoding="utf-8") as f: json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    model, tokenizer = load_model_and_tokenizer(cfg)
    df = build_dataset(tokenizer, cfg, save_dir)
    base_df, h_by_layer, pca_decision = extract_baseline(model, tokenizer, df, cfg, save_dir)
    train_idx, test_idx, _ = make_split(base_df, cfg, save_dir)
    clean_anchor_R = {}
    for g, row in base_df[base_df["condition"] == "clean"].set_index("graph_id").iterrows(): clean_anchor_R[int(g)] = {l: float(row[f"R_L{l}"]) for l in cfg.decision_layers}
    stable_ops = fit_stable_operators(base_df, h_by_layer, train_idx, cfg, save_dir)
    precursor_dirs, probes, vel_stats = fit_precursor_and_probes(base_df, h_by_layer, train_idx, cfg, save_dir)
    test_df = base_df.iloc[test_idx].reset_index(drop=True); base_test = base_df.iloc[test_idx][["prompt_id", "DeltaU", "pair_choice"]].reset_index(drop=True)
    controls = ["none", "operator_only_fixed", "precursor_only_fixed", "same_combo_fixed", "closed_loop_boundary", "closed_loop_inertia", "closed_loop_full", "closed_loop_full_safe", "random_gain_combo"]
    if cfg.include_answer_control: controls.append("answer")
    results, traces = [], []
    for control in controls:
        print(f"[RUN] control={control}")
        out, tr = forward_eval(model, tokenizer, test_df, cfg, pca_decision, clean_anchor_R, stable_ops, precursor_dirs, probes, vel_stats, control)
        out["control"] = control; results.append(out)
        if len(tr): traces.append(tr)
    results_df = pd.concat(results, ignore_index=True); results_df.to_csv(save_dir / "asa8_steering_results.csv", index=False, encoding="utf-8-sig")
    spec_df, merged = summarize_specificity(results_df, base_test); spec_df.to_csv(save_dir / "asa8_specificity_summary.csv", index=False, encoding="utf-8-sig"); merged.to_csv(save_dir / "asa8_merged_shift_table.csv", index=False, encoding="utf-8-sig")
    trace_df = pd.concat(traces, ignore_index=True) if len(traces) else pd.DataFrame(); trace_df.to_csv(save_dir / "asa8_control_trace.csv", index=False, encoding="utf-8-sig")
    gain_df = summarize_gain_trace(trace_df); gain_df.to_csv(save_dir / "asa8_gain_profile.csv", index=False, encoding="utf-8-sig")
    best = spec_df.sort_values("specificity", ascending=False).head(1).iloc[0].to_dict()
    fixed = spec_df[spec_df["control"] == "same_combo_fixed"]; fixed_spec = float(fixed.iloc[0]["specificity"]) if len(fixed) else np.nan
    closed = spec_df[spec_df["control"].str.startswith("closed_loop")]
    best_closed = closed.sort_values("specificity", ascending=False).head(1).iloc[0].to_dict() if len(closed) else {}
    gain_over_fixed = best_closed.get("specificity", np.nan) - fixed_spec if len(best_closed) else np.nan
    if len(best_closed) and gain_over_fixed > 0.10: verdict = "PASS_CLOSED_LOOP_IMPROVES_FIXED"
    elif len(best_closed) and gain_over_fixed > 0.03: verdict = "PASS_LITE_CLOSED_LOOP_HINT"
    elif str(best.get("control")) == "same_combo_fixed": verdict = "FIXED_COMBO_REMAINS_BEST"
    else: verdict = "MIXED_OR_UNRESOLVED"
    verdict_df = pd.DataFrame([{"audit": "ASA-8 Closed-Loop / Inertia-Aware Control Audit", "verdict": verdict, "best_control": best.get("control"), "best_specificity": best.get("specificity"), "fixed_combo_specificity": fixed_spec, "best_closed_loop_control": best_closed.get("control", None), "best_closed_loop_specificity": best_closed.get("specificity", np.nan), "closed_loop_gain_over_fixed": gain_over_fixed, "best_target_hit_rate": best.get("target_hit_rate"), "best_nontarget_hit_rate": best.get("nontarget_hit_rate")}])
    verdict_df.to_csv(save_dir / "asa8_verdict.csv", index=False, encoding="utf-8-sig")
    summary = {"audit": "ASA-8 Closed-Loop / Inertia-Aware Control Audit", "config": asdict(cfg), "n_test_prompts": int(len(test_df)), "deltaU_pca_explained_variance": float(pca_decision.explained_variance_ratio_[0]), "num_runs": int(len(controls)), "verdict": verdict, "best_specificity_row": best, "best_closed_loop_row": best_closed, "fixed_combo_specificity": fixed_spec, "closed_loop_gain_over_fixed": gain_over_fixed, "key_outputs": {"steering_results": "asa8_steering_results.csv", "specificity": "asa8_specificity_summary.csv", "control_trace": "asa8_control_trace.csv", "gain_profile": "asa8_gain_profile.csv", "verdict": "asa8_verdict.csv", "operator_audit": "asa8_operator_audit.csv", "probe_audit": "asa8_probe_audit.csv", "direction_audit": "asa8_direction_audit.csv"}, "note": "ASA-8 tests adaptive gain control after ASA-7B validated fixed operator+order-precursor combo. Closed-loop gates use layerwise probes for DeltaU, D_B, and velocity/inertia proxy."}
    with open(save_dir / "asa8_summary.json", "w", encoding="utf-8") as f: json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nASA-8 complete."); print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main(CFG)
