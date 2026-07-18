# -*- coding: utf-8 -*-
"""Shared prompt construction and functional-window utilities."""

import os

MODEL_PATHS = {
    "qwen": os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen2.5-1.5B-Instruct"),
    "llama": os.environ.get("LLAMA_MODEL_PATH", "meta-llama/Llama-3.2-1B-Instruct"),
    "gemma": os.environ.get("GEMMA_MODEL_PATH", "google/gemma-2-2b-it"),
}

"""
Functional-window shared audit: True train-CV window selection + held-out test evaluation
=====================================================================

Fix from v0.1
-------------
v0.1 could still select windows by held-out score. v0.2 separates:
    1. TRAIN-CV window selection on train graph IDs only.
    2. FINAL held-out test evaluation on test graph IDs only.

Outputs
-------
functional_window_outputs/
    window_true_heldout_summary.csv
    <model>_traincv_window_scan.csv
    <model>_selected_windows_test_eval.csv
    <model>_prompt_table.csv
    window_config.json
"""

import argparse, json, math, os, random, gc, warnings
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, accuracy_score, r2_score, roc_auc_score

warnings.filterwarnings("ignore")

LABEL_CANDIDATES = [
    "Red","Blue","Green","Yellow","North","South","East","West",
    "Copper","Silver","Gold","Iron","Circle","Square","Triangle","Star",
    "River","Mountain","Forest","Ocean","Sun","Moon","Cloud","Stone",
    "Alpha","Beta","Gamma","Delta","Apple","Orange","Lemon","Pear",
]
ENTITIES = [
    ("Ava","Bela","Cora"),("Darin","Elo","Faye"),("Galen","Hera","Ivo"),
    ("Juno","Kira","Lio"),("Mira","Nero","Orin"),("Pia","Quin","Rhea"),
    ("Sola","Taro","Una"),("Vera","Wen","Xio"),("Yara","Zeno","Nia"),
    ("Orla","Pavel","Rin"),("Nora","Silas","Tess"),("Uma","Vito","Willa"),
]
REL_WORDS = [
    ("belongs to","is located at"),
    ("is assigned to","maps to"),
    ("is part of","points to"),
    ("is grouped under","has label"),
]
REL_PRESERVE_CONDS = ["rename","permuted","redundant","irrelevant","paraphrase"]
STRUCTURE_CHANGE_CONDS = [
    "weak_distractor","competition_balanced","direct_conflict",
    "closure_update","closure_override","exception_override",
]
ALL_CONDS = ["clean"] + REL_PRESERVE_CONDS + STRUCTURE_CHANGE_CONDS
MECHANISM = {
    "clean":"stable","rename":"stable","permuted":"stable","redundant":"stable",
    "irrelevant":"stable","paraphrase":"stable","weak_distractor":"stable_shift",
    "competition_balanced":"competition","direct_conflict":"competition",
    "closure_update":"closure","closure_override":"closure","exception_override":"closure",
}
COND_CLASS = {
    "clean":"clean",
    **{c:"relation_preserving" for c in REL_PRESERVE_CONDS},
    **{c:"structure_changing" for c in STRUCTURE_CHANGE_CONDS},
}
FEATURE_COLS = [
    "rel_center_dist","rel_jaccard_dist","rel_weighted_jaccard_dist",
    "rel_composite_dist","spread","entropy",
]

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def continuation_ids(tok, text):
    return tok(" " + text, add_special_tokens=False)["input_ids"]

def make_prompt(cond, a, b, d, clean_label, conflict_label, aux_label, relation1, relation2):
    if cond == "clean":
        lines = ["You are given a small relation graph.",
                 f"Fact 1: {a} {relation1} {b}.",
                 f"Fact 2: {b} {relation2} {clean_label}.",
                 f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
                 "Answer with exactly one word:"]
    elif cond == "rename":
        lines = ["Consider a tiny mapping network.",
                 f"Link one: {a} {relation1} {b}.",
                 f"Link two: {b} {relation2} {clean_label}.",
                 f"Choose the label reached from {a}: {clean_label} or {conflict_label}.",
                 "One word answer:"]
    elif cond == "permuted":
        lines = ["You are given a small relation graph.",
                 f"Fact 2: {b} {relation2} {clean_label}.",
                 f"Fact 1: {a} {relation1} {b}.",
                 f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
                 "Answer with exactly one word:"]
    elif cond == "redundant":
        lines = ["You are given a small relation graph.",
                 f"Fact 1: {a} {relation1} {b}.",
                 f"Fact 2: {b} {relation2} {clean_label}.",
                 f"Repeated confirmation: {a} still goes through {b}.",
                 f"Repeated confirmation: {b} still points to {clean_label}.",
                 f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
                 "Answer with exactly one word:"]
    elif cond == "irrelevant":
        lines = ["You are given a small relation graph.",
                 f"Fact 1: {a} {relation1} {b}.",
                 f"Fact 2: {b} {relation2} {clean_label}.",
                 f"Irrelevant fact: {d} is associated with {aux_label}.",
                 f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
                 "Answer with exactly one word:"]
    elif cond == "paraphrase":
        lines = [f"In this example, {a} reaches {b}.",
                 f"The destination connected to {b} is {clean_label}.",
                 f"Based on those two links, select the label for {a}: {clean_label} or {conflict_label}.",
                 "Answer:"]
    elif cond == "weak_distractor":
        lines = ["You are given a small relation graph.",
                 f"Fact 1: {a} {relation1} {b}.",
                 f"Fact 2: {b} {relation2} {clean_label}.",
                 f"Weak note: some unrelated source mentions {conflict_label}, but does not update the graph.",
                 f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
                 "Answer with exactly one word:"]
    elif cond == "competition_balanced":
        lines = ["You are given a small relation graph.",
                 f"Fact 1: {a} {relation1} {b}.",
                 f"Fact 2: {b} {relation2} {clean_label}.",
                 f"Competing fact: {a} is also associated with {conflict_label}.",
                 f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
                 "Answer with exactly one word:"]
    elif cond == "direct_conflict":
        lines = ["You are given a small relation graph.",
                 f"Fact 1: {a} {relation1} {b}.",
                 f"Fact 2: {b} {relation2} {clean_label}.",
                 f"Direct conflicting fact: {a} {relation2} {conflict_label}.",
                 f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
                 "Answer with exactly one word:"]
    elif cond == "closure_update":
        lines = ["You are given a small relation graph.",
                 f"Fact 1: {a} {relation1} {b}.",
                 f"Old record: {b} {relation2} {clean_label}.",
                 f"Updated record: {b} {relation2} {conflict_label}, not {clean_label}.",
                 f"Question: Which label is associated with {a}: {clean_label} or {conflict_label}?",
                 "Answer with exactly one word:"]
    elif cond == "closure_override":
        lines = ["You are given a rule system.",
                 f"General rule: items that {relation1} {b} receive label {clean_label}.",
                 f"Override rule: in this case, items that {relation1} {b} receive label {conflict_label}.",
                 f"Fact: {a} {relation1} {b}.",
                 f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                 "Answer with exactly one word:"]
    elif cond == "exception_override":
        lines = ["You are given a rule system.",
                 f"General rule: items connected to {b} use label {clean_label}.",
                 f"Exception: {a} is a special case and uses label {conflict_label}.",
                 f"Fact: {a} is connected to {b}.",
                 f"Question: Which label applies to {a}: {clean_label} or {conflict_label}?",
                 "Answer with exactly one word:"]
    else:
        raise ValueError(cond)
    return "\n".join(lines)

def build_dataset(common_labels, n_graphs):
    rows, graph_count, n_labels = [], 0, len(common_labels)
    for ent_i, (a,b,d) in enumerate(ENTITIES):
        for rel_i, (r1,r2) in enumerate(REL_WORDS):
            if graph_count >= n_graphs: break
            clean = common_labels[(2*graph_count) % n_labels]
            conflict = common_labels[(2*graph_count+1) % n_labels]
            aux = common_labels[(2*graph_count+2) % n_labels]
            for cond in ALL_CONDS:
                rows.append({
                    "prompt_id": f"g{graph_count:03d}_{cond}",
                    "graph_id": graph_count,
                    "entity_id": ent_i,
                    "relation_id": rel_i,
                    "condition": cond,
                    "condition_class": COND_CLASS[cond],
                    "mechanism": MECHANISM[cond],
                    "clean_label": clean,
                    "conflict_label": conflict,
                    "aux_label": aux,
                    "text": make_prompt(cond, a, b, d, clean, conflict, aux, r1, r2),
                })
            graph_count += 1
        if graph_count >= n_graphs: break
    return pd.DataFrame(rows)

def load_tok_model(path, device, dtype):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=True,
        torch_dtype=dtype, device_map="auto" if device=="cuda" else None
    )
    if device != "cuda": model.to(device)
    model.eval()
    return tok, model

def get_layers(model):
    return len(model.model.layers) if hasattr(model, "model") and hasattr(model.model, "layers") else len(model.layers)

def decision_layers(num_layers, frac=(0.70,0.92)):
    a,b = frac
    l0 = max(0, min(num_layers-1, int(math.floor(num_layers*a))))
    l1 = max(l0, min(num_layers-1, int(math.floor(num_layers*b))))
    return list(range(l0, l1+1))

def scan_windows(num_layers, max_init_frac, lengths):
    max_layer = max(1, int(math.floor((num_layers-1)*max_init_frac)))
    seen, out = set(), []
    for length in lengths:
        if length > max_layer+1: continue
        for start in range(0, max_layer-length+2):
            layers = list(range(start,start+length))
            key = tuple(layers)
            if key not in seen:
                seen.add(key)
                out.append({
                    "window": f"L{layers[0]}_{layers[-1]}",
                    "start": layers[0],
                    "end": layers[-1],
                    "length": len(layers),
                    "start_frac": layers[0]/max(1,num_layers-1),
                    "end_frac": layers[-1]/max(1,num_layers-1),
                    "layers": layers,
                })
    return out

def cosine_matrix(X):
    X = X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-8)
    return X @ X.T

def jaccard(a,b):
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb)/max(1,len(sa | sb))

def weighted_jaccard(ids_a, vals_a, ids_b, vals_b):
    va = vals_a - np.min(vals_a); vb = vals_b - np.min(vals_b)
    va = va/(np.sum(va)+1e-8); vb = vb/(np.sum(vb)+1e-8)
    da = {int(k):float(v) for k,v in zip(ids_a,va)}
    db = {int(k):float(v) for k,v in zip(ids_b,vb)}
    keys = set(da.keys()) | set(db.keys())
    return sum(min(da.get(k,0.0), db.get(k,0.0)) for k in keys)/(sum(max(da.get(k,0.0), db.get(k,0.0)) for k in keys)+1e-8)

def safe_corr(x,y):
    if np.std(x)<1e-8 or np.std(y)<1e-8: return 0.0
    return float(np.corrcoef(x,y)[0,1])

def cv_ridge_score(X, y, groups):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 3: return np.nan, np.nan
    pred = np.zeros_like(y, dtype=float)
    gkf = GroupKFold(n_splits=min(5, len(unique)))
    for tr, te in gkf.split(X, y, groups):
        reg = Pipeline([("scaler",StandardScaler()),("ridge",Ridge(alpha=10.0))])
        reg.fit(X[tr], y[tr])
        pred[te] = reg.predict(X[te])
    return float(r2_score(y, pred)), safe_corr(y, pred)

def cv_mech_score(X, y, groups):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 3 or len(np.unique(y)) < 2: return np.nan, np.nan
    pred = np.zeros_like(y, dtype=int)
    gkf = GroupKFold(n_splits=min(5, len(unique)))
    for tr, te in gkf.split(X, y, groups):
        clf = Pipeline([("scaler",StandardScaler()),("lr",LogisticRegression(max_iter=1000, class_weight="balanced"))])
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return float(accuracy_score(y, pred)), float(f1_score(y, pred, average="macro", zero_division=0))

def test_ridge_score(X, y, train_mask, test_mask):
    reg = Pipeline([("scaler",StandardScaler()),("ridge",Ridge(alpha=10.0))])
    reg.fit(X[train_mask], y[train_mask])
    pred = reg.predict(X[test_mask])
    return float(r2_score(y[test_mask], pred)), safe_corr(y[test_mask], pred)

def test_mech_score(X, y, train_mask, test_mask):
    if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[test_mask])) < 2:
        return np.nan, np.nan
    clf = Pipeline([("scaler",StandardScaler()),("lr",LogisticRegression(max_iter=1000, class_weight="balanced"))])
    clf.fit(X[train_mask], y[train_mask])
    pred = clf.predict(X[test_mask])
    return float(accuracy_score(y[test_mask], pred)), float(f1_score(y[test_mask], pred, average="macro", zero_division=0))

def extract_features_for_model(model_key, path, df, args, common_labels):
    device = args.device
    dtype = torch.float16 if args.dtype=="float16" else torch.float32
    tok, model = load_tok_model(path, device, dtype)
    L = get_layers(model)
    dec_layers = decision_layers(L, tuple(args.decision_frac))
    windows = scan_windows(L, args.max_init_frac, args.window_lengths)
    W = model.lm_head.weight.detach().float().to(model.device)
    W_norm = F.normalize(W.float(), dim=1)

    clean_ids, conflict_ids = [], []
    for _, row in df.iterrows():
        cids = continuation_ids(tok, row["clean_label"])
        eids = continuation_ids(tok, row["conflict_label"])
        if len(cids)!=1 or len(eids)!=1:
            raise RuntimeError(f"{model_key}: label tokenization not single-token.")
        clean_ids.append(cids[0]); conflict_ids.append(eids[0])

    n = len(df); texts = df["text"].tolist(); max_k = max(args.k_list)
    max_init_layer = max(max(w["layers"]) for w in windows)
    needed = sorted(set(list(range(0,max_init_layer+1)) + dec_layers))
    layer_data = {k: {"centers":{}, "ids":{}, "vals":{}, "spread":{}, "entropy":{}} for k in args.k_list}
    R = {l: np.zeros(n, dtype=np.float32) for l in dec_layers}

    with torch.no_grad():
        for start in range(0,n,args.batch_size):
            end = min(n, start+args.batch_size)
            inputs = tok(texts[start:end], return_tensors="pt", padding=True, truncation=True, max_length=args.max_len).to(model.device)
            outs = model(**inputs, output_hidden_states=True, use_cache=False)
            hstates = outs.hidden_states
            pos = torch.full((end-start,), inputs["attention_mask"].shape[1]-1, dtype=torch.long, device=model.device)
            cids_t = torch.tensor(clean_ids[start:end], dtype=torch.long, device=model.device)
            eids_t = torch.tensor(conflict_ids[start:end], dtype=torch.long, device=model.device)
            for l in needed:
                h = hstates[l+1][torch.arange(end-start,device=model.device),pos,:].detach().float()
                if l in R:
                    R[l][start:end] = (torch.sum(h*W[cids_t].float(),dim=1)-torch.sum(h*W[eids_t].float(),dim=1)).cpu().numpy()
                if l <= max_init_layer:
                    logits = h @ W.float().T
                    vals, ids = torch.topk(logits, k=max_k, dim=1)
                    for k in args.k_list:
                        ids_k = ids[:,:k]; vals_k = vals[:,:k].float()
                        emb = W_norm[ids_k].float()
                        center = F.normalize(emb.mean(dim=1), dim=1)
                        spread = (1.0 - torch.sum(emb*center[:,None,:], dim=2)).mean(dim=1)
                        p = torch.softmax(vals_k, dim=1)
                        entropy = -(p*torch.log(p+1e-8)).sum(dim=1)/math.log(k)
                        if l not in layer_data[k]["centers"]:
                            d = center.shape[1]
                            layer_data[k]["centers"][l] = np.zeros((n,d), dtype=np.float32)
                            layer_data[k]["ids"][l] = np.zeros((n,k), dtype=np.int32)
                            layer_data[k]["vals"][l] = np.zeros((n,k), dtype=np.float32)
                            layer_data[k]["spread"][l] = np.zeros(n, dtype=np.float32)
                            layer_data[k]["entropy"][l] = np.zeros(n, dtype=np.float32)
                        layer_data[k]["centers"][l][start:end] = center.cpu().numpy()
                        layer_data[k]["ids"][l][start:end] = ids_k.cpu().numpy().astype(np.int32)
                        layer_data[k]["vals"][l][start:end] = vals_k.cpu().numpy()
                        layer_data[k]["spread"][l][start:end] = spread.cpu().numpy()
                        layer_data[k]["entropy"][l][start:end] = entropy.cpu().numpy()
            print(f"[{model_key}] processed {end}/{n}")
            del outs, hstates, inputs
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    model_df = df.copy()
    for l in dec_layers: model_df[f"R_L{l}"] = R[l]
    clean_df = model_df[model_df["condition"]=="clean"].set_index("graph_id")
    dR_cols = []
    for l in dec_layers:
        col = f"dR_L{l}"
        model_df[col] = [row[f"R_L{l}"] - clean_df.loc[row["graph_id"], f"R_L{l}"] for _, row in model_df.iterrows()]
        dR_cols.append(col)
    X_dec = model_df[dR_cols].values.astype(np.float32)
    pc = PCA(n_components=min(3, X_dec.shape[1])).fit_transform(X_dec)
    du = pc[:,0]
    if np.mean(du[model_df["mechanism"]=="stable"]) < np.mean(du[model_df["mechanism"]=="closure"]): du = -du
    model_df["DeltaU"] = du.astype(np.float32)
    del model, tok
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return model_df, layer_data, L, dec_layers, windows

def compute_feature_df(model_df, layer_data, k, layers):
    centers = np.mean([layer_data[k]["centers"][l] for l in layers], axis=0)
    dist = 1.0 - cosine_matrix(centers)
    spread = np.mean([layer_data[k]["spread"][l] for l in layers], axis=0)
    entropy = np.mean([layer_data[k]["entropy"][l] for l in layers], axis=0)
    clean_idx = {int(row.graph_id): idx for idx,row in model_df.reset_index(drop=True).iterrows() if row.condition=="clean"}
    rows, preserve_d, structure_d = [], [], []
    for i,row in model_df.reset_index(drop=True).iterrows():
        ci = clean_idx[int(row.graph_id)]
        cd = float(dist[i,ci])
        if i != ci:
            js, wjs = [], []
            for l in layers:
                js.append(jaccard(layer_data[k]["ids"][l][i], layer_data[k]["ids"][l][ci]))
                wjs.append(weighted_jaccard(layer_data[k]["ids"][l][i], layer_data[k]["vals"][l][i],
                                            layer_data[k]["ids"][l][ci], layer_data[k]["vals"][l][ci]))
            jd, wjd = 1.0-float(np.mean(js)), 1.0-float(np.mean(wjs))
        else:
            jd, wjd = 0.0, 0.0
        comp = 0.50*cd + 0.25*jd + 0.25*wjd
        if row.condition in REL_PRESERVE_CONDS: preserve_d.append(comp)
        elif row.condition in STRUCTURE_CHANGE_CONDS: structure_d.append(comp)
        rows.append({
            "graph_id": int(row.graph_id), "condition": row.condition,
            "mechanism": row.mechanism, "rel_center_dist": cd,
            "rel_jaccard_dist": jd, "rel_weighted_jaccard_dist": wjd,
            "rel_composite_dist": comp, "spread": float(spread[i]),
            "entropy": float(entropy[i]), "DeltaU": float(row.DeltaU),
        })
    feat_df = pd.DataFrame(rows)
    geom = {
        "preserve_distance_mean": float(np.mean(preserve_d)),
        "structure_change_distance_mean": float(np.mean(structure_d)),
        "invariance_score_1_minus_preserve": float(1.0 - np.mean(preserve_d)),
        "structure_lift_over_preserve": float((np.mean(structure_d)-np.mean(preserve_d))/(np.mean(preserve_d)+1e-8)),
    }
    return feat_df, geom

def score_one(feat_df, train_mask, test_mask):
    X = feat_df[FEATURE_COLS].values.astype(np.float32)
    y_du = feat_df["DeltaU"].values.astype(float)
    y_mech = feat_df["mechanism"].map({"stable":0,"stable_shift":0,"competition":1,"closure":2}).values.astype(int)
    r2_test, corr_test = test_ridge_score(X, y_du, train_mask, test_mask)
    acc_test, f1_test = test_mech_score(X, y_mech, train_mask, test_mask)
    return r2_test, corr_test, acc_test, f1_test

def traincv_score_window(feat_df, train_mask):
    train_idx = np.where(train_mask)[0]
    X = feat_df[FEATURE_COLS].values.astype(np.float32)[train_idx]
    y_du = feat_df["DeltaU"].values.astype(float)[train_idx]
    y_mech = feat_df["mechanism"].map({"stable":0,"stable_shift":0,"competition":1,"closure":2}).values.astype(int)[train_idx]
    groups = feat_df["graph_id"].values.astype(int)[train_idx]
    du_r2, du_corr = cv_ridge_score(X, y_du, groups)
    mech_acc, mech_f1 = cv_mech_score(X, y_mech, groups)
    return du_r2, du_corr, mech_acc, mech_f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen","llama","gemma"], choices=list(MODEL_PATHS))
    ap.add_argument("--out_dir", default="functional_window_outputs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16", choices=["float16","float32"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--n_graphs", type=int, default=36)
    ap.add_argument("--k_list", type=int, nargs="+", default=[100,500])
    ap.add_argument("--max_init_frac", type=float, default=0.45)
    ap.add_argument("--window_lengths", type=int, nargs="+", default=[2,3,4,5,6,8])
    ap.add_argument("--decision_frac", type=float, nargs=2, default=[0.70,0.92])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    common = []
    token_audit = []
    for lab in LABEL_CANDIDATES:
        ok = True; row = {"label": lab}
        for mk in args.models:
            tok = AutoTokenizer.from_pretrained(MODEL_PATHS[mk], local_files_only=True, trust_remote_code=True)
            ids = continuation_ids(tok, lab)
            row[f"{mk}_ids"] = str(ids); row[f"{mk}_len"] = len(ids)
            if len(ids) != 1: ok = False
        row["is_common_single_token"] = ok
        token_audit.append(row)
        if ok: common.append(lab)
    pd.DataFrame(token_audit).to_csv(out/"window_common_label_tokenization_audit.csv", index=False, encoding="utf-8-sig")
    if len(common) < 12: raise RuntimeError(f"Need >=12 common labels, got {common}")
    common = common[:12]
    df = build_dataset(common, args.n_graphs)
    df.to_csv(out/"window_prompt_dataset.csv", index=False, encoding="utf-8-sig")

    rng = np.random.default_rng(args.seed)
    graph_ids = np.array(sorted(df.graph_id.unique()))
    rng.shuffle(graph_ids)
    split = int(len(graph_ids)*0.5)
    train_graphs, test_graphs = set(map(int,graph_ids[:split])), set(map(int,graph_ids[split:]))

    summary_rows = []
    for mk in args.models:
        model_df, layer_data, L, dec_layers, windows = extract_features_for_model(mk, MODEL_PATHS[mk], df, args, common)
        model_df.to_csv(out/f"{mk}_prompt_table.csv", index=False, encoding="utf-8-sig")
        train_mask = model_df["graph_id"].isin(train_graphs).values
        test_mask = model_df["graph_id"].isin(test_graphs).values

        scan_rows = []
        feature_cache = {}
        for w in windows:
            for k in args.k_list:
                feat_df, geom = compute_feature_df(model_df, layer_data, k, w["layers"])
                train_r2, train_corr, train_acc, train_f1 = traincv_score_window(feat_df, train_mask)
                test_r2, test_corr, test_acc, test_f1 = score_one(feat_df, train_mask, test_mask)
                row = {
                    "model_key": mk, "k": k, **{kk:vv for kk,vv in w.items() if kk!="layers"},
                    "layers": str(w["layers"]), **geom,
                    "traincv_downstream_r2": train_r2, "traincv_downstream_corr": train_corr,
                    "traincv_mechanism_acc": train_acc, "traincv_mechanism_f1": train_f1,
                    "test_downstream_r2": test_r2, "test_downstream_corr": test_corr,
                    "test_mechanism_acc": test_acc, "test_mechanism_f1": test_f1,
                }
                row["traincv_topology_score"] = row["invariance_score_1_minus_preserve"]
                row["traincv_mechanism_score"] = train_f1 if not np.isnan(train_f1) else -999
                row["traincv_downstream_score"] = train_corr if not np.isnan(train_corr) else -999
                row["traincv_overall_score"] = (
                    0.5*row["traincv_topology_score"] +
                    0.8*max(row["traincv_mechanism_score"],0) +
                    0.8*max(row["traincv_downstream_score"],0)
                )
                scan_rows.append(row)
        scan = pd.DataFrame(scan_rows)
        scan.to_csv(out/f"{mk}_traincv_window_scan.csv", index=False, encoding="utf-8-sig")
        for target, col, test_col in [
            ("topology","traincv_topology_score","invariance_score_1_minus_preserve"),
            ("mechanism","traincv_mechanism_score","test_mechanism_f1"),
            ("downstream","traincv_downstream_score","test_downstream_corr"),
            ("overall","traincv_overall_score","test_downstream_corr"),
        ]:
            best = scan.sort_values(col, ascending=False).iloc[0].to_dict()
            if target == "topology":
                test_score = best["invariance_score_1_minus_preserve"]
            elif target == "mechanism":
                test_score = best["test_mechanism_f1"]
            elif target == "downstream":
                test_score = best["test_downstream_corr"]
            else:
                test_score = best["test_downstream_corr"]
            summary_rows.append({
                "model_key": mk, "selection_target": target,
                "selected_by": col,
                "selected_window": best["window"],
                "selected_layers": best["layers"],
                "selected_k": best["k"],
                "traincv_selected_score": best[col],
                "heldout_test_score": test_score,
                "test_topology_score": best["invariance_score_1_minus_preserve"],
                "test_mechanism_f1": best["test_mechanism_f1"],
                "test_downstream_corr": best["test_downstream_corr"],
                "test_downstream_r2": best["test_downstream_r2"],
                "num_layers": L,
                "decision_layers": str(dec_layers),
            })
    pd.DataFrame(summary_rows).to_csv(out/"window_true_heldout_summary.csv", index=False, encoding="utf-8-sig")
    with open(out/"window_config.json","w",encoding="utf-8") as f:
        json.dump(vars(args)|{"model_paths":MODEL_PATHS, "common_labels":common,
                              "train_graphs":sorted(train_graphs), "test_graphs":sorted(test_graphs)}, f, indent=2, ensure_ascii=False)
    print("Saved:", out/"window_true_heldout_summary.csv")

if __name__ == "__main__":
    main()
