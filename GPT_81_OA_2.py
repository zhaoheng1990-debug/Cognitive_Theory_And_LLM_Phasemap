# ============================================================
# OA-2: W -> C Coupling Audit Lite
#
# Goal:
#   Test whether static W geometry predicts inference-time
#   constraint trajectory C(x), approximated by ΔR20:25.
#
# Main hypothesis:
#   W geometry around clean/conflict labels and relation tokens
#   predicts ΔR20:25 mechanism: stable / competition / closure.
#
# Output:
#   oa2_dataset.csv
#   oa2_summary.json
# ============================================================

import json, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

SAVE_DIR = Path("./oa2_w_to_c_outputs")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

SEED = 42
N_GRAPHS = 96
MAX_LEN = 260
BATCH_SIZE = 8
TRACK_LAYERS = list(range(20, 26))   # ΔR20:25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# LOAD
# =========================

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
    local_files_only=True,
    trust_remote_code=True,
)
if DEVICE == "cpu":
    model.to(DEVICE)

model.eval()

W = model.lm_head.weight.detach().float().cpu().numpy()
Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)

print("W shape:", W.shape)

# =========================
# TOKEN HELPERS
# =========================

def tok_ids(text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]

def cont_ids(label):
    return tok_ids(" " + label)

def single_token_id(label):
    ids = cont_ids(label)
    if len(ids) == 1:
        return ids[0]
    ids2 = tok_ids(label)
    if len(ids2) == 1:
        return ids2[0]
    return None

def vec_for_label(label):
    tid = single_token_id(label)
    if tid is None:
        return None, None
    return Wn[tid], tid

def cos(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a)+1e-9)*(np.linalg.norm(b)+1e-9)))

# =========================
# LABELS
# =========================

candidate_labels = [
    "Red", "Blue", "Green", "Yellow",
    "North", "South", "East", "West",
    "Alpha", "Beta", "Gamma", "Delta",
    "Circle", "Square", "Triangle", "Star",
    "Copper", "Silver", "Gold", "Iron",
    "Apple", "Orange", "Lemon", "Pear",
    "River", "Mountain", "Forest", "Ocean",
    "Sun", "Moon", "Cloud", "Stone",
]

label_pool = [x for x in candidate_labels if single_token_id(x) is not None]
if len(label_pool) < 12:
    raise RuntimeError("Need at least 12 single-token labels.")

print("Using labels:", label_pool[:12])

# =========================
# DATASET
# =========================

CONDITIONS = [
    ("clean", 0.0, "stable"),
    ("weak_distractor", 1.0, "stable"),
    ("ambiguous_branch", 2.0, "competition"),
    ("strong_branch_conflict", 3.0, "competition"),
    ("direct_location_conflict", 4.0, "competition"),
    ("closure_negation_conflict", 5.0, "closure"),
    ("closure_update", 5.0, "closure"),
    ("closure_exception", 5.0, "closure"),
]

def make_entity(prefix, i):
    return f"{prefix}{i:03d}"

def make_prompt(a, b, clean_label, conflict_label, irrelevant_label, variant):
    option_line = f"Possible location labels: {clean_label} or {conflict_label}."

    if variant == "clean":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Question: Which location label is {a} located at?",
            f"Answer with exactly one word: {clean_label} or {conflict_label}.",
            "Answer:",
        ]

    elif variant == "weak_distractor":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Weak note: another unrelated item may be associated with {conflict_label}.",
            f"Question: Which location label is {a} located at?",
            f"Answer with exactly one word: {clean_label} or {conflict_label}.",
            "Answer:",
        ]

    elif variant == "ambiguous_branch":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Ambiguous note: some descriptions also connect {a} with {conflict_label}.",
            f"Question: Which location label is {a} located at?",
            f"Answer with exactly one word: {clean_label} or {conflict_label}.",
            "Answer:",
        ]

    elif variant == "strong_branch_conflict":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Conflicting branch: {a} is also described as located at {conflict_label}.",
            f"Question: Which location label is {a} located at?",
            f"Answer with exactly one word: {clean_label} or {conflict_label}.",
            "Answer:",
        ]

    elif variant == "direct_location_conflict":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Fact 3: {a} is located at {conflict_label}.",
            f"Question: Which location label is {a} located at?",
            f"Answer with exactly one word: {clean_label} or {conflict_label}.",
            "Answer:",
        ]

    elif variant == "closure_negation_conflict":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Correction: the relation from {b} to {clean_label} is invalid in this case.",
            f"Fact 3: {b} is located at {conflict_label}.",
            f"Question: Which location label is {a} located at?",
            f"Answer with exactly one word: {clean_label} or {conflict_label}.",
            "Answer:",
        ]

    elif variant == "closure_update":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Initial rule: {b} is located at {clean_label}.",
            f"Update: for this graph, {b} is now located at {conflict_label}.",
            f"Fact: {a} belongs to {b}.",
            f"Question: Which location label is {a} located at after the update?",
            f"Answer with exactly one word: {clean_label} or {conflict_label}.",
            "Answer:",
        ]

    elif variant == "closure_exception":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"General rule: items belonging to {b} are located at {clean_label}.",
            f"Exception: {a} is a special case located at {conflict_label}.",
            f"Fact: {a} belongs to {b}.",
            f"Question: Which location label is {a} located at?",
            f"Answer with exactly one word: {clean_label} or {conflict_label}.",
            "Answer:",
        ]
    else:
        raise ValueError(variant)

    return "\n".join(lines)

rows = []
for gid in range(N_GRAPHS):
    a = make_entity("A", gid)
    b = make_entity("B", gid)
    clean_label = label_pool[(2 * gid) % len(label_pool)]
    conflict_label = label_pool[(2 * gid + 1) % len(label_pool)]
    irrelevant_label = label_pool[(2 * gid + 2) % len(label_pool)]

    for cond, eps, mech in CONDITIONS:
        prompt = make_prompt(a, b, clean_label, conflict_label, irrelevant_label, cond)
        rows.append({
            "graph_id": gid,
            "condition": cond,
            "epsilon": eps,
            "mechanism": mech,
            "prompt": prompt,
            "clean_label": clean_label,
            "conflict_label": conflict_label,
            "irrelevant_label": irrelevant_label,
        })

df = pd.DataFrame(rows)

# =========================
# STATIC W FEATURES
# =========================

evidence_tokens = ["evidence", "source", "proof", "claim", "citation", "document", "record"]
rule_tokens = ["rule", "exception", "override", "policy", "law", "authority", "requirement"]

E_vecs = [vec_for_label(t)[0] for t in evidence_tokens if vec_for_label(t)[0] is not None]
R_vecs = [vec_for_label(t)[0] for t in rule_tokens if vec_for_label(t)[0] is not None]
E_cent = np.mean(E_vecs, axis=0)
R_cent = np.mean(R_vecs, axis=0)
axis_ER = R_cent - E_cent
axis_ER = axis_ER / (np.linalg.norm(axis_ER) + 1e-9)

def w_static_features(row):
    vc, tc = vec_for_label(row["clean_label"])
    ve, te = vec_for_label(row["conflict_label"])
    vi, ti = vec_for_label(row["irrelevant_label"])

    if vc is None or ve is None:
        raise RuntimeError("Label tokenization failed.")

    # label-pair geometry
    clean_conflict_cos = cos(vc, ve)
    clean_er = cos(vc, axis_ER)
    conflict_er = cos(ve, axis_ER)
    er_gap_conflict_minus_clean = conflict_er - clean_er

    # local density around clean/conflict labels
    sims_c = Wn @ vc
    sims_e = Wn @ ve
    density_c_top50 = float(np.mean(np.sort(sims_c)[-51:-1]))
    density_e_top50 = float(np.mean(np.sort(sims_e)[-51:-1]))

    # geometry bias: which label lies in denser semantic region
    density_gap_conflict_minus_clean = density_e_top50 - density_c_top50

    # condition lexical binding: rule-like prompts should align with rule axis
    cond = row["condition"]
    cond_vecs = []
    for word in cond.replace("_", " ").split():
        v, _ = vec_for_label(word)
        if v is not None:
            cond_vecs.append(v)
    cond_er = float(np.mean([cos(v, axis_ER) for v in cond_vecs])) if cond_vecs else 0.0

    return {
        "W_clean_conflict_cos": clean_conflict_cos,
        "W_clean_ER": clean_er,
        "W_conflict_ER": conflict_er,
        "W_ER_gap_E_minus_C": er_gap_conflict_minus_clean,
        "W_clean_density50": density_c_top50,
        "W_conflict_density50": density_e_top50,
        "W_density_gap_E_minus_C": density_gap_conflict_minus_clean,
        "W_condition_ER": cond_er,
    }

wfeat = pd.DataFrame([w_static_features(r) for _, r in df.iterrows()])
df = pd.concat([df.reset_index(drop=True), wfeat], axis=1)

# =========================
# FORWARD FEATURES: R_l = logit(C)-logit(E)
# =========================

@torch.no_grad()
def compute_R_batch(prompts, clean_labels, conflict_labels):
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states

    # last non-padding position
    attn = enc["attention_mask"]
    last_idx = attn.sum(dim=1) - 1

    result = []
    for bi in range(len(prompts)):
        c_id = single_token_id(clean_labels[bi])
        e_id = single_token_id(conflict_labels[bi])

        r_by_layer = {}
        for layer in TRACK_LAYERS:
            # hidden_states index: 0 embedding, 1 after layer0...
            h = hs[layer + 1][bi, last_idx[bi], :]
            logits = model.lm_head(h).float()
            r = float(logits[c_id].detach().cpu() - logits[e_id].detach().cpu())
            r_by_layer[f"R{layer}"] = r
        result.append(r_by_layer)

    return result

all_R = []
for start in range(0, len(df), BATCH_SIZE):
    sub = df.iloc[start:start+BATCH_SIZE]
    Rs = compute_R_batch(
        sub["prompt"].tolist(),
        sub["clean_label"].tolist(),
        sub["conflict_label"].tolist(),
    )
    all_R.extend(Rs)
    print(f"Processed {min(start+BATCH_SIZE, len(df))}/{len(df)}")

rdf = pd.DataFrame(all_R)
df = pd.concat([df.reset_index(drop=True), rdf], axis=1)

# =========================
# ΔR relative to each graph's clean condition
# =========================

for layer in TRACK_LAYERS:
    base = df[df["condition"] == "clean"][["graph_id", f"R{layer}"]].rename(columns={f"R{layer}": f"R{layer}_clean"})
    df = df.merge(base, on="graph_id", how="left")
    df[f"dR{layer}"] = df[f"R{layer}"] - df[f"R{layer}_clean"]

delta_cols = [f"dR{l}" for l in TRACK_LAYERS]
df["deltaR_l2"] = np.sqrt(np.sum(np.square(df[delta_cols].values), axis=1))
df["deltaR_mean"] = np.mean(df[delta_cols].values, axis=1)
df["deltaR_min"] = np.min(df[delta_cols].values, axis=1)

# label: closure vs non-closure, mechanism multiclass
df["is_closure"] = (df["mechanism"] == "closure").astype(int)
df["is_comp_or_closure"] = (df["mechanism"] != "stable").astype(int)

# =========================
# COUPLING TESTS
# =========================

W_cols = [
    "W_clean_conflict_cos",
    "W_clean_ER",
    "W_conflict_ER",
    "W_ER_gap_E_minus_C",
    "W_clean_density50",
    "W_conflict_density50",
    "W_density_gap_E_minus_C",
    "W_condition_ER",
]

C_cols = delta_cols + ["deltaR_l2", "deltaR_mean", "deltaR_min"]

# correlations
corr_rows = []
for wc in W_cols:
    for cc in C_cols:
        x = df[wc].values
        y = df[cc].values
        if np.std(x) < 1e-9 or np.std(y) < 1e-9:
            corr = np.nan
        else:
            corr = float(np.corrcoef(x, y)[0,1])
        corr_rows.append({"W_feature": wc, "C_feature": cc, "pearson": corr})

corr_df = pd.DataFrame(corr_rows)

# prediction: W-only -> closure / unstable
def cv_binary_auc(X, y, groups=None, use_group=False):
    if use_group:
        cv = GroupKFold(n_splits=5)
        splits = cv.split(X, y, groups)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = cv.split(X, y)

    probs = np.zeros(len(y))
    for tr, te in splits:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
        clf.fit(X[tr], y[tr])
        probs[te] = clf.predict_proba(X[te])[:,1]

    auc = roc_auc_score(y, probs)
    pred = (probs >= 0.5).astype(int)
    return {
        "auc": float(auc),
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
    }

Xw = df[W_cols].values.astype(float)
groups = df["graph_id"].values

summary = {
    "n_rows": int(len(df)),
    "n_graphs": int(N_GRAPHS),
    "W_features": W_cols,
    "C_features": C_cols,
    "W_only_predict_closure_stratified": cv_binary_auc(Xw, df["is_closure"].values, use_group=False),
    "W_only_predict_closure_group": cv_binary_auc(Xw, df["is_closure"].values, groups=groups, use_group=True),
    "W_only_predict_nonstable_stratified": cv_binary_auc(Xw, df["is_comp_or_closure"].values, use_group=False),
    "W_only_predict_nonstable_group": cv_binary_auc(Xw, df["is_comp_or_closure"].values, groups=groups, use_group=True),
    "top_abs_correlations": corr_df.assign(abs_corr=corr_df["pearson"].abs()).sort_values("abs_corr", ascending=False).head(20).to_dict(orient="records"),
}

# =========================
# SAVE
# =========================

df.to_csv(SAVE_DIR / "oa2_dataset.csv", index=False, encoding="utf-8-sig")
corr_df.to_csv(SAVE_DIR / "oa2_w_c_correlations.csv", index=False, encoding="utf-8-sig")

with open(SAVE_DIR / "oa2_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\nOA-2 finished.")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("Saved to:", SAVE_DIR.resolve())