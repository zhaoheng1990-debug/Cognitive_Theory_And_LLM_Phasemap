# ============================================================
# OA-4I.3: Cross-Surface Invariance Audit
# Purpose: audit whether TopK_init_delta captures constraint structure
# rather than prompt surface/template leakage.
# ============================================================

import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, f1_score

warnings.filterwarnings('ignore')

# ===================== CONFIG =====================
MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
SAVE_DIR = Path('./oa4i3_cross_surface_outputs')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16 if DEVICE == 'cuda' else torch.float32
BATCH_SIZE = 4
MAX_LEN = 280
INIT_STATE_INDICES = list(range(0, 8))  # embedding + L0-L6
DECISION_LAYERS = [20, 21, 22, 23, 24, 25]
TOPK_LIST = [100, 500, 1000]
MAX_TOPK = max(TOPK_LIST)
ENTITY_LIMIT = 10
RELATION_LIMIT = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===================== LOAD MODEL =====================
print('Loading model/tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    device_map='auto' if DEVICE == 'cuda' else None,
    local_files_only=True,
    trust_remote_code=True,
)
if DEVICE == 'cpu':
    model.to(DEVICE)
model.eval()

num_layers = len(model.model.layers)
print('Num layers:', num_layers)
if max(DECISION_LAYERS) >= num_layers:
    raise RuntimeError(f'DECISION_LAYERS includes {max(DECISION_LAYERS)}, but model has only {num_layers} layers.')

W_cpu = model.lm_head.weight.detach().float().cpu().numpy()
W_norm = W_cpu / (np.linalg.norm(W_cpu, axis=1, keepdims=True) + 1e-9)
print('W shape:', W_norm.shape)

# ===================== TOKEN HELPERS =====================
def encode_no_special(text: str):
    return tokenizer(text, add_special_tokens=False)['input_ids']

def single_token_id(text: str):
    for cand in [text, ' ' + text, text.lower(), ' ' + text.lower()]:
        ids = encode_no_special(cand)
        if len(ids) == 1:
            return ids[0]
    return None

def is_single_token(text: str):
    return single_token_id(text) is not None

# ===================== FACTORS =====================
RAW_ENTITY_GRAPHS = [
    ('Paris', 'France', 'Red', 'Blue'),
    ('Berlin', 'Germany', 'Green', 'Yellow'),
    ('Tokyo', 'Japan', 'North', 'South'),
    ('Beijing', 'China', 'East', 'West'),
    ('doctor', 'hospital', 'Gold', 'Iron'),
    ('judge', 'court', 'Apple', 'Orange'),
    ('teacher', 'school', 'River', 'Mountain'),
    ('dog', 'animal', 'Sun', 'Moon'),
    ('cat', 'animal', 'Cloud', 'Stone'),
    ('river', 'water', 'Circle', 'Square'),
    ('tree', 'plant', 'Copper', 'Silver'),
    ('chef', 'kitchen', 'Lemon', 'Pear'),
]
ENTITY_GRAPHS = []
for g in RAW_ENTITY_GRAPHS:
    if all(is_single_token(x) for x in g):
        ENTITY_GRAPHS.append(g)
    else:
        print('Skipping non-single-token entity graph:', g)
ENTITY_GRAPHS = ENTITY_GRAPHS[:ENTITY_LIMIT]
if len(ENTITY_GRAPHS) < 6:
    raise RuntimeError('Too few valid entity graphs.')

RELATION_FORMS = [
    {'relation_id': 'belongs_associated', 'edge_ab': '{A} belongs to {B}.', 'edge_bc': '{B} is associated with {X}.', 'question': 'Which label is {A} associated with?'},
    {'relation_id': 'inside_maps', 'edge_ab': '{A} is inside {B}.', 'edge_bc': '{B} maps to {X}.', 'question': 'Which label does {A} map to?'},
    {'relation_id': 'member_category', 'edge_ab': '{A} is a member of {B}.', 'edge_bc': '{B} is linked to {X}.', 'question': 'Which label is {A} linked to?'},
    {'relation_id': 'contained_points', 'edge_ab': '{A} is contained in {B}.', 'edge_bc': '{B} points toward {X}.', 'question': 'Which label does {A} point toward?'},
][:RELATION_LIMIT]

STRUCTURES = [
    {'structure_id': 'clean', 'mechanism': 'stable', 'closure_like': 0},
    {'structure_id': 'weak_distractor', 'mechanism': 'stable_shift', 'closure_like': 0},
    {'structure_id': 'ambiguous_branch', 'mechanism': 'competition', 'closure_like': 0},
    {'structure_id': 'direct_conflict', 'mechanism': 'competition', 'closure_like': 0},
    {'structure_id': 'exception_override', 'mechanism': 'closure', 'closure_like': 1},
    {'structure_id': 'rule_update', 'mechanism': 'closure', 'closure_like': 1},
    {'structure_id': 'meta_override', 'mechanism': 'closure', 'closure_like': 1},
]

SURFACE_FAMILIES = [
    {'surface_family': 'canonical_fact', 'variant': 'facts', 'prefix': '', 'format': 'fact'},
    {'surface_family': 'compact_arrow', 'variant': 'arrows', 'prefix': 'Use the relation arrows below.\n', 'format': 'arrow'},
    {'surface_family': 'narrative', 'variant': 'story', 'prefix': 'Read the short description and answer.\n', 'format': 'narrative'},
    {'surface_family': 'database', 'variant': 'db', 'prefix': 'Database records:\n', 'format': 'database'},
    {'surface_family': 'instructional', 'variant': 'instruction', 'prefix': 'Follow the current instruction exactly.\n', 'format': 'instruction'},
    {'surface_family': 'minimal', 'variant': 'minimal', 'prefix': '', 'format': 'minimal'},
]

# ===================== PROMPT GENERATION =====================
def rel_edge_ab(rel, A, B): return rel['edge_ab'].format(A=A, B=B)
def rel_edge_bc(rel, B, X): return rel['edge_bc'].format(B=B, X=X)
def rel_question(rel, A): return rel['question'].format(A=A)

def core_semantics(A, B, C, E, rel, structure_id):
    edge_ab, edge_bc_C, edge_bc_E = rel_edge_ab(rel, A, B), rel_edge_bc(rel, B, C), rel_edge_bc(rel, B, E)
    question = rel_question(rel, A)
    if structure_id == 'clean':
        return {'lines': [edge_ab, edge_bc_C], 'question': question, 'exception': None}
    if structure_id == 'weak_distractor':
        return {'lines': [edge_ab, edge_bc_C, f'An unrelated note mentions the label {E}.'], 'question': question, 'exception': None}
    if structure_id == 'ambiguous_branch':
        return {'lines': [edge_ab, edge_bc_C, f'Some descriptions also connect {A} with {E}.'], 'question': question, 'exception': None}
    if structure_id == 'direct_conflict':
        return {'lines': [edge_ab, edge_bc_C, f'{A} is associated with {E}.'], 'question': question, 'exception': None}
    if structure_id == 'exception_override':
        return {'lines': [f'General rule: objects connected through {B} use label {C}.', edge_ab], 'question': question, 'exception': f'{A} is a special case using label {E}.'}
    if structure_id == 'rule_update':
        return {'lines': [f'Old rule: {edge_bc_C}', f'New rule: in this graph, {edge_bc_E}', edge_ab], 'question': question, 'exception': None}
    if structure_id == 'meta_override':
        return {'lines': ['Use only the latest rule. Ignore older or general rules.', f'Latest rule: {edge_bc_E}', edge_ab], 'question': question, 'exception': None}
    raise ValueError(structure_id)

def render_prompt(A, B, C, E, rel, structure_id, surface):
    sem = core_semantics(A, B, C, E, rel, structure_id)
    fmt, prefix = surface['format'], surface['prefix']
    options = f'{C} or {E}'
    if fmt == 'fact':
        body = [f'Possible answer labels: {options}.']
        body += [f'Fact {i}: {line}' for i, line in enumerate(sem['lines'], 1)]
        if sem['exception']: body.append(f'Exception: {sem["exception"]}')
        body += [f'Question: {sem["question"]}', f'Answer with exactly one word: {options}.', 'Answer:']
        return prefix + '\n'.join(body)
    if fmt == 'arrow':
        body = [f'Labels = [{C}, {E}]'] + [f'- {line}' for line in sem['lines']]
        if sem['exception']: body.append(f'- Exception override: {sem["exception"]}')
        body += [f'Query: {sem["question"]}', f'Return only one label from [{C}, {E}].', 'Answer:']
        return prefix + '\n'.join(body)
    if fmt == 'narrative':
        text = ' '.join(sem['lines'])
        if sem['exception']: text += ' However, ' + sem['exception']
        return prefix + f'{text}\nThe available labels are {C} and {E}. Now decide: {sem["question"]}\nUse a single word, either {C} or {E}.\nAnswer:'
    if fmt == 'database':
        body = [f'ANSWER_LABELS: {C} | {E}'] + [f'RECORD_{i}: {line}' for i, line in enumerate(sem['lines'], 1)]
        if sem['exception']: body.append(f'OVERRIDE_RECORD: {sem["exception"]}')
        body += [f'QUERY: {sem["question"]}', 'OUTPUT_FIELD: label', 'Answer:']
        return prefix + '\n'.join(body)
    if fmt == 'instruction':
        body = [f'Choose between {C} and {E}.']
        if structure_id in ['rule_update', 'meta_override']: body.append('Current information has priority over older information.')
        if structure_id == 'exception_override': body.append('Specific exceptions have priority over general rules.')
        body += sem['lines']
        if sem['exception']: body.append(sem['exception'])
        body += [sem['question'], f'Final answer must be exactly {C} or {E}.', 'Answer:']
        return prefix + '\n'.join(body)
    if fmt == 'minimal':
        body = sem['lines'][:]
        if sem['exception']: body.append(sem['exception'])
        body += [f'{sem["question"]} ({C}/{E})', 'Answer:']
        return prefix + '\n'.join(body)
    raise ValueError(fmt)

# ===================== BUILD DATASET =====================
rows = []
for entity_idx, (A, B, C, E) in enumerate(ENTITY_GRAPHS):
    for rel_idx, rel in enumerate(RELATION_FORMS):
        for surf_idx, surface in enumerate(SURFACE_FAMILIES):
            for struct_idx, struct in enumerate(STRUCTURES):
                rows.append({
                    'entity_idx': entity_idx, 'relation_idx': rel_idx, 'surface_idx': surf_idx, 'structure_idx': struct_idx,
                    'entity_id': f'E{entity_idx}', 'relation_id': rel['relation_id'],
                    'surface_family': surface['surface_family'], 'surface_variant': surface['variant'],
                    'structure_id': struct['structure_id'], 'mechanism': struct['mechanism'], 'closure_like': int(struct['closure_like']),
                    'A': A, 'B': B, 'C': C, 'E': E,
                    'prompt': render_prompt(A, B, C, E, rel, struct['structure_id'], surface),
                })
df = pd.DataFrame(rows)
print('Dataset rows:', len(df))
print('Entities:', df['entity_id'].nunique(), 'Relations:', df['relation_id'].nunique(), 'Surface families:', df['surface_family'].nunique(), 'Structures:', df['structure_id'].nunique())

# ===================== FORWARD HELPERS =====================
def state_name(si): return 'emb' if si == 0 else f'L{si-1}'
def entropy_np(vals):
    vals = vals.astype(np.float64); vals = vals - np.max(vals)
    p = np.exp(vals); p = p / (np.sum(p) + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))

@torch.no_grad()
def compute_batch(prompts, clean_labels, conflict_labels):
    enc = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs, attn = out.hidden_states, enc['attention_mask']
    last_idx = attn.sum(dim=1) - 1
    rows, centers_payload, topids_payload = [], {}, {}
    for si in INIT_STATE_INDICES:
        lname = state_name(si)
        for k in TOPK_LIST:
            key = f'{lname}_k{k}'
            centers_payload[key], topids_payload[key] = [], []
    for bi in range(len(prompts)):
        c_id, e_id = single_token_id(clean_labels[bi]), single_token_id(conflict_labels[bi])
        if c_id is None or e_id is None:
            raise RuntimeError(f'Non-single label: {clean_labels[bi]} / {conflict_labels[bi]}')
        item = {}
        for si in INIT_STATE_INDICES:
            lname = state_name(si)
            h = hs[si][bi, last_idx[bi], :].to(model.lm_head.weight.dtype)
            logits = model.lm_head(h)
            logit_c, logit_e = logits[c_id], logits[e_id]
            item[f'init_{lname}_R_CminusE'] = float((logit_c - logit_e).detach().cpu())
            item[f'init_{lname}_rank_C'] = int((logits > logit_c).sum().detach().cpu().item() + 1)
            item[f'init_{lname}_rank_E'] = int((logits > logit_e).sum().detach().cpu().item() + 1)
            item[f'init_{lname}_rank_gap_EminusC'] = item[f'init_{lname}_rank_E'] - item[f'init_{lname}_rank_C']
            top_vals, top_idx = torch.topk(logits, k=MAX_TOPK)
            vals, idx = top_vals.detach().float().cpu().numpy(), top_idx.detach().cpu().numpy().astype(np.int32)
            for k in TOPK_LIST:
                vals_k, idx_k = vals[:k], idx[:k]
                item[f'init_{lname}_k{k}_top1'] = float(vals_k[0])
                item[f'init_{lname}_k{k}_kth'] = float(vals_k[-1])
                item[f'init_{lname}_k{k}_meanlogit'] = float(np.mean(vals_k))
                item[f'init_{lname}_k{k}_stdlogit'] = float(np.std(vals_k))
                item[f'init_{lname}_k{k}_gap_top1_kth'] = float(vals_k[0] - vals_k[-1])
                item[f'init_{lname}_k{k}_entropy'] = entropy_np(vals_k)
                item[f'init_{lname}_k{k}_has_C'] = int(c_id in set(idx_k.tolist()))
                item[f'init_{lname}_k{k}_has_E'] = int(e_id in set(idx_k.tolist()))
                center = np.mean(W_norm[idx_k], axis=0)
                center = center / (np.linalg.norm(center) + 1e-9)
                key = f'{lname}_k{k}'
                centers_payload[key].append(center.astype(np.float32))
                topids_payload[key].append(idx_k.copy())
        for layer in DECISION_LAYERS:
            h = hs[layer + 1][bi, last_idx[bi], :].to(model.lm_head.weight.dtype)
            logits = model.lm_head(h)
            item[f'R{layer}'] = float((logits[c_id] - logits[e_id]).detach().cpu())
        rows.append(item)
    return rows, centers_payload, topids_payload

# ===================== RUN FORWARD =====================
all_rows, center_store, topid_store = [], {}, {}
for si in INIT_STATE_INDICES:
    lname = state_name(si)
    for k in TOPK_LIST:
        key = f'{lname}_k{k}'
        center_store[key], topid_store[key] = [], []

for start in range(0, len(df), BATCH_SIZE):
    sub = df.iloc[start:start+BATCH_SIZE]
    rows, centers_payload, topids_payload = compute_batch(sub['prompt'].tolist(), sub['C'].tolist(), sub['E'].tolist())
    all_rows.extend(rows)
    for key in center_store:
        center_store[key].extend(centers_payload[key])
        topid_store[key].extend(topids_payload[key])
    print(f'Processed {min(start+BATCH_SIZE, len(df))}/{len(df)}')
    if torch.cuda.is_available(): torch.cuda.empty_cache()

feat_df = pd.DataFrame(all_rows)
df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
for key in center_store:
    center_store[key] = np.stack(center_store[key], axis=0)
    topid_store[key] = np.stack(topid_store[key], axis=0)

# ===================== CLEAN BASELINE =====================
clean_key_cols = ['entity_id', 'relation_id', 'surface_family']
clean_lookup = {}
for idx, row in df[df['structure_id'] == 'clean'].iterrows():
    clean_lookup[tuple(row[c] for c in clean_key_cols)] = idx
expected = df['entity_id'].nunique() * df['relation_id'].nunique() * df['surface_family'].nunique()
if len(clean_lookup) != expected:
    raise RuntimeError(f'Clean lookup incomplete: got {len(clean_lookup)}, expected {expected}')

for layer in DECISION_LAYERS:
    clean_vals = []
    for _, row in df.iterrows():
        clean_vals.append(df.loc[clean_lookup[tuple(row[c] for c in clean_key_cols)], f'R{layer}'])
    df[f'R{layer}_clean'] = clean_vals
    df[f'dR{layer}'] = df[f'R{layer}'] - df[f'R{layer}_clean']

dR_cols = [f'dR{x}' for x in DECISION_LAYERS]
df['deltaR_l2'] = np.sqrt(np.sum(np.square(df[dR_cols].values), axis=1))
df['deltaR_mean'] = np.mean(df[dR_cols].values, axis=1)
df['deltaR_min'] = np.min(df[dR_cols].values, axis=1)
df['deltaR_final25'] = df['dR25']

def cosine_rows(a, b):
    return np.sum(a*b, axis=1) / ((np.linalg.norm(a, axis=1)+1e-9) * (np.linalg.norm(b, axis=1)+1e-9))

for key in center_store:
    centers, topids = center_store[key], topid_store[key]
    clean_centers, clean_topids = np.zeros_like(centers), np.zeros_like(topids)
    for i, row in df[clean_key_cols].iterrows():
        clean_idx = clean_lookup[tuple(row[c] for c in clean_key_cols)]
        clean_centers[i], clean_topids[i] = centers[clean_idx], topids[clean_idx]
    cos = cosine_rows(centers, clean_centers)
    df[f'topk_{key}_center_cos_to_clean'] = cos
    df[f'topk_{key}_center_dist_to_clean'] = 1.0 - cos
    k = int(key.split('_k')[-1])
    jaccards = []
    for i in range(len(df)):
        s1, s2 = set(topids[i, :k].tolist()), set(clean_topids[i, :k].tolist())
        jaccards.append(len(s1.intersection(s2)) / max(len(s1.union(s2)), 1))
    df[f'topk_{key}_jaccard_to_clean'] = jaccards
    df[f'topk_{key}_jaccard_loss'] = 1.0 - np.array(jaccards)

# ===================== FEATURE SETS =====================
rank_features = [c for c in df.columns if c.startswith('init_') and (c.endswith('_R_CminusE') or '_rank_' in c)]
topk_logit_features = [c for c in df.columns if c.startswith('init_') and any(s in c for s in ['_top1','_kth','_meanlogit','_stdlogit','_gap_top1_kth','_entropy','_has_C','_has_E'])]
topk_delta_features = [c for c in df.columns if c.startswith('topk_') and any(s in c for s in ['center_cos_to_clean','center_dist_to_clean','jaccard_to_clean','jaccard_loss'])]
all_topk_features = rank_features + topk_logit_features + topk_delta_features
feature_sets = {'rank_features': rank_features, 'topk_logit_features': topk_logit_features, 'topk_delta_features': topk_delta_features, 'all_topk_features': all_topk_features}
print('Feature counts:', {k: len(v) for k, v in feature_sets.items()})

# ===================== FACTOR VARIANCE =====================
def factor_eta2(feature_cols, factor_col):
    X = df[feature_cols].values.astype(float)
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True)+1e-9)
    grand = X.mean(axis=0, keepdims=True)
    ss_total = np.sum((X-grand)**2)
    ss_factor = 0.0
    for _, sub in df.groupby(factor_col):
        idx = sub.index.values
        m = X[idx].mean(axis=0, keepdims=True)
        ss_factor += len(idx) * np.sum((m-grand)**2)
    return float(ss_factor / (ss_total+1e-12))

factor_cols = ['entity_id','relation_id','surface_family','structure_id','mechanism']
factor_rows = []
for fs_name, cols in feature_sets.items():
    for fac in factor_cols:
        factor_rows.append({'feature_set': fs_name, 'factor': fac, 'eta2': factor_eta2(cols, fac)})
for fac in factor_cols:
    factor_rows.append({'feature_set': 'decision_dR', 'factor': fac, 'eta2': factor_eta2(dR_cols, fac)})
factor_variance_df = pd.DataFrame(factor_rows)

# ===================== PREDICTION =====================
def group_regression(feature_cols, target='deltaR_l2', group_col='surface_family'):
    X, y, groups = df[feature_cols].values.astype(float), df[target].values.astype(float), df[group_col].values
    n_splits = min(5, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)
    pred = np.zeros(len(y))
    for tr, te in cv.split(X, y, groups):
        reg = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
        reg.fit(X[tr], y[tr])
        pred[te] = reg.predict(X[te])
    return {'target': target, 'group_col': group_col, 'n_splits': int(n_splits), 'r2': float(r2_score(y, pred)), 'corr': float(np.corrcoef(y, pred)[0,1]) if np.std(pred)>1e-9 else np.nan}

def group_binary(feature_cols, target_col='closure_like', group_col='surface_family'):
    X, y, groups = df[feature_cols].values.astype(float), df[target_col].values.astype(int), df[group_col].values
    n_splits = min(5, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)
    prob = np.zeros(len(y))
    for tr, te in cv.split(X, y, groups):
        clf = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=3000, class_weight='balanced'))])
        clf.fit(X[tr], y[tr])
        prob[te] = clf.predict_proba(X[te])[:,1]
    pred = (prob >= 0.5).astype(int)
    return {'target': target_col, 'group_col': group_col, 'n_splits': int(n_splits), 'auc': float(roc_auc_score(y, prob)), 'acc': float(accuracy_score(y, pred)), 'f1': float(f1_score(y, pred))}

def logo_regression(feature_cols, target='deltaR_l2', group_col='surface_family'):
    X, y, groups = df[feature_cols].values.astype(float), df[target].values.astype(float), df[group_col].values
    logo = LeaveOneGroupOut(); pred = np.zeros(len(y))
    for tr, te in logo.split(X, y, groups):
        reg = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
        reg.fit(X[tr], y[tr]); pred[te] = reg.predict(X[te])
    return {'target': target, 'group_col': group_col, 'n_groups': int(len(np.unique(groups))), 'r2': float(r2_score(y, pred)), 'corr': float(np.corrcoef(y, pred)[0,1]) if np.std(pred)>1e-9 else np.nan}

def logo_binary(feature_cols, target_col='closure_like', group_col='surface_family'):
    X, y, groups = df[feature_cols].values.astype(float), df[target_col].values.astype(int), df[group_col].values
    logo = LeaveOneGroupOut(); prob = np.zeros(len(y))
    for tr, te in logo.split(X, y, groups):
        clf = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=3000, class_weight='balanced'))])
        clf.fit(X[tr], y[tr]); prob[te] = clf.predict_proba(X[te])[:,1]
    pred = (prob >= 0.5).astype(int)
    return {'target': target_col, 'group_col': group_col, 'n_groups': int(len(np.unique(groups))), 'auc': float(roc_auc_score(y, prob)), 'acc': float(accuracy_score(y, pred)), 'f1': float(f1_score(y, pred))}

group_cols = ['surface_family', 'entity_id', 'relation_id']
groupkfold_reg, groupkfold_bin, logo_reg, logo_bin = {}, {}, {}, {}
for fs_name, cols in feature_sets.items():
    groupkfold_reg[fs_name], groupkfold_bin[fs_name], logo_reg[fs_name], logo_bin[fs_name] = {}, {}, {}, {}
    for gcol in group_cols:
        groupkfold_reg[fs_name][gcol] = group_regression(cols, 'deltaR_l2', gcol)
        groupkfold_bin[fs_name][gcol] = group_binary(cols, 'closure_like', gcol)
        logo_reg[fs_name][gcol] = logo_regression(cols, 'deltaR_l2', gcol)
        logo_bin[fs_name][gcol] = logo_binary(cols, 'closure_like', gcol)

# ===================== IMPORTANCE + MEANS =====================
def feature_importance(feature_cols, target='deltaR_l2'):
    X, y = df[feature_cols].values.astype(float), df[target].values.astype(float)
    pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
    pipe.fit(X, y)
    rows = [{'feature': f, 'coef': float(w), 'abs_coef': float(abs(w))} for f, w in zip(feature_cols, pipe.named_steps['ridge'].coef_)]
    return pd.DataFrame(rows).sort_values('abs_coef', ascending=False)
importance_df = feature_importance(all_topk_features, 'deltaR_l2')

compact_cols = ['deltaR_l2','deltaR_mean','deltaR_min','deltaR_final25']
for c in topk_delta_features:
    if ('L0_k500' in c or 'L2_k500' in c or 'L6_k500' in c) and ('center_dist_to_clean' in c or 'jaccard_loss' in c):
        compact_cols.append(c)
compact_cols = compact_cols[:80]
condition_means = df.groupby(['structure_id','mechanism'])[compact_cols].mean().reset_index().to_dict(orient='records')
surface_means = df.groupby('surface_family')[compact_cols].mean().reset_index().to_dict(orient='records')

# ===================== SAVE =====================
summary = {
    'experiment': 'OA-4I.3 Cross-Surface Invariance Audit',
    'n_rows': int(len(df)),
    'n_entities': int(df['entity_id'].nunique()),
    'n_relations': int(df['relation_id'].nunique()),
    'n_surface_families': int(df['surface_family'].nunique()),
    'n_structures': int(df['structure_id'].nunique()),
    'init_state_indices': INIT_STATE_INDICES,
    'decision_layers': DECISION_LAYERS,
    'topk_list': TOPK_LIST,
    'feature_counts': {k: len(v) for k, v in feature_sets.items()},
    'factor_variance_eta2': factor_variance_df.to_dict(orient='records'),
    'groupkfold_regression_deltaR_l2': groupkfold_reg,
    'groupkfold_binary_closure_like': groupkfold_bin,
    'leave_one_group_regression_deltaR_l2': logo_reg,
    'leave_one_group_binary_closure_like': logo_bin,
    'condition_means': condition_means,
    'surface_means': surface_means,
    'top_50_feature_importance': importance_df.head(50).to_dict(orient='records'),
}

df.to_csv(SAVE_DIR / 'oa4i3_dataset.csv', index=False, encoding='utf-8-sig')
factor_variance_df.to_csv(SAVE_DIR / 'oa4i3_factor_variance.csv', index=False, encoding='utf-8-sig')
importance_df.to_csv(SAVE_DIR / 'oa4i3_feature_importance.csv', index=False, encoding='utf-8-sig')
with open(SAVE_DIR / 'oa4i3_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print('\nSaved to:', SAVE_DIR.resolve())
