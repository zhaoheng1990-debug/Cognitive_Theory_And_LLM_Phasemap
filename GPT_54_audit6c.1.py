# ============================================================
# Audit-6C.1
# Answer-Targeted Breakdown Test
#
# Goal:
#   Audit-6C showed:
#       relation / closure perturbation -> VIM neighborhood drift
#       but macro tau/path remained stable.
#
#   Audit-6C.1 asks:
#       Does the clean answer C lose advantage against conflict answer E?
#
# We track per layer:
#   1. first-token logit-lens score:
#        score(C), score(E), margin = C - E
#
#   2. first-token rank:
#        rank(C), rank(E), rank_gap = rank(E) - rank(C)
#
#   3. VIM TopK membership:
#        C in TopK?
#        E in TopK?
#
#   4. VIM neighborhood center bias:
#        cos(C_k(H_l), center(C)) - cos(C_k(H_l), center(E))
#
#   5. boundary L20-L22 summary.
#
# Important:
#   - Logit/rank uses raw hidden states and LM head.
#   - VIM TopK and VIM center use CENTER_HIDDEN_FOR_TOPK=True
#     to stay consistent with Audit-5B / 6A / 6B / 6C.
#
# This is still an internal diagnostic, not final hallucination evaluation.
# ============================================================

import os
import gc
import random
import warnings
from collections import defaultdict

import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

device = "cuda"
dtype = torch.float16

MAX_LEN = 220
BATCH_SIZE = 4

RANDOM_SEED = 42

TOPK = 5000
CENTER_HIDDEN_FOR_TOPK = True

N_GRAPHS = 96

EXPECTED_TAU_ZONE = {20, 21, 22}
BOUNDARY_LAYERS = [20, 21, 22]

# Optional final-generation check. Keep False for faster internal audit.
DO_GENERATION_CHECK = False
GEN_MAX_NEW_TOKENS = 8

SAVE_DIR = "./audit6c1_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

# ============================================================
# SYNTHETIC RELATION GRAPH DATA
# ============================================================

def make_entity(prefix, i):
    return f"{prefix}{i:03d}"

def make_prompt(a, b, c, d, e, f, variant):
    if variant == "clean":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located in {c}.",
            f"Question: Where is {a} located?",
            "Answer:",
        ]

    elif variant == "preserve_irrelevant":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located in {c}.",
            f"Irrelevant fact 1: {d} belongs to {e}.",
            f"Irrelevant fact 2: {e} is located in {f}.",
            f"Question: Where is {a} located?",
            "Answer:",
        ]

    elif variant == "weak_distractor":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located in {c}.",
            f"Weak distractor: Some sources say {a} belongs to {d}.",
            f"Distractor fact: {d} is located in {e}.",
            f"Question: Where is {a} located?",
            "Answer:",
        ]

    elif variant == "ambiguous_branch":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located in {c}.",
            f"Ambiguous fact: {a} may also belong to {d}.",
            f"Fact 4: {d} is located in {e}.",
            f"Question: Where is {a} located?",
            "Answer:",
        ]

    elif variant == "strong_branch_conflict":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located in {c}.",
            f"Conflicting fact 1: {a} belongs to {d}.",
            f"Conflicting fact 2: {d} is located in {e}.",
            f"Question: Where is {a} located?",
            "Answer:",
        ]

    elif variant == "direct_location_conflict":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located in {c}.",
            f"Conflicting direct fact: {a} is located in {e}.",
            f"Question: Where is {a} located?",
            "Answer:",
        ]

    elif variant == "closure_negation_conflict":
        lines = [
            "You are given a small relation graph.",
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located in {c}.",
            f"Updated record: {b} is located in {e}, not in {c}.",
            f"Question: Where is {a} located?",
            "Answer:",
        ]

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return "\n".join(lines)

CONDITION_META = [
    ("clean", 0.0, "clean"),
    ("preserve_irrelevant", 0.5, "relation_preserving"),
    ("weak_distractor", 1.0, "weak_break"),
    ("ambiguous_branch", 2.0, "ambiguous_break"),
    ("strong_branch_conflict", 3.0, "competing_closure"),
    ("direct_location_conflict", 4.0, "direct_conflict"),
    ("closure_negation_conflict", 5.0, "closure_conflict"),
]

conditions = [x[0] for x in CONDITION_META]
epsilon_by_condition = {x[0]: x[1] for x in CONDITION_META}
status_by_condition = {x[0]: x[2] for x in CONDITION_META}

def build_dataset(n_graphs):
    records = []
    texts_by_condition = {c: [] for c in conditions}

    for i in range(n_graphs):
        a = make_entity("A", i)
        b = make_entity("B", i)
        c = make_entity("C", i)

        d = make_entity("D", i)
        e = make_entity("E", i)
        f = make_entity("F", i)

        prompts = {}

        for cond in conditions:
            prompts[cond] = make_prompt(a, b, c, d, e, f, variant=cond)
            texts_by_condition[cond].append(prompts[cond])

        records.append({
            "idx": i,
            "core_entities": (a, b, c),
            "distractor_entities": (d, e, f),
            "answer_clean": c,
            "answer_conflict": e,
            "prompts": prompts,
        })

    return records, texts_by_condition

records, texts_by_condition = build_dataset(N_GRAPHS)

print("Conditions:")
for cond in conditions:
    print(f"  {cond:<28} eps={epsilon_by_condition[cond]} status={status_by_condition[cond]}")

print("\nExample clean prompt:\n")
print(texts_by_condition["clean"][0])

print("\nExample closure_negation_conflict prompt:\n")
print(texts_by_condition["closure_negation_conflict"][0])

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
)

model.eval()

num_layers = len(model.model.layers)
print("Num layers:", num_layers)

W_lm = model.lm_head.weight.detach().float().cpu()
vocab_size, d_model = W_lm.shape
print("LM head:", tuple(W_lm.shape))

W_np = W_lm.numpy().astype(np.float32)
W_norm = W_np / (np.linalg.norm(W_np, axis=1, keepdims=True) + 1e-12)

# ============================================================
# CANDIDATE TOKEN HELPERS
# ============================================================

def tokenize_variants(text):
    """
    Candidate can tokenize differently depending on leading whitespace.
    We keep multiple variants and use the max first-token logit.
    """
    variants = [
        text,
        " " + text,
        "\n" + text,
    ]

    token_lists = []

    for v in variants:
        ids = tokenizer(
            v,
            add_special_tokens=False,
            return_tensors=None,
        )["input_ids"]

        if isinstance(ids, list) and len(ids) > 0:
            token_lists.append(ids)

    return token_lists

def candidate_info(entity):
    token_lists = tokenize_variants(entity)

    first_ids = sorted(set([ids[0] for ids in token_lists if len(ids) > 0]))

    all_ids = []
    for ids in token_lists:
        all_ids.extend(ids)

    all_ids = sorted(set(all_ids))

    if len(first_ids) == 0:
        raise ValueError(f"No token ids for entity={entity}")

    E = W_norm[np.array(all_ids, dtype=np.int64)]
    center = E.mean(axis=0)
    center = center / (np.linalg.norm(center) + 1e-12)

    return {
        "entity": entity,
        "first_ids": first_ids,
        "all_ids": all_ids,
        "center": center.astype(np.float32),
        "token_lists": token_lists,
    }

# Precompute candidate data per sample
candidate_data = []

for rec in records:
    c = rec["answer_clean"]
    e = rec["answer_conflict"]

    c_info = candidate_info(c)
    e_info = candidate_info(e)

    candidate_data.append({
        "idx": rec["idx"],
        "clean_entity": c,
        "conflict_entity": e,
        "clean": c_info,
        "conflict": e_info,
    })

print("\nExample candidate tokenization:")
print("  clean entity   :", candidate_data[0]["clean_entity"])
print("  clean variants :", candidate_data[0]["clean"]["token_lists"])
print("  conflict entity:", candidate_data[0]["conflict_entity"])
print("  conflict vars  :", candidate_data[0]["conflict"]["token_lists"])

# ============================================================
# HIDDEN CAPTURE
# ============================================================

def capture_hidden_states(texts, condition_name=""):
    all_layer_states = [[] for _ in range(num_layers)]

    print(f"\nCapturing hidden states for condition={condition_name} ...")

    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start:start + BATCH_SIZE]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
            ).to(device)

            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

            hidden_states = outputs.hidden_states[1:]

            attn = inputs["attention_mask"]
            last_idx = attn.sum(dim=1) - 1

            for l in range(num_layers):
                h = hidden_states[l]
                picked = h[
                    torch.arange(h.shape[0], device=h.device),
                    last_idx,
                    :
                ]
                all_layer_states[l].append(picked.detach().float().cpu())

            if (start // BATCH_SIZE) % 10 == 0:
                print(f"  captured {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")

    H_raws = []

    for l in range(num_layers):
        H = torch.cat(all_layer_states[l], dim=0).numpy().astype(np.float32)
        H_raws.append(H)

    print(f"Done condition={condition_name}. Shape L00={H_raws[0].shape}")

    return H_raws

# ============================================================
# TOPK / VIM HELPERS
# ============================================================

def compute_topk_ids_and_center(H, k):
    """
    H: [N, D]
    Returns:
      ids: [N, k]
      Ck : [N, D] normalized center of TopK token embeddings
      spread: [N]
    """
    logits = H @ W_np.T
    ids = np.argpartition(logits, -k, axis=1)[:, -k:]

    vals = np.take_along_axis(logits, ids, axis=1)
    order = np.argsort(vals, axis=1)[:, ::-1]
    ids = np.take_along_axis(ids, order, axis=1).astype(np.int32)

    N = ids.shape[0]
    Ck = np.zeros((N, d_model), dtype=np.float32)
    spread = np.zeros(N, dtype=np.float32)

    batch_rows = 4

    for s in range(0, N, batch_rows):
        e = min(s + batch_rows, N)
        batch_ids = ids[s:e]

        E = W_norm[batch_ids]
        C = E.mean(axis=1)
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)

        sims = np.einsum("bkd,bd->bk", E, C)
        spr = np.mean(1.0 - sims, axis=1)

        Ck[s:e] = C.astype(np.float32)
        spread[s:e] = spr.astype(np.float32)

    return ids, Ck, spread

def ids_membership(topk_ids_row, candidate_ids):
    s = set(topk_ids_row.tolist())
    return any(t in s for t in candidate_ids)

def candidate_score_from_logits(logits_row, first_ids):
    vals = logits_row[np.array(first_ids, dtype=np.int64)]
    return float(np.max(vals))

def candidate_rank_from_logits(logits_row, score):
    # 1 = best rank
    return int(1 + np.sum(logits_row > score))

def answer_target_metrics_for_condition(cond, H_raws):
    """
    For each layer:
      raw logit-lens score/rank for C and E
      VIM TopK membership
      VIM center bias to C-center vs E-center
    """

    print(f"\nComputing answer-targeted metrics for condition={cond} ...")

    rows = []

    for l in range(num_layers):
        H_raw = H_raws[l]

        # raw logits for answer logit-lens
        raw_logits = H_raw @ W_np.T

        # VIM TopK / center
        if CENTER_HIDDEN_FOR_TOPK:
            H_vim = H_raw - H_raw.mean(axis=0, keepdims=True)
        else:
            H_vim = H_raw

        topk_ids, Ck, spread = compute_topk_ids_and_center(H_vim, TOPK)

        c_scores = []
        e_scores = []
        margins = []

        c_ranks = []
        e_ranks = []
        rank_gaps = []

        c_in_topk = []
        e_in_topk = []

        center_bias = []
        clean_center_cos = []
        conflict_center_cos = []

        seq_center_margin = []

        for i in range(N_GRAPHS):
            c_info = candidate_data[i]["clean"]
            e_info = candidate_data[i]["conflict"]

            logits_i = raw_logits[i]

            c_score = candidate_score_from_logits(logits_i, c_info["first_ids"])
            e_score = candidate_score_from_logits(logits_i, e_info["first_ids"])

            c_rank = candidate_rank_from_logits(logits_i, c_score)
            e_rank = candidate_rank_from_logits(logits_i, e_score)

            c_in = ids_membership(topk_ids[i], c_info["first_ids"])
            e_in = ids_membership(topk_ids[i], e_info["first_ids"])

            ck = Ck[i]

            cc = float(np.dot(ck, c_info["center"]))
            ec = float(np.dot(ck, e_info["center"]))

            # Multi-token candidate center proxy using normalized hidden
            h_norm = H_raw[i] / (np.linalg.norm(H_raw[i]) + 1e-12)
            c_seq = float(np.dot(h_norm, c_info["center"]))
            e_seq = float(np.dot(h_norm, e_info["center"]))

            c_scores.append(c_score)
            e_scores.append(e_score)
            margins.append(c_score - e_score)

            c_ranks.append(c_rank)
            e_ranks.append(e_rank)
            rank_gaps.append(e_rank - c_rank)

            c_in_topk.append(float(c_in))
            e_in_topk.append(float(e_in))

            clean_center_cos.append(cc)
            conflict_center_cos.append(ec)
            center_bias.append(cc - ec)

            seq_center_margin.append(c_seq - e_seq)

        row = {
            "condition": cond,
            "epsilon": epsilon_by_condition[cond],
            "status": status_by_condition[cond],
            "layer": l,

            "mean_clean_score": float(np.mean(c_scores)),
            "mean_conflict_score": float(np.mean(e_scores)),
            "mean_score_margin_C_minus_E": float(np.mean(margins)),

            "median_rank_clean": float(np.median(c_ranks)),
            "median_rank_conflict": float(np.median(e_ranks)),
            "mean_rank_gap_E_minus_C": float(np.mean(rank_gaps)),

            "clean_in_topk_rate": float(np.mean(c_in_topk)),
            "conflict_in_topk_rate": float(np.mean(e_in_topk)),
            "topk_membership_gap_C_minus_E": float(np.mean(c_in_topk) - np.mean(e_in_topk)),

            "mean_clean_center_cos": float(np.mean(clean_center_cos)),
            "mean_conflict_center_cos": float(np.mean(conflict_center_cos)),
            "mean_center_bias_C_minus_E": float(np.mean(center_bias)),

            "mean_seq_center_margin_C_minus_E": float(np.mean(seq_center_margin)),

            "spread_mean": float(np.mean(spread)),
            "spread_std": float(np.std(spread)),
        }

        rows.append(row)

        print(
            f"L{l:02d} "
            f"margin={row['mean_score_margin_C_minus_E']:+.4f} "
            f"rankC={row['median_rank_clean']:.0f} "
            f"rankE={row['median_rank_conflict']:.0f} "
            f"CtopK={row['clean_in_topk_rate']:.3f} "
            f"EtopK={row['conflict_in_topk_rate']:.3f} "
            f"bias={row['mean_center_bias_C_minus_E']:+.5f}"
        )

    return rows

# ============================================================
# OPTIONAL GENERATION CHECK
# ============================================================

def generation_check_for_condition(cond, n=20):
    if not DO_GENERATION_CHECK:
        return []

    print(f"\nGeneration check for {cond} ...")

    rows = []

    prompts = texts_by_condition[cond][:n]

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            out = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
            )

            gen_ids = out[0][inputs["input_ids"].shape[1]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            clean_ans = records[i]["answer_clean"]
            conflict_ans = records[i]["answer_conflict"]

            has_clean = clean_ans in gen_text
            has_conflict = conflict_ans in gen_text

            row = {
                "condition": cond,
                "idx": i,
                "generated": gen_text,
                "clean_answer": clean_ans,
                "conflict_answer": conflict_ans,
                "has_clean": has_clean,
                "has_conflict": has_conflict,
            }

            rows.append(row)

            print(f"[{i}] gen={repr(gen_text)} clean={clean_ans} conflict={conflict_ans}")

    return rows

# ============================================================
# MAIN LOOP
# ============================================================

all_rows = []
condition_summaries = []
generation_rows = []

for cond in conditions:
    print("\n\n############################################################")
    print(f"Condition: {cond}")
    print(f"epsilon={epsilon_by_condition[cond]}, status={status_by_condition[cond]}")
    print("############################################################\n")

    H_raws = capture_hidden_states(texts_by_condition[cond], condition_name=cond)

    rows = answer_target_metrics_for_condition(cond, H_raws)
    all_rows.extend(rows)

    gen_rows = generation_check_for_condition(cond, n=20)
    generation_rows.extend(gen_rows)

    # Summary
    boundary = [r for r in rows if r["layer"] in BOUNDARY_LAYERS]
    late = [r for r in rows if r["layer"] >= 20]
    all_layer = rows

    def mean_of(xs, key):
        return float(np.mean([x[key] for x in xs]))

    def median_of(xs, key):
        return float(np.median([x[key] for x in xs]))

    summary = {
        "condition": cond,
        "epsilon": epsilon_by_condition[cond],
        "status": status_by_condition[cond],

        "all_margin": mean_of(all_layer, "mean_score_margin_C_minus_E"),
        "boundary_margin": mean_of(boundary, "mean_score_margin_C_minus_E"),
        "late_margin": mean_of(late, "mean_score_margin_C_minus_E"),

        "all_rank_gap": mean_of(all_layer, "mean_rank_gap_E_minus_C"),
        "boundary_rank_gap": mean_of(boundary, "mean_rank_gap_E_minus_C"),
        "late_rank_gap": mean_of(late, "mean_rank_gap_E_minus_C"),

        "boundary_clean_in_topk": mean_of(boundary, "clean_in_topk_rate"),
        "boundary_conflict_in_topk": mean_of(boundary, "conflict_in_topk_rate"),
        "boundary_topk_gap": mean_of(boundary, "topk_membership_gap_C_minus_E"),

        "boundary_center_bias": mean_of(boundary, "mean_center_bias_C_minus_E"),
        "boundary_seq_center_margin": mean_of(boundary, "mean_seq_center_margin_C_minus_E"),

        "L20_margin": rows[20]["mean_score_margin_C_minus_E"],
        "L21_margin": rows[21]["mean_score_margin_C_minus_E"],
        "L22_margin": rows[22]["mean_score_margin_C_minus_E"],

        "L20_center_bias": rows[20]["mean_center_bias_C_minus_E"],
        "L21_center_bias": rows[21]["mean_center_bias_C_minus_E"],
        "L22_center_bias": rows[22]["mean_center_bias_C_minus_E"],

        "L20_clean_in_topk": rows[20]["clean_in_topk_rate"],
        "L20_conflict_in_topk": rows[20]["conflict_in_topk_rate"],
        "L21_clean_in_topk": rows[21]["clean_in_topk_rate"],
        "L21_conflict_in_topk": rows[21]["conflict_in_topk_rate"],
        "L22_clean_in_topk": rows[22]["clean_in_topk_rate"],
        "L22_conflict_in_topk": rows[22]["conflict_in_topk_rate"],
    }

    condition_summaries.append(summary)

    print("\nCondition answer-targeted summary:")
    print(f"  boundary margin C-E       : {summary['boundary_margin']:+.4f}")
    print(f"  boundary rank gap E-C     : {summary['boundary_rank_gap']:+.2f}")
    print(f"  boundary C in TopK        : {summary['boundary_clean_in_topk']:.4f}")
    print(f"  boundary E in TopK        : {summary['boundary_conflict_in_topk']:.4f}")
    print(f"  boundary TopK gap C-E     : {summary['boundary_topk_gap']:+.4f}")
    print(f"  boundary center bias C-E  : {summary['boundary_center_bias']:+.5f}")
    print(f"  boundary seq margin C-E   : {summary['boundary_seq_center_margin']:+.5f}")

    del H_raws
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# FINAL CURVE
# ============================================================

print("\n\n============================================================")
print("Audit-6C.1 Final Answer-Targeted Breakdown Curve")
print("============================================================\n")

print(
    f"{'Cond':<28}"
    f"{'eps':<6}"
    f"{'status':<22}"
    f"{'B_margin':<12}"
    f"{'B_rankGap':<12}"
    f"{'B_CtopK':<10}"
    f"{'B_EtopK':<10}"
    f"{'B_TopKGap':<12}"
    f"{'B_bias':<12}"
    f"{'B_seq':<12}"
)

for s in sorted(condition_summaries, key=lambda x: x["epsilon"]):
    print(
        f"{s['condition']:<28}"
        f"{s['epsilon']:<6.1f}"
        f"{s['status']:<22}"
        f"{s['boundary_margin']:<12.4f}"
        f"{s['boundary_rank_gap']:<12.2f}"
        f"{s['boundary_clean_in_topk']:<10.4f}"
        f"{s['boundary_conflict_in_topk']:<10.4f}"
        f"{s['boundary_topk_gap']:<12.4f}"
        f"{s['boundary_center_bias']:<12.5f}"
        f"{s['boundary_seq_center_margin']:<12.5f}"
    )

# ============================================================
# SAVE OUTPUTS
# ============================================================

try:
    import pandas as pd

    df_layers = pd.DataFrame(all_rows)
    layer_path = os.path.join(SAVE_DIR, "audit6c1_layerwise_answer_metrics.csv")
    df_layers.to_csv(layer_path, index=False)

    df_summary = pd.DataFrame(condition_summaries)
    summary_path = os.path.join(SAVE_DIR, "audit6c1_answer_breakdown_curve.csv")
    df_summary.to_csv(summary_path, index=False)

    if len(generation_rows) > 0:
        df_gen = pd.DataFrame(generation_rows)
        gen_path = os.path.join(SAVE_DIR, "audit6c1_generation_check.csv")
        df_gen.to_csv(gen_path, index=False)
    else:
        gen_path = None

    prompt_path = os.path.join(SAVE_DIR, "audit6c1_prompts.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records[:10]):
            f.write(f"===== graph {i} =====\n")
            f.write(f"clean answer: {rec['answer_clean']}\n")
            f.write(f"conflict answer: {rec['answer_conflict']}\n")
            for cond in conditions:
                f.write(f"\n--- {cond} | eps={epsilon_by_condition[cond]} | {status_by_condition[cond]} ---\n")
                f.write(rec["prompts"][cond] + "\n")
            f.write("\n")

    print("\nSaved outputs:")
    print(" ", layer_path)
    print(" ", summary_path)
    if gen_path is not None:
        print(" ", gen_path)
    print(" ", prompt_path)

except Exception as e:
    print("\nCould not save outputs:", repr(e))

# ============================================================
# INTERPRETATION GUIDE
# ============================================================

print("\n\n============================================================")
print("Audit-6C.1 Interpretation Guide")
print("============================================================\n")

print("Main expected pattern:")
print("  As epsilon increases:")
print("    boundary margin C-E should decrease.")
print("    boundary rank gap E-C should decrease.")
print("      positive rank_gap means C ranks better than E.")
print("    conflict answer E should enter TopK more often.")
print("    center bias C-E should move toward zero or negative.")
print()

print("If 6C.1 shows degradation while 6C tau remained stable:")
print("  relation breaking affects answer-level neighborhood competition")
print("  before it affects macro phase boundary tau.")
print()

print("If closure_negation_conflict causes the strongest C->E shift:")
print("  closure preservation is a stronger invariant candidate than raw relation identity.")
print()

print("If all answer-targeted metrics stay stable:")
print("  current synthetic conflicts are not strong enough,")
print("  or first-token logit lens is not capturing entity-level answer competition.")
print()

print("Recommended next step after 6C.1:")
print("  If C/E competition appears, run 6C.2 with actual generation labels.")
print("  If no C/E competition appears, strengthen prompts or use explicit answer options.")
print()

print("Done.")