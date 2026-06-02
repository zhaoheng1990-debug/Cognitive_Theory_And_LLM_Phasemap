# ============================================================
# Audit-6C.2b
# Single-token / Low-shared-token Answer Label Test
#
# Goal:
#   Audit-6C.2 showed output-level C->E flip, but boundary B_seq
#   was hard to interpret because C000/E000 are multi-token labels
#   sharing suffix tokens such as "000".
#
#   Audit-6C.2b replaces C000/E000 answers with short label words:
#       Red / Blue / Green / Yellow / ...
#
#   It auto-selects labels that tokenize as single tokens under the
#   model tokenizer when used as a continuation after "Answer:".
#
# Metrics:
#   1. layerwise sequence/logit-lens margin:
#        Score(clean_label) - Score(conflict_label)
#
#   2. boundary L20-L22 margin
#   3. late L23-L27 margin
#   4. final layer margin
#   5. greedy generation exact label check
#
# Expected:
#   - relation-preserving conditions should keep clean label.
#   - conflict conditions should shift toward conflict label.
#   - boundary score should become more interpretable than 6C.2.
# ============================================================

import os
import re
import gc
import random
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH =  r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"

device = "cuda"
dtype = torch.float16

MAX_LEN = 260
BATCH_SIZE = 4

RANDOM_SEED = 42
N_GRAPHS = 96

BOUNDARY_LAYERS = [20, 21, 22]
LATE_LAYERS = [23, 24, 25, 26, 27]

# For this audit, canonical scoring is deliberately strict.
# Candidate continuation is exactly " " + label after "Answer:".
CANONICAL_ONLY = True

# Generation check
DO_GENERATION_CHECK = True
GEN_BATCH_SIZE = 4
GEN_MAX_NEW_TOKENS = 8

SAVE_DIR = "./audit6c2b_outputs"
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
print("LM head:", tuple(model.lm_head.weight.shape))

# ============================================================
# TOKENIZATION HELPERS
# ============================================================

def tokenize_no_special(text):
    ids = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors=None,
    )["input_ids"]
    return list(ids)

def continuation_ids(label):
    # The prompt ends with "Answer:", so the natural continuation is " Red".
    return tokenize_no_special(" " + label)

def is_single_token_label(label):
    return len(continuation_ids(label)) == 1

def select_single_token_labels():
    """
    Try to find short labels that are single-token continuations.
    If too few are single-token, fall back to low-shared-token labels.
    """

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

    single = []
    multi = []

    for lab in candidate_labels:
        ids = continuation_ids(lab)
        if len(ids) == 1:
            single.append(lab)
        else:
            multi.append(lab)

    print("\nLabel tokenization audit:")
    for lab in candidate_labels[:20]:
        print(f"  {lab:<10} continuation_ids={continuation_ids(lab)} len={len(continuation_ids(lab))}")

    if len(single) >= 8:
        chosen = single[:12]
        print("\nUsing single-token labels:", chosen)
        return chosen

    # Fallback: use shortest tokenization labels.
    sorted_labels = sorted(candidate_labels, key=lambda x: len(continuation_ids(x)))
    chosen = sorted_labels[:12]

    print("\nWARNING: fewer than 8 single-token labels found.")
    print("Using shortest-token labels:", chosen)

    return chosen

LABEL_POOL = select_single_token_labels()

if len(LABEL_POOL) < 4:
    raise RuntimeError("Need at least 4 usable labels. Add more candidate labels.")

# ============================================================
# SYNTHETIC RELATION GRAPH DATA
# ============================================================

def make_entity(prefix, i):
    return f"{prefix}{i:03d}"

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

def make_prompt(a, b, d, clean_label, conflict_label, irrelevant_label_1, irrelevant_label_2, variant):
    """
    Base graph:
        A belongs_to B
        B located_at clean_label
        Therefore A located_at clean_label

    Conflict graph uses conflict_label.

    Labels are answer words, ideally single-token continuations.
    """

    option_line = (
        f"Possible location labels for the queried item are: {clean_label} or {conflict_label}."
    )

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

    elif variant == "preserve_irrelevant":
        lines = [
            "You are given a small relation graph.",
            option_line,
            f"Fact 1: {a} belongs to {b}.",
            f"Fact 2: {b} is located at {clean_label}.",
            f"Irrelevant fact 1: {d} belongs to an unrelated group.",
            f"Irrelevant fact 2: that unrelated group is located at {irrelevant_label_1}.",
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
            f"Weak distractor: Some sources say {a} may be associated with {conflict_label}.",
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
            f"Ambiguous fact: {a} may also be located at {conflict_label}.",
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
            f"Conflicting fact 1: {a} belongs to another group.",
            f"Conflicting fact 2: that other group is located at {conflict_label}.",
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
            f"Conflicting direct fact: {a} is located at {conflict_label}.",
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
            f"Updated record: {b} is located at {conflict_label}, not at {clean_label}.",
            f"Question: Which location label is {a} located at?",
            f"Answer with exactly one word: {clean_label} or {conflict_label}.",
            "Answer:",
        ]

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return "\n".join(lines)

def build_dataset(n_graphs):
    records = []
    texts_by_condition = {c: [] for c in conditions}

    n_labels = len(LABEL_POOL)

    for i in range(n_graphs):
        a = make_entity("A", i)
        b = make_entity("B", i)
        d = make_entity("D", i)

        # Rotate labels across samples to reduce fixed label bias.
        clean_label = LABEL_POOL[(2 * i) % n_labels]
        conflict_label = LABEL_POOL[(2 * i + 1) % n_labels]

        # Pick irrelevant labels that are not clean/conflict when possible.
        irrelevant_label_1 = LABEL_POOL[(2 * i + 2) % n_labels]
        irrelevant_label_2 = LABEL_POOL[(2 * i + 3) % n_labels]

        prompts = {}

        for cond in conditions:
            prompts[cond] = make_prompt(
                a=a,
                b=b,
                d=d,
                clean_label=clean_label,
                conflict_label=conflict_label,
                irrelevant_label_1=irrelevant_label_1,
                irrelevant_label_2=irrelevant_label_2,
                variant=cond,
            )
            texts_by_condition[cond].append(prompts[cond])

        records.append({
            "idx": i,
            "entities": (a, b, d),
            "clean_label": clean_label,
            "conflict_label": conflict_label,
            "irrelevant_labels": (irrelevant_label_1, irrelevant_label_2),
            "prompts": prompts,
        })

    return records, texts_by_condition

records, texts_by_condition = build_dataset(N_GRAPHS)

print("\nConditions:")
for cond in conditions:
    print(f"  {cond:<28} eps={epsilon_by_condition[cond]} status={status_by_condition[cond]}")

print("\nExample clean prompt:\n")
print(texts_by_condition["clean"][0])

print("\nExample closure_negation_conflict prompt:\n")
print(texts_by_condition["closure_negation_conflict"][0])

# ============================================================
# SCORING EXAMPLES
# ============================================================

def candidate_continuations(label):
    if CANONICAL_ONLY:
        return [" " + label]
    return [" " + label, "\n" + label, label]

def build_scoring_examples_for_condition(cond):
    examples = []

    for rec in records:
        idx = rec["idx"]
        prompt = rec["prompts"][cond]

        for answer_type, label in [
            ("clean", rec["clean_label"]),
            ("conflict", rec["conflict_label"]),
        ]:
            for variant_id, continuation in enumerate(candidate_continuations(label)):
                prompt_ids = tokenize_no_special(prompt)
                continuation_ids_list = tokenize_no_special(continuation)

                full_ids = prompt_ids + continuation_ids_list
                prompt_len = len(prompt_ids)
                target_ids = continuation_ids_list

                if len(target_ids) == 0:
                    continue

                pred_positions = list(range(prompt_len - 1, prompt_len - 1 + len(target_ids)))

                if len(full_ids) > MAX_LEN:
                    continue

                examples.append({
                    "idx": idx,
                    "condition": cond,
                    "answer_type": answer_type,
                    "label": label,
                    "variant_id": variant_id,
                    "continuation": continuation,
                    "input_ids": full_ids,
                    "prompt_len": prompt_len,
                    "target_ids": target_ids,
                    "pred_positions": pred_positions,
                })

    return examples

# ============================================================
# BATCHING
# ============================================================

def pad_batch(input_id_lists, pad_id):
    max_len = max(len(x) for x in input_id_lists)
    batch = []
    attn = []

    for ids in input_id_lists:
        pad_len = max_len - len(ids)
        batch.append(ids + [pad_id] * pad_len)
        attn.append([1] * len(ids) + [0] * pad_len)

    return (
        torch.tensor(batch, dtype=torch.long, device=device),
        torch.tensor(attn, dtype=torch.long, device=device),
    )

def get_lm_logits_from_hidden(hidden):
    return model.lm_head(hidden)

# ============================================================
# LAYERWISE SEQUENCE SCORING
# ============================================================

def score_examples_layerwise(examples, condition_name):
    print(f"\nScoring condition={condition_name} ...")
    print("Total scoring examples:", len(examples))

    scores = {
        l: defaultdict(lambda: {"clean": [], "conflict": []})
        for l in range(num_layers)
    }

    pad_id = tokenizer.pad_token_id

    with torch.no_grad():
        for start in range(0, len(examples), BATCH_SIZE):
            batch_examples = examples[start:start + BATCH_SIZE]

            input_ids_list = [ex["input_ids"] for ex in batch_examples]
            input_ids, attention_mask = pad_batch(input_ids_list, pad_id)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

            hidden_states = outputs.hidden_states[1:]

            for l in range(num_layers):
                hs = hidden_states[l]

                for bi, ex in enumerate(batch_examples):
                    pred_positions = ex["pred_positions"]
                    target_ids = ex["target_ids"]

                    h_sel = hs[bi, pred_positions, :]
                    logits = get_lm_logits_from_hidden(h_sel)
                    log_probs = F.log_softmax(logits.float(), dim=-1)

                    target = torch.tensor(target_ids, dtype=torch.long, device=log_probs.device)
                    token_logps = log_probs[
                        torch.arange(len(target), device=log_probs.device),
                        target
                    ]

                    seq_logp = float(token_logps.sum().detach().cpu())

                    scores[l][ex["idx"]][ex["answer_type"]].append(seq_logp)

            if (start // BATCH_SIZE) % 20 == 0:
                print(f"  scored {min(start + BATCH_SIZE, len(examples))}/{len(examples)} examples")

            del outputs, hidden_states, input_ids, attention_mask
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    layer_rows = []

    for l in range(num_layers):
        clean_scores = []
        conflict_scores = []
        margins = []

        clean_win = 0
        conflict_win = 0
        ties = 0

        for idx in range(N_GRAPHS):
            c_list = scores[l][idx]["clean"]
            e_list = scores[l][idx]["conflict"]

            if len(c_list) == 0 or len(e_list) == 0:
                continue

            c_score = max(c_list)
            e_score = max(e_list)
            margin = c_score - e_score

            clean_scores.append(c_score)
            conflict_scores.append(e_score)
            margins.append(margin)

            if margin > 1e-6:
                clean_win += 1
            elif margin < -1e-6:
                conflict_win += 1
            else:
                ties += 1

        row = {
            "condition": condition_name,
            "epsilon": epsilon_by_condition[condition_name],
            "status": status_by_condition[condition_name],
            "layer": l,
            "mean_clean_seq_logp": float(np.mean(clean_scores)),
            "mean_conflict_seq_logp": float(np.mean(conflict_scores)),
            "mean_seq_margin_C_minus_E": float(np.mean(margins)),
            "median_seq_margin_C_minus_E": float(np.median(margins)),
            "clean_win_rate": clean_win / max(len(margins), 1),
            "conflict_win_rate": conflict_win / max(len(margins), 1),
            "tie_rate": ties / max(len(margins), 1),
            "n_scored": len(margins),
        }

        layer_rows.append(row)

        print(
            f"L{l:02d} "
            f"seq_margin={row['mean_seq_margin_C_minus_E']:+.4f} "
            f"Cwin={row['clean_win_rate']:.3f} "
            f"Ewin={row['conflict_win_rate']:.3f} "
            f"tie={row['tie_rate']:.3f}"
        )

    return layer_rows

# ============================================================
# GENERATION CHECK
# ============================================================

def normalize_first_word(text):
    text = text.strip()
    if not text:
        return ""

    # Keep alphabetic label-like first word.
    m = re.search(r"[A-Za-z]+", text)
    if not m:
        return ""

    return m.group(0).lower()

def generation_check_condition(cond):
    if not DO_GENERATION_CHECK:
        return []

    print(f"\nGeneration check condition={cond} ...")

    rows = []
    prompts = texts_by_condition[cond]
    pad_id = tokenizer.pad_token_id

    with torch.no_grad():
        for start in range(0, len(prompts), GEN_BATCH_SIZE):
            batch_prompts = prompts[start:start + GEN_BATCH_SIZE]

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
            ).to(device)

            prompt_lens = inputs["attention_mask"].sum(dim=1).detach().cpu().numpy().tolist()

            out = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            for bi in range(len(batch_prompts)):
                idx = start + bi
                rec = records[idx]

                gen_ids = out[bi, prompt_lens[bi]:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

                first_word = normalize_first_word(gen_text)

                clean_label = rec["clean_label"]
                conflict_label = rec["conflict_label"]

                is_clean = first_word == clean_label.lower()
                is_conflict = first_word == conflict_label.lower()

                # Secondary containment check for inspection only.
                contains_clean = re.search(rf"\b{re.escape(clean_label)}\b", gen_text, re.IGNORECASE) is not None
                contains_conflict = re.search(rf"\b{re.escape(conflict_label)}\b", gen_text, re.IGNORECASE) is not None

                row = {
                    "condition": cond,
                    "epsilon": epsilon_by_condition[cond],
                    "status": status_by_condition[cond],
                    "idx": idx,
                    "generated": gen_text,
                    "first_word": first_word,
                    "clean_label": clean_label,
                    "conflict_label": conflict_label,
                    "is_clean_first_word": bool(is_clean),
                    "is_conflict_first_word": bool(is_conflict),
                    "contains_clean": bool(contains_clean),
                    "contains_conflict": bool(contains_conflict),
                }

                rows.append(row)

            if (start // GEN_BATCH_SIZE) % 10 == 0:
                print(f"  generated {min(start + GEN_BATCH_SIZE, len(prompts))}/{len(prompts)}")

            del inputs, out
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    clean_rate = float(np.mean([r["is_clean_first_word"] for r in rows]))
    conflict_rate = float(np.mean([r["is_conflict_first_word"] for r in rows]))
    other_rate = 1.0 - clean_rate - conflict_rate

    print(
        f"Generation summary {cond}: "
        f"clean_rate={clean_rate:.4f}, conflict_rate={conflict_rate:.4f}, other_rate={other_rate:.4f}"
    )

    return rows

# ============================================================
# SUMMARY HELPERS
# ============================================================

def mean_of(rows, key):
    return float(np.mean([r[key] for r in rows]))

def summarize_condition(layer_rows, gen_rows):
    cond = layer_rows[0]["condition"]

    boundary = [r for r in layer_rows if r["layer"] in BOUNDARY_LAYERS]
    late = [r for r in layer_rows if r["layer"] in LATE_LAYERS]
    final_layer = layer_rows[-1]

    if len(gen_rows) > 0:
        gen_clean_rate = float(np.mean([r["is_clean_first_word"] for r in gen_rows]))
        gen_conflict_rate = float(np.mean([r["is_conflict_first_word"] for r in gen_rows]))
        gen_other_rate = 1.0 - gen_clean_rate - gen_conflict_rate
        gen_contains_clean = float(np.mean([r["contains_clean"] for r in gen_rows]))
        gen_contains_conflict = float(np.mean([r["contains_conflict"] for r in gen_rows]))
    else:
        gen_clean_rate = np.nan
        gen_conflict_rate = np.nan
        gen_other_rate = np.nan
        gen_contains_clean = np.nan
        gen_contains_conflict = np.nan

    summary = {
        "condition": cond,
        "epsilon": epsilon_by_condition[cond],
        "status": status_by_condition[cond],

        "boundary_seq_margin": mean_of(boundary, "mean_seq_margin_C_minus_E"),
        "boundary_clean_win_rate": mean_of(boundary, "clean_win_rate"),
        "boundary_conflict_win_rate": mean_of(boundary, "conflict_win_rate"),

        "late_seq_margin": mean_of(late, "mean_seq_margin_C_minus_E"),
        "late_clean_win_rate": mean_of(late, "clean_win_rate"),
        "late_conflict_win_rate": mean_of(late, "conflict_win_rate"),

        "final_seq_margin": final_layer["mean_seq_margin_C_minus_E"],
        "final_clean_win_rate": final_layer["clean_win_rate"],
        "final_conflict_win_rate": final_layer["conflict_win_rate"],

        "L20_seq_margin": layer_rows[20]["mean_seq_margin_C_minus_E"],
        "L21_seq_margin": layer_rows[21]["mean_seq_margin_C_minus_E"],
        "L22_seq_margin": layer_rows[22]["mean_seq_margin_C_minus_E"],
        "L23_seq_margin": layer_rows[23]["mean_seq_margin_C_minus_E"],
        "L24_seq_margin": layer_rows[24]["mean_seq_margin_C_minus_E"],
        "L27_seq_margin": layer_rows[27]["mean_seq_margin_C_minus_E"],

        "gen_clean_rate": gen_clean_rate,
        "gen_conflict_rate": gen_conflict_rate,
        "gen_other_rate": gen_other_rate,
        "gen_contains_clean": gen_contains_clean,
        "gen_contains_conflict": gen_contains_conflict,
    }

    return summary

# ============================================================
# MAIN
# ============================================================

all_layer_rows = []
all_generation_rows = []
condition_summaries = []

for cond in conditions:
    print("\n\n############################################################")
    print(f"Condition: {cond}")
    print(f"epsilon={epsilon_by_condition[cond]}, status={status_by_condition[cond]}")
    print("############################################################\n")

    examples = build_scoring_examples_for_condition(cond)

    layer_rows = score_examples_layerwise(examples, cond)
    all_layer_rows.extend(layer_rows)

    gen_rows = generation_check_condition(cond)
    all_generation_rows.extend(gen_rows)

    summary = summarize_condition(layer_rows, gen_rows)
    condition_summaries.append(summary)

    print("\nCondition single-label sequence summary:")
    print(f"  boundary seq margin C-E   : {summary['boundary_seq_margin']:+.4f}")
    print(f"  boundary C win rate       : {summary['boundary_clean_win_rate']:.4f}")
    print(f"  boundary E win rate       : {summary['boundary_conflict_win_rate']:.4f}")
    print(f"  late seq margin C-E       : {summary['late_seq_margin']:+.4f}")
    print(f"  final seq margin C-E      : {summary['final_seq_margin']:+.4f}")
    print(f"  generation clean rate     : {summary['gen_clean_rate']:.4f}")
    print(f"  generation conflict rate  : {summary['gen_conflict_rate']:.4f}")
    print(f"  generation other rate     : {summary['gen_other_rate']:.4f}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# FINAL TABLE
# ============================================================

print("\n\n============================================================")
print("Audit-6C.2b Final Single-label Breakdown Curve")
print("============================================================\n")

print(
    f"{'Cond':<28}"
    f"{'eps':<6}"
    f"{'status':<22}"
    f"{'B_seq':<12}"
    f"{'B_Cwin':<10}"
    f"{'B_Ewin':<10}"
    f"{'Late_seq':<12}"
    f"{'Final_seq':<12}"
    f"{'Gen_C':<10}"
    f"{'Gen_E':<10}"
    f"{'Gen_Other':<10}"
)

for s in sorted(condition_summaries, key=lambda x: x["epsilon"]):
    print(
        f"{s['condition']:<28}"
        f"{s['epsilon']:<6.1f}"
        f"{s['status']:<22}"
        f"{s['boundary_seq_margin']:<12.4f}"
        f"{s['boundary_clean_win_rate']:<10.4f}"
        f"{s['boundary_conflict_win_rate']:<10.4f}"
        f"{s['late_seq_margin']:<12.4f}"
        f"{s['final_seq_margin']:<12.4f}"
        f"{s['gen_clean_rate']:<10.4f}"
        f"{s['gen_conflict_rate']:<10.4f}"
        f"{s['gen_other_rate']:<10.4f}"
    )

# ============================================================
# SAVE OUTPUTS
# ============================================================

try:
    import pandas as pd

    df_layers = pd.DataFrame(all_layer_rows)
    layer_path = os.path.join(SAVE_DIR, "audit6c2b_layerwise_single_label_scores.csv")
    df_layers.to_csv(layer_path, index=False)

    df_summary = pd.DataFrame(condition_summaries)
    summary_path = os.path.join(SAVE_DIR, "audit6c2b_single_label_breakdown_curve.csv")
    df_summary.to_csv(summary_path, index=False)

    if len(all_generation_rows) > 0:
        df_gen = pd.DataFrame(all_generation_rows)
        gen_path = os.path.join(SAVE_DIR, "audit6c2b_generation_check.csv")
        df_gen.to_csv(gen_path, index=False)
    else:
        gen_path = None

    prompt_path = os.path.join(SAVE_DIR, "audit6c2b_prompts.txt")
    with open(prompt_path, "w", encoding="utf-8") as fp:
        fp.write("LABEL_POOL:\n")
        fp.write(str(LABEL_POOL) + "\n\n")

        fp.write("Label tokenization:\n")
        for lab in LABEL_POOL:
            fp.write(f"{lab}: continuation_ids={continuation_ids(lab)}\n")

        fp.write("\n\nPrompts:\n")
        for i, rec in enumerate(records[:10]):
            fp.write(f"===== graph {i} =====\n")
            fp.write(f"clean label: {rec['clean_label']}\n")
            fp.write(f"conflict label: {rec['conflict_label']}\n")
            for cond in conditions:
                fp.write(
                    f"\n--- {cond} | eps={epsilon_by_condition[cond]} | "
                    f"{status_by_condition[cond]} ---\n"
                )
                fp.write(rec["prompts"][cond] + "\n")
            fp.write("\n")

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
print("Audit-6C.2b Interpretation Guide")
print("============================================================\n")

print("Primary question:")
print("  Does replacing C000/E000 with short answer labels make boundary B_seq interpretable?")
print()

print("Strong confirmation if:")
print("  1. clean / preserve_irrelevant keep positive late/final margins and Gen_C high.")
print("  2. ambiguous / strong / direct / closure conflicts reduce margins and increase Gen_E.")
print("  3. closure_negation_conflict remains among the strongest E-shift conditions.")
print("  4. B_seq is less pathologically negative in clean than in Audit-6C.2.")
print()

print("Important possible outcomes:")
print("  A. If B_seq becomes positive for clean:")
print("     The negative clean B_seq in 6C.2 was mainly tokenization/shared-suffix artifact.")
print()
print("  B. If B_seq is still negative for clean, but late/final/generation are correct:")
print("     L20-L22 is not answer-commitment; it is candidate re-coupling / competition.")
print()
print("  C. If late/final/generation still flip under conflicts:")
print("     6C.2 output-level result is robust to answer-label tokenization.")
print()
print("  D. If generation produces 'other' frequently:")
print("     The label prompt is too weak; strengthen with multiple-choice formatting.")
print()

print("Recommended next step:")
print("  If 6C.2b confirms robust flip, freeze Audit-6C as complete and move to Constraint-Audit-1.")
print()

print("Done.")