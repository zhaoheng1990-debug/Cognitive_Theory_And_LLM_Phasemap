# ============================================================
# Audit-6C.2
# Sequence-level Answer Scoring + Generation Check
#
# Goal:
#   Audit-6C.1 showed first-token answer competition:
#       C first-token vs E first-token
#
#   Audit-6C.2 upgrades this to full sequence scoring:
#       log P(C000 | prompt) - log P(E000 | prompt)
#
#   For each layer l, we compute a logit-lens style
#   teacher-forced candidate sequence score:
#
#       Score_l(answer)
#       =
#       sum_j log P_l(answer_j | prompt, answer_<j)
#
#   Candidate variants:
#       " C000"
#       "\nC000"
#       "C000"
#
#   For each entity, we take max score over variants.
#
# Metrics:
#   1. Layerwise sequence margin:
#        seq_margin = score(C) - score(E)
#
#   2. Boundary L20-L22 sequence margin.
#
#   3. Late L23-L27 sequence margin.
#
#   4. Final layer sequence margin.
#
#   5. Greedy generation output check:
#        does output contain clean answer C?
#        does output contain conflict answer E?
#
# Interpretation:
#   If seq_margin flips from positive to negative under conflict,
#   then relation/closure breaking affects full answer sequence,
#   not only first-token competition.
#
# ============================================================

import os
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

# Sequence scoring uses more forward passes than 6C.1.
# If VRAM is limited, reduce to 2 or 4.
BATCH_SIZE = 4

RANDOM_SEED = 42

N_GRAPHS = 96

BOUNDARY_LAYERS = [20, 21, 22]
LATE_LAYERS = [23, 24, 25, 26, 27]

# Generation check
DO_GENERATION_CHECK = True
GEN_BATCH_SIZE = 4
GEN_MAX_NEW_TOKENS = 8

SAVE_DIR = "./audit6c2_outputs"
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

def make_prompt(a, b, c, d, e, f_entity, variant):
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
            f"Irrelevant fact 2: {e} is located in {f_entity}.",
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
        f_entity = make_entity("F", i)

        prompts = {}

        for cond in conditions:
            prompts[cond] = make_prompt(a, b, c, d, e, f_entity, variant=cond)
            texts_by_condition[cond].append(prompts[cond])

        records.append({
            "idx": i,
            "core_entities": (a, b, c),
            "distractor_entities": (d, e, f_entity),
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

# Move lm_head weight handling through model.lm_head directly.
# This avoids manually storing huge logits matrices on CPU.
vocab_size = model.lm_head.weight.shape[0]
d_model = model.lm_head.weight.shape[1]
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

def candidate_variants(entity):
    return [
        " " + entity,
        "\n" + entity,
        entity,
    ]

def build_scoring_examples_for_condition(cond):
    """
    Each example is one prompt + candidate variant.

    For each record:
      clean candidate C variants
      conflict candidate E variants

    Later we max over variants for each answer type.
    """

    examples = []

    for rec in records:
        idx = rec["idx"]
        prompt = rec["prompts"][cond]

        clean_answer = rec["answer_clean"]
        conflict_answer = rec["answer_conflict"]

        for answer_type, entity in [
            ("clean", clean_answer),
            ("conflict", conflict_answer),
        ]:
            for variant_id, continuation in enumerate(candidate_variants(entity)):
                full_text = prompt + continuation

                prompt_ids = tokenize_no_special(prompt)
                full_ids = tokenize_no_special(full_text)

                # Usually full_ids starts with prompt_ids.
                # If tokenizer boundary merging breaks prefix, fall back to explicit concatenation.
                if len(full_ids) < len(prompt_ids) or full_ids[:len(prompt_ids)] != prompt_ids:
                    cand_ids = tokenize_no_special(continuation)
                    full_ids = prompt_ids + cand_ids
                    prompt_len = len(prompt_ids)
                else:
                    prompt_len = len(prompt_ids)

                target_ids = full_ids[prompt_len:]

                if len(target_ids) == 0:
                    continue

                # Standard causal scoring:
                # target token j is predicted at position prompt_len + j - 1.
                pred_positions = list(range(prompt_len - 1, prompt_len - 1 + len(target_ids)))

                if pred_positions[-1] >= len(full_ids):
                    raise RuntimeError("Prediction positions exceed sequence length.")

                if len(full_ids) > MAX_LEN:
                    # Should not happen with this dataset, but keep safe.
                    full_ids = full_ids[-MAX_LEN:]

                    # If truncated, skip because prompt_len no longer valid.
                    continue

                examples.append({
                    "idx": idx,
                    "condition": cond,
                    "answer_type": answer_type,
                    "entity": entity,
                    "variant_id": variant_id,
                    "continuation": continuation,
                    "input_ids": full_ids,
                    "prompt_len": prompt_len,
                    "target_ids": target_ids,
                    "pred_positions": pred_positions,
                })

    return examples

# ============================================================
# BATCHING HELPERS
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
    """
    hidden: [M, D] or [B, T, D]
    Returns logits over vocab.
    """
    return model.lm_head(hidden)

# ============================================================
# LAYERWISE SEQUENCE SCORING
# ============================================================

def score_examples_layerwise(examples, condition_name):
    """
    Returns:
      variant_scores[(idx, answer_type, variant_id, layer)] = seq_logprob
      final_logits are included as layer num_layers-1 because hidden_states[1:] are block outputs.
    """

    print(f"\nScoring condition={condition_name} ...")
    print("Total scoring examples:", len(examples))

    # scores[layer][idx][answer_type] -> list of variant scores
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

            hidden_states = outputs.hidden_states[1:]  # layer outputs only

            for l in range(num_layers):
                hs = hidden_states[l]  # [B, T, D]

                for bi, ex in enumerate(batch_examples):
                    pred_positions = ex["pred_positions"]
                    target_ids = ex["target_ids"]

                    h_sel = hs[bi, pred_positions, :]  # [T_candidate, D]
                    logits = get_lm_logits_from_hidden(h_sel)  # [T_candidate, vocab]
                    log_probs = F.log_softmax(logits.float(), dim=-1)

                    target = torch.tensor(target_ids, dtype=torch.long, device=log_probs.device)
                    token_logps = log_probs[torch.arange(len(target), device=log_probs.device), target]

                    seq_logp = float(token_logps.sum().detach().cpu())

                    scores[l][ex["idx"]][ex["answer_type"]].append(seq_logp)

            if (start // BATCH_SIZE) % 20 == 0:
                print(f"  scored {min(start + BATCH_SIZE, len(examples))}/{len(examples)} examples")

            del outputs, hidden_states, input_ids, attention_mask
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Collapse variants by max score.
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

                # Because generation output is padded batch, slice by actual prompt length.
                gen_ids = out[bi, prompt_lens[bi]:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

                clean_answer = records[idx]["answer_clean"]
                conflict_answer = records[idx]["answer_conflict"]

                has_clean = clean_answer in gen_text
                has_conflict = conflict_answer in gen_text

                # First non-empty token-ish text for quick inspection.
                row = {
                    "condition": cond,
                    "epsilon": epsilon_by_condition[cond],
                    "status": status_by_condition[cond],
                    "idx": idx,
                    "generated": gen_text,
                    "clean_answer": clean_answer,
                    "conflict_answer": conflict_answer,
                    "has_clean": bool(has_clean),
                    "has_conflict": bool(has_conflict),
                }

                rows.append(row)

            if (start // GEN_BATCH_SIZE) % 10 == 0:
                print(f"  generated {min(start + GEN_BATCH_SIZE, len(prompts))}/{len(prompts)}")

            del inputs, out
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    clean_rate = np.mean([r["has_clean"] for r in rows])
    conflict_rate = np.mean([r["has_conflict"] for r in rows])

    print(f"Generation summary {cond}: clean_rate={clean_rate:.4f}, conflict_rate={conflict_rate:.4f}")

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
        gen_clean_rate = float(np.mean([r["has_clean"] for r in gen_rows]))
        gen_conflict_rate = float(np.mean([r["has_conflict"] for r in gen_rows]))
        gen_both_rate = float(np.mean([r["has_clean"] and r["has_conflict"] for r in gen_rows]))
        gen_neither_rate = float(np.mean([(not r["has_clean"]) and (not r["has_conflict"]) for r in gen_rows]))
    else:
        gen_clean_rate = np.nan
        gen_conflict_rate = np.nan
        gen_both_rate = np.nan
        gen_neither_rate = np.nan

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
        "gen_both_rate": gen_both_rate,
        "gen_neither_rate": gen_neither_rate,
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

    print("\nCondition sequence-level summary:")
    print(f"  boundary seq margin C-E   : {summary['boundary_seq_margin']:+.4f}")
    print(f"  boundary C win rate       : {summary['boundary_clean_win_rate']:.4f}")
    print(f"  boundary E win rate       : {summary['boundary_conflict_win_rate']:.4f}")
    print(f"  late seq margin C-E       : {summary['late_seq_margin']:+.4f}")
    print(f"  final seq margin C-E      : {summary['final_seq_margin']:+.4f}")
    print(f"  generation clean rate     : {summary['gen_clean_rate']:.4f}")
    print(f"  generation conflict rate  : {summary['gen_conflict_rate']:.4f}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# FINAL TABLE
# ============================================================

print("\n\n============================================================")
print("Audit-6C.2 Final Sequence-level Breakdown Curve")
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
    )

# ============================================================
# SAVE OUTPUTS
# ============================================================

try:
    import pandas as pd

    df_layers = pd.DataFrame(all_layer_rows)
    layer_path = os.path.join(SAVE_DIR, "audit6c2_layerwise_sequence_scores.csv")
    df_layers.to_csv(layer_path, index=False)

    df_summary = pd.DataFrame(condition_summaries)
    summary_path = os.path.join(SAVE_DIR, "audit6c2_sequence_breakdown_curve.csv")
    df_summary.to_csv(summary_path, index=False)

    if len(all_generation_rows) > 0:
        df_gen = pd.DataFrame(all_generation_rows)
        gen_path = os.path.join(SAVE_DIR, "audit6c2_generation_check.csv")
        df_gen.to_csv(gen_path, index=False)
    else:
        gen_path = None

    prompt_path = os.path.join(SAVE_DIR, "audit6c2_prompts.txt")
    with open(prompt_path, "w", encoding="utf-8") as fp:
        for i, rec in enumerate(records[:10]):
            fp.write(f"===== graph {i} =====\n")
            fp.write(f"clean answer: {rec['answer_clean']}\n")
            fp.write(f"conflict answer: {rec['answer_conflict']}\n")
            for cond in conditions:
                fp.write(f"\n--- {cond} | eps={epsilon_by_condition[cond]} | {status_by_condition[cond]} ---\n")
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
print("Audit-6C.2 Interpretation Guide")
print("============================================================\n")

print("Main expected pattern:")
print("  As epsilon increases:")
print("    boundary sequence margin C-E should decrease.")
print("    conflict answer E should win more often.")
print("    generation conflict rate may increase under stronger conflicts.")
print()

print("If 6C.2 confirms 6C.1:")
print("  relation / closure breaking affects full answer sequence probability,")
print("  not only first-token answer competition.")
print()

print("If generation still outputs C even when sequence margin favors E:")
print("  internal answer competition is a precursor,")
print("  but decoding / final layer dynamics still need separate analysis.")
print()

print("If closure_negation_conflict gives strongest E advantage:")
print("  closure preservation is a stronger invariant candidate than single relation identity.")
print()

print("If sequence margins do not flip but 6C.1 first-token margins did:")
print("  multi-token entity scoring is dominated by shared suffix tokens such as '000'.")
print("  Next step: use single-token answer labels or answer options like Alpha/Beta/Gamma.")
print()

print("Done.")