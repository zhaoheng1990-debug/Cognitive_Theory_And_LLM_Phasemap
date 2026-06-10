# ============================================================
# PhaseMap-8B.1
# True Path Score Sampler for PhaseMap-5A / 6B Dataset
#
# Purpose:
#   Extract real per-layer path/basin scores from the original
#   phasemap5a_dataset.csv prompts.
#
# Inputs:
#   phasemap5a_dataset.csv
#
# Outputs:
#   phasemap8b1_outputs/
#     phasemap8b_prompts.csv
#     phasemap8b_path_scores_answer_binary.csv
#     phasemap8b_path_scores_mechanism_probe.csv
#     phasemap8b_path_scores.csv
#     phasemap8b1_summary.json
#
# IMPORTANT INTERPRETATION:
#   The original PhaseMap-5A task is binary: answer exactly Red or Blue.
#   Therefore the only truly comparable answer-basin paths are:
#
#       clean_answer     = logit(C_id)
#       conflict_answer  = logit(E_id)
#
#   The mechanism-probe paths below are real logit probes, but they are
#   not answer options. They should be treated as exploratory geometry
#   probes, not as decisive multi-answer geodesics.
#
# Recommended strict 8B usage:
#   For conservative strict audit, copy:
#
#       phasemap8b_path_scores_answer_binary.csv
#
#   to:
#
#       phasemap8b_path_scores.csv
#
#   then run:
#
#       phasemap8b_strict_multipath_geodesic_audit.py
#
# For exploratory multi-probe audit:
#   copy phasemap8b_path_scores_mechanism_probe.csv to
#   phasemap8b_path_scores.csv.
# ============================================================

import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")

# -----------------------------
# CONFIG
# -----------------------------

MODEL_PATH = r"D:\model\models--Qwen--Qwen2.5-1.5B-Instruct\main"
DATASET_PATH = Path(r"C:\Windows\System32\phasemap5a_outputs\phasemap5a_dataset.csv")

OUT_DIR = Path("./phasemap8b1_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MAX_LEN = 256
BATCH_SIZE = 4

# Qwen2.5-1.5B convention in earlier experiments:
# hidden_states[1] -> layer 0 output
TRACK_LAYERS = list(range(7, 26))  # match 6B/8A windows, L7-L25

# Mechanism probes are real logit probes at the answer position.
# They are NOT answer classes; use only as exploratory path geometry.
MECHANISM_PATH_SPECS = {
    "clean_closure_probe": [
        "Red", "correct", "valid", "primary", "rule", "closure"
    ],
    "conflict_override_probe": [
        "Blue", "conflict", "correction", "direct", "override", "claim"
    ],
    "update_rule_probe": [
        "updated", "now", "new", "changed", "revised"
    ],
    "exception_rule_probe": [
        "exception", "special", "except", "particular"
    ],
    "ambiguity_probe": [
        "ambiguous", "uncertain", "both", "depends", "unknown"
    ],
}

# A second mechanism file can be made label-aware:
# clean probes include actual C_label; conflict probes include actual E_label.
MAKE_LABEL_AWARE_MECHANISM = True

# -----------------------------
# HELPERS
# -----------------------------

def encode_probe_ids(tokenizer, terms):
    ids = []
    for t in terms:
        text = str(t)
        # Try raw and leading-space variants.
        for v in [text, " " + text, "\n" + text]:
            toks = tokenizer.encode(v, add_special_tokens=False)
            if toks:
                # Last token is a simple next-token lexical probe.
                ids.append(int(toks[-1]))
    return sorted(set(ids))

def token_id_from_dataset_value(x):
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None

@torch.no_grad()
def forward_logits_by_layer(model, tokenizer, prompts):
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    ).to(model.device)

    out = model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
    )

    attn = enc["attention_mask"]
    last_pos = attn.sum(dim=1) - 1

    hs = out.hidden_states
    result = {}

    for l in TRACK_LAYERS:
        h = hs[l + 1]  # layer l output
        logits_list = []
        for i, p in enumerate(last_pos):
            vec = h[i, p]
            logits = model.lm_head(vec).detach().float().cpu().numpy()
            logits_list.append(logits)
        result[l] = logits_list

    return result

def mean_logit(logits, ids):
    ids = [i for i in ids if i is not None and 0 <= int(i) < len(logits)]
    if not ids:
        return np.nan
    vals = logits[np.asarray(ids, dtype=int)]
    return float(np.mean(vals))

def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATASET_PATH}. Put this script beside phasemap5a_dataset.csv."
        )

    df = pd.read_csv(DATASET_PATH)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    required = ["prompt", "graph_id", "condition", "phase", "C_id", "E_id", "C_label", "E_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in dataset: {missing}")

    # sample_id matches 6B-full: global row index.
    df = df.reset_index(drop=True)
    df["sample_id"] = np.arange(len(df), dtype=int)

    prompt_cols = ["sample_id", "graph_id", "condition", "phase", "prompt", "C_id", "E_id", "C_label", "E_label"]
    if "A" in df.columns:
        prompt_cols.append("A")
    if "B" in df.columns:
        prompt_cols.append("B")

    prompts_df = df[prompt_cols].copy()
    prompts_df.to_csv(OUT_DIR / "phasemap8b_prompts.csv", index=False, encoding="utf-8")

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

    # Pre-tokenize mechanism probes.
    mechanism_ids = {
        name: encode_probe_ids(tokenizer, terms)
        for name, terms in MECHANISM_PATH_SPECS.items()
    }

    print("Mechanism probe token ids:")
    for k, v in mechanism_ids.items():
        print(k, v)

    answer_rows = []
    mechanism_rows = []
    label_aware_rows = []

    print("Extracting path scores...")

    for start in range(0, len(df), BATCH_SIZE):
        sub = df.iloc[start:start + BATCH_SIZE].reset_index(drop=True)
        Hlogits = forward_logits_by_layer(model, tokenizer, sub["prompt"].tolist())

        for i in range(len(sub)):
            row = sub.iloc[i]
            sample_id = int(row["sample_id"])
            graph_id = int(row["graph_id"])
            condition = row["condition"]
            phase = row["phase"]

            c_id = token_id_from_dataset_value(row["C_id"])
            e_id = token_id_from_dataset_value(row["E_id"])
            c_label = str(row["C_label"])
            e_label = str(row["E_label"])

            # Label-aware mechanism probes: make clean/conflict label tokens explicit.
            label_aware_specs = dict(MECHANISM_PATH_SPECS)
            label_aware_specs["clean_answer_plus_closure_probe"] = [c_label, "correct", "primary", "rule", "closure"]
            label_aware_specs["conflict_answer_plus_override_probe"] = [e_label, "conflict", "correction", "override", "exception"]
            label_aware_ids = {
                name: encode_probe_ids(tokenizer, terms)
                for name, terms in label_aware_specs.items()
            }

            for l in TRACK_LAYERS:
                logits = Hlogits[l][i]

                # Strict answer-basin paths.
                answer_rows.append({
                    "sample_id": sample_id,
                    "graph_id": graph_id,
                    "condition": condition,
                    "phase": phase,
                    "layer": int(l),
                    "path_name": "clean_answer",
                    "path_score": float(logits[c_id]),
                    "path_type": "answer_binary",
                    "token_ids": str([c_id]),
                    "label": c_label,
                })
                answer_rows.append({
                    "sample_id": sample_id,
                    "graph_id": graph_id,
                    "condition": condition,
                    "phase": phase,
                    "layer": int(l),
                    "path_name": "conflict_answer",
                    "path_score": float(logits[e_id]),
                    "path_type": "answer_binary",
                    "token_ids": str([e_id]),
                    "label": e_label,
                })

                # Global mechanism probes.
                for path_name, ids in mechanism_ids.items():
                    mechanism_rows.append({
                        "sample_id": sample_id,
                        "graph_id": graph_id,
                        "condition": condition,
                        "phase": phase,
                        "layer": int(l),
                        "path_name": path_name,
                        "path_score": mean_logit(logits, ids),
                        "path_type": "mechanism_probe",
                        "token_ids": str(ids),
                        "label": "",
                    })

                # Label-aware mechanism probes.
                if MAKE_LABEL_AWARE_MECHANISM:
                    for path_name, ids in label_aware_ids.items():
                        label_aware_rows.append({
                            "sample_id": sample_id,
                            "graph_id": graph_id,
                            "condition": condition,
                            "phase": phase,
                            "layer": int(l),
                            "path_name": path_name,
                            "path_score": mean_logit(logits, ids),
                            "path_type": "label_aware_mechanism_probe",
                            "token_ids": str(ids),
                            "label": "",
                        })

        del Hlogits
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (start + len(sub)) % 32 == 0:
            print(f"Processed {start + len(sub)}/{len(df)}")

    answer_df = pd.DataFrame(answer_rows)
    mech_df = pd.DataFrame(mechanism_rows)
    label_mech_df = pd.DataFrame(label_aware_rows)

    # Conservative strict 8B input.
    answer_path = OUT_DIR / "phasemap8b_path_scores_answer_binary.csv"
    answer_df.to_csv(answer_path, index=False, encoding="utf-8")

    # Exploratory multi-probe input.
    mech_path = OUT_DIR / "phasemap8b_path_scores_mechanism_probe.csv"
    mech_df.to_csv(mech_path, index=False, encoding="utf-8")

    label_mech_path = OUT_DIR / "phasemap8b_path_scores_label_aware_mechanism_probe.csv"
    label_mech_df.to_csv(label_mech_path, index=False, encoding="utf-8")

    # Default phasemap8b_path_scores.csv:
    # Use answer-binary by default because it is the only strictly comparable answer path score.
    default_path = OUT_DIR / "phasemap8b_path_scores.csv"
    answer_df[["sample_id", "layer", "path_name", "path_score"]].to_csv(
        default_path,
        index=False,
        encoding="utf-8",
    )

    # Combined exploratory file.
    combined = pd.concat([answer_df, mech_df, label_mech_df], ignore_index=True)
    combined_path = OUT_DIR / "phasemap8b_path_scores_all_exploratory.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8")

    summary = {
        "dataset_path": str(DATASET_PATH),
        "n_samples": int(len(df)),
        "track_layers": TRACK_LAYERS,
        "n_answer_rows": int(len(answer_df)),
        "n_mechanism_rows": int(len(mech_df)),
        "n_label_aware_mechanism_rows": int(len(label_mech_df)),
        "outputs": {
            "prompts": str(OUT_DIR / "phasemap8b_prompts.csv"),
            "answer_binary": str(answer_path),
            "mechanism_probe": str(mech_path),
            "label_aware_mechanism_probe": str(label_mech_path),
            "default_strict_input": str(default_path),
            "all_exploratory": str(combined_path),
        },
        "mechanism_probe_token_ids": mechanism_ids,
        "interpretation_warning": (
            "answer_binary is the conservative comparable path score set. "
            "mechanism_probe and label_aware_mechanism_probe are real logit probes but not answer classes."
        ),
    }

    with open(OUT_DIR / "phasemap8b1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
