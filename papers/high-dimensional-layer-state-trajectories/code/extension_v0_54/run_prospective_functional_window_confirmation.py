# -*- coding: utf-8 -*-
"""Prospective confirmation of previously selected functional windows.

The protocol in protocol_v0_1.md was frozen before this script was run. The
confirmation prompts never enter window selection or fitted preprocessing.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.environ.get("WINDOW_OUTPUT_DIR", str(HERE / "outputs" / "functional_window_confirmation")))
SOURCE_DIR = Path(os.environ.get("WINDOW_DISCOVERY_DIR", str(HERE.parents[1] / "source_data" / "extension_v0_54" / "functional_window_confirmation" / "discovery")))
BASE_SCRIPT = Path(os.environ.get("WINDOW_SHARED_SCRIPT", str(HERE / "functional_window_shared.py")))
PROTOCOL = HERE.parents[1] / "source_data" / "extension_v0_54" / "functional_window_confirmation" / "protocol.md"

MODEL_PATHS = {
    "qwen": Path(os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen2.5-1.5B-Instruct")),
    "llama": Path(os.environ.get("LLAMA_MODEL_PATH", "meta-llama/Llama-3.2-1B-Instruct")),
    "gemma": Path(os.environ.get("GEMMA_MODEL_PATH", "google/gemma-2-2b-it")),
}

SEED = 20260718
BATCH_SIZE = 4
MAX_LEN = 320
K_LIST = [100, 500]
N_BOOT = 2000
N_NULL = 999
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
FEATURE_COLS = [
    "rel_center_dist",
    "rel_jaccard_dist",
    "rel_weighted_jaccard_dist",
    "rel_composite_dist",
    "spread",
    "entropy",
]
MECH_MAP = {"stable": 0, "stable_shift": 0, "competition": 1, "closure": 2}

CONFIRM_ENTITIES = [
    ("Alden", "Bria", "Celyn"),
    ("Devon", "Elara", "Fintan"),
    ("Greer", "Halen", "Idra"),
    ("Jarek", "Kaia", "Lucan"),
    ("Maelis", "Noam", "Odel"),
    ("Priya", "Ronan", "Selene"),
]
CONFIRM_RELATIONS = [
    ("routes through", "resolves to"),
    ("connects via", "carries label"),
    ("passes through", "is coded as"),
    ("links through", "terminates at"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_base_module():
    spec = importlib.util.spec_from_file_location("suppc_base", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_confirmation_dataset(base, labels: list[str]) -> pd.DataFrame:
    if len(labels) != 12:
        raise ValueError("The frozen confirmation design requires 12 labels.")
    rows = []
    graph_count = 0
    for entity_id, (a, b, d) in enumerate(CONFIRM_ENTITIES):
        for relation_id, (r1, r2) in enumerate(CONFIRM_RELATIONS):
            clean = labels[(2 * graph_count) % len(labels)]
            conflict = labels[(2 * graph_count + 1) % len(labels)]
            aux = labels[(2 * graph_count + 2) % len(labels)]
            graph_id = 1000 + graph_count
            for condition in base.ALL_CONDS:
                rows.append(
                    {
                        "prompt_id": f"confirm_g{graph_count:03d}_{condition}",
                        "graph_id": graph_id,
                        "entity_id": entity_id,
                        "relation_id": relation_id,
                        "condition": condition,
                        "condition_class": base.COND_CLASS[condition],
                        "mechanism": base.MECHANISM[condition],
                        "clean_label": clean,
                        "conflict_label": conflict,
                        "aux_label": aux,
                        "text": base.make_prompt(condition, a, b, d, clean, conflict, aux, r1, r2),
                        "split": "confirmation",
                    }
                )
            graph_count += 1
    if graph_count != 24:
        raise AssertionError(f"Expected 24 confirmation graphs, found {graph_count}.")
    return pd.DataFrame(rows)


def audit_common_labels(base) -> tuple[list[str], pd.DataFrame]:
    rows = []
    common = []
    tokenizers = {
        key: base.AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=True
        )
        for key, path in MODEL_PATHS.items()
    }
    for label in base.LABEL_CANDIDATES:
        row = {"label": label}
        ok = True
        for key, tokenizer in tokenizers.items():
            ids = base.continuation_ids(tokenizer, label)
            row[f"{key}_ids"] = str(ids)
            row[f"{key}_len"] = len(ids)
            ok = ok and len(ids) == 1
        row["is_common_single_token"] = bool(ok)
        rows.append(row)
        if ok:
            common.append(label)
    if len(common) < 24:
        raise RuntimeError(f"Need at least 24 common single-token labels, found {len(common)}.")
    return common, pd.DataFrame(rows)


def get_lm_head_weight(model):
    if hasattr(model, "lm_head"):
        return model.lm_head.weight
    return model.get_output_embeddings().weight


def extract_combined(base, model_key: str, combined: pd.DataFrame):
    tokenizer, model = base.load_tok_model(str(MODEL_PATHS[model_key]), DEVICE, DTYPE)
    num_layers = base.get_layers(model)
    decision = base.decision_layers(num_layers, (0.70, 0.92))
    windows = base.scan_windows(num_layers, 0.45, [2, 3, 4, 5, 6, 8])
    max_init_layer = max(max(window["layers"]) for window in windows)
    needed_layers = sorted(set(range(max_init_layer + 1)).union(decision))

    output_weight = get_lm_head_weight(model).detach().float().to(model.device)
    output_weight_norm = F.normalize(output_weight, dim=1)
    clean_ids = []
    conflict_ids = []
    for row in combined.itertuples(index=False):
        clean = base.continuation_ids(tokenizer, row.clean_label)
        conflict = base.continuation_ids(tokenizer, row.conflict_label)
        if len(clean) != 1 or len(conflict) != 1:
            raise RuntimeError(f"{model_key}: candidate label is not one continuation token.")
        clean_ids.append(clean[0])
        conflict_ids.append(conflict[0])

    n_rows = len(combined)
    max_k = max(K_LIST)
    layer_data = {
        k: {"centers": {}, "ids": {}, "vals": {}, "spread": {}, "entropy": {}}
        for k in K_LIST
    }
    margins = {layer: np.zeros(n_rows, dtype=np.float32) for layer in decision}
    texts = combined["text"].tolist()

    with torch.no_grad():
        for start in range(0, n_rows, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_rows)
            inputs = tokenizer(
                texts[start:end],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
            ).to(model.device)
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            positions = torch.full(
                (end - start,),
                inputs["attention_mask"].shape[1] - 1,
                dtype=torch.long,
                device=model.device,
            )
            clean_tensor = torch.tensor(clean_ids[start:end], dtype=torch.long, device=model.device)
            conflict_tensor = torch.tensor(conflict_ids[start:end], dtype=torch.long, device=model.device)
            row_index = torch.arange(end - start, device=model.device)

            for layer in needed_layers:
                h = outputs.hidden_states[layer + 1][row_index, positions].detach().float()
                if layer in margins:
                    margins[layer][start:end] = (
                        torch.sum(h * output_weight[clean_tensor], dim=1)
                        - torch.sum(h * output_weight[conflict_tensor], dim=1)
                    ).cpu().numpy()
                if layer <= max_init_layer:
                    logits = h @ output_weight.T
                    values, ids = torch.topk(logits, k=max_k, dim=1)
                    for k in K_LIST:
                        ids_k = ids[:, :k]
                        values_k = values[:, :k].float()
                        embeddings = output_weight_norm[ids_k]
                        centre = F.normalize(embeddings.mean(dim=1), dim=1)
                        spread = (1.0 - torch.sum(embeddings * centre[:, None, :], dim=2)).mean(dim=1)
                        probabilities = torch.softmax(values_k, dim=1)
                        entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum(dim=1) / math.log(k)
                        if layer not in layer_data[k]["centers"]:
                            width = centre.shape[1]
                            layer_data[k]["centers"][layer] = np.zeros((n_rows, width), dtype=np.float32)
                            layer_data[k]["ids"][layer] = np.zeros((n_rows, k), dtype=np.int32)
                            layer_data[k]["vals"][layer] = np.zeros((n_rows, k), dtype=np.float32)
                            layer_data[k]["spread"][layer] = np.zeros(n_rows, dtype=np.float32)
                            layer_data[k]["entropy"][layer] = np.zeros(n_rows, dtype=np.float32)
                        layer_data[k]["centers"][layer][start:end] = centre.cpu().numpy()
                        layer_data[k]["ids"][layer][start:end] = ids_k.cpu().numpy().astype(np.int32)
                        layer_data[k]["vals"][layer][start:end] = values_k.cpu().numpy()
                        layer_data[k]["spread"][layer][start:end] = spread.cpu().numpy()
                        layer_data[k]["entropy"][layer][start:end] = entropy.cpu().numpy()
            print(f"[{model_key}] extracted {end}/{n_rows}", flush=True)
            del outputs, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    model_df = combined.copy().reset_index(drop=True)
    clean_index = {
        int(row.graph_id): i
        for i, row in model_df.iterrows()
        if row["condition"] == "clean"
    }
    d_cols = []
    for layer in decision:
        raw_col = f"R_L{layer}"
        delta_col = f"dR_L{layer}"
        model_df[raw_col] = margins[layer]
        model_df[delta_col] = [
            margins[layer][i] - margins[layer][clean_index[int(row.graph_id)]]
            for i, row in model_df.iterrows()
        ]
        d_cols.append(delta_col)

    discovery_mask = model_df["split"].eq("discovery").to_numpy()
    pca = PCA(n_components=min(3, len(d_cols)))
    pca.fit(model_df.loc[discovery_mask, d_cols].to_numpy(dtype=np.float32))
    delta_u = pca.transform(model_df[d_cols].to_numpy(dtype=np.float32))[:, 0]
    discovery_mechanism = model_df.loc[discovery_mask, "mechanism"].to_numpy()
    if np.mean(delta_u[discovery_mask][np.isin(discovery_mechanism, ["stable", "stable_shift"])]) < np.mean(
        delta_u[discovery_mask][discovery_mechanism == "closure"]
    ):
        delta_u = -delta_u
    model_df["DeltaU"] = delta_u.astype(np.float32)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model_df, layer_data, num_layers, decision, windows, pca.explained_variance_ratio_


def fit_models_and_score(feature_df: pd.DataFrame):
    discovery = feature_df[feature_df["split"] == "discovery"].copy()
    confirmation = feature_df[feature_df["split"] == "confirmation"].copy()

    train_class = discovery[discovery["condition"] != "clean"]
    test_class = confirmation[confirmation["condition"] != "clean"]
    class_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )
    class_model.fit(
        train_class[FEATURE_COLS].to_numpy(dtype=np.float32),
        train_class["mechanism"].map(MECH_MAP).to_numpy(dtype=int),
    )
    class_prediction = class_model.predict(test_class[FEATURE_COLS].to_numpy(dtype=np.float32))
    class_target = test_class["mechanism"].map(MECH_MAP).to_numpy(dtype=int)

    downstream_model = Pipeline(
        [("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))]
    )
    downstream_model.fit(
        discovery[FEATURE_COLS].to_numpy(dtype=np.float32),
        discovery["DeltaU"].to_numpy(dtype=float),
    )
    downstream_prediction = downstream_model.predict(
        confirmation[FEATURE_COLS].to_numpy(dtype=np.float32)
    )
    downstream_target = confirmation["DeltaU"].to_numpy(dtype=float)

    preserve = confirmation[confirmation["condition"].isin(BASE.REL_PRESERVE_CONDS)]
    structure = confirmation[confirmation["condition"].isin(BASE.STRUCTURE_CHANGE_CONDS)]
    preserve_mean = float(preserve["rel_composite_dist"].mean())
    structure_mean = float(structure["rel_composite_dist"].mean())
    topology_score = 1.0 - preserve_mean
    macro_f1 = float(f1_score(class_target, class_prediction, average="macro", zero_division=0))
    class_accuracy = float(accuracy_score(class_target, class_prediction))
    downstream_corr = safe_corr(downstream_target, downstream_prediction)
    downstream_r2 = float(r2_score(downstream_target, downstream_prediction))
    composite = 0.5 * topology_score + 0.8 * max(macro_f1, 0) + 0.8 * max(downstream_corr, 0)

    return {
        "topology_score": topology_score,
        "preserve_distance_mean": preserve_mean,
        "structure_distance_mean": structure_mean,
        "topology_separation": structure_mean - preserve_mean,
        "mechanism_accuracy": class_accuracy,
        "mechanism_macro_f1": macro_f1,
        "downstream_corr": downstream_corr,
        "downstream_r2": downstream_r2,
        "composite_score": composite,
    }, {
        "discovery": discovery,
        "confirmation": confirmation,
        "train_class": train_class,
        "test_class": test_class,
        "class_target": class_target,
        "class_prediction": class_prediction,
        "downstream_target": downstream_target,
        "downstream_prediction": downstream_prediction,
    }


def safe_corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_intervals(parts, rng: np.random.Generator):
    confirmation = parts["confirmation"].reset_index(drop=True)
    test_class = parts["test_class"].reset_index(drop=True)
    graph_ids = np.array(sorted(confirmation["graph_id"].unique()))
    topology_by_graph = {}
    for graph_id, graph in confirmation.groupby("graph_id"):
        preserve = graph[graph["condition"].isin(BASE.REL_PRESERVE_CONDS)]["rel_composite_dist"].mean()
        structure = graph[graph["condition"].isin(BASE.STRUCTURE_CHANGE_CONDS)]["rel_composite_dist"].mean()
        topology_by_graph[int(graph_id)] = float(structure - preserve)

    class_graph = test_class["graph_id"].to_numpy(dtype=int)
    downstream_graph = confirmation["graph_id"].to_numpy(dtype=int)
    topology_values = []
    f1_values = []
    correlation_values = []
    for _ in range(N_BOOT):
        sampled = rng.choice(graph_ids, size=len(graph_ids), replace=True)
        topology_values.append(float(np.mean([topology_by_graph[int(g)] for g in sampled])))
        class_indices = np.concatenate([np.flatnonzero(class_graph == g) for g in sampled])
        downstream_indices = np.concatenate([np.flatnonzero(downstream_graph == g) for g in sampled])
        f1_values.append(
            float(
                f1_score(
                    parts["class_target"][class_indices],
                    parts["class_prediction"][class_indices],
                    average="macro",
                    zero_division=0,
                )
            )
        )
        correlation_values.append(
            safe_corr(
                parts["downstream_target"][downstream_indices],
                parts["downstream_prediction"][downstream_indices],
            )
        )
    return {
        "topology_separation_ci95": np.quantile(topology_values, [0.025, 0.975]).tolist(),
        "mechanism_macro_f1_ci95": np.quantile(f1_values, [0.025, 0.975]).tolist(),
        "downstream_corr_ci95": np.quantile(correlation_values, [0.025, 0.975]).tolist(),
    }


def permutation_nulls(parts, rng: np.random.Generator):
    train_class = parts["train_class"]
    test_class = parts["test_class"]
    x_train_class = train_class[FEATURE_COLS].to_numpy(dtype=np.float32)
    x_test_class = test_class[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train_class = train_class["mechanism"].map(MECH_MAP).to_numpy(dtype=int)
    y_test_class = parts["class_target"]
    class_null = []
    for _ in range(N_NULL):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        )
        model.fit(x_train_class, rng.permutation(y_train_class))
        class_null.append(
            float(
                f1_score(
                    y_test_class,
                    model.predict(x_test_class),
                    average="macro",
                    zero_division=0,
                )
            )
        )

    discovery = parts["discovery"].reset_index(drop=True)
    confirmation = parts["confirmation"].reset_index(drop=True)
    x_train = discovery[FEATURE_COLS].to_numpy(dtype=np.float32)
    x_test = confirmation[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = discovery["DeltaU"].to_numpy(dtype=float)
    y_test = parts["downstream_target"]
    graph_values = discovery["graph_id"].to_numpy(dtype=int)
    graph_indices = [np.flatnonzero(graph_values == graph_id) for graph_id in np.unique(graph_values)]
    downstream_null = []
    for _ in range(N_NULL):
        permuted = y_train.copy()
        for indices in graph_indices:
            permuted[indices] = rng.permutation(permuted[indices])
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        model.fit(x_train, permuted)
        downstream_null.append(safe_corr(y_test, model.predict(x_test)))

    return {
        "mechanism_null": class_null,
        "downstream_null": downstream_null,
    }


def empirical_p(null_values, observed):
    null = np.asarray(null_values, dtype=float)
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))


def selected_spec(summary: pd.DataFrame, model_key: str):
    row = summary.loc[summary["model_key"] == model_key].iloc[0]
    return {
        "window": str(row["overall_best_window"]),
        "layers": json.loads(str(row["overall_best_layers"])),
        "k": int(row["overall_best_k"]),
    }


def main():
    global BASE
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BASE = load_base_module()
    BASE.set_seed(SEED)
    rng = np.random.default_rng(SEED)

    common_labels, token_audit = audit_common_labels(BASE)
    discovery_labels = common_labels[:12]
    confirmation_labels = common_labels[12:24]
    if set(discovery_labels).intersection(confirmation_labels):
        raise AssertionError("Discovery and confirmation labels must be disjoint.")
    token_audit.to_csv(OUTPUT_DIR / "common_label_tokenization_audit.csv", index=False, encoding="utf-8-sig")

    discovery = pd.read_csv(SOURCE_DIR / "discovery_prompt_dataset.csv")
    discovery["split"] = "discovery"
    confirmation = build_confirmation_dataset(BASE, confirmation_labels)
    combined = pd.concat([discovery, confirmation], ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "prospective_prompt_dataset.csv", index=False, encoding="utf-8-sig")

    if discovery["graph_id"].nunique() != 36 or confirmation["graph_id"].nunique() != 24:
        raise AssertionError("Unexpected discovery or confirmation graph count.")
    if set(discovery["graph_id"]).intersection(set(confirmation["graph_id"])):
        raise AssertionError("Graph IDs overlap across discovery and confirmation.")
    if set(discovery["clean_label"]).intersection(set(confirmation["clean_label"])):
        raise AssertionError("Candidate label vocabularies overlap across splits.")

    summary = pd.read_csv(SOURCE_DIR / "discovery_model_summary.csv")
    model_results = []
    audit_rows = []
    for model_index, model_key in enumerate(["qwen", "llama", "gemma"]):
        print(f"\n===== {model_key.upper()} =====", flush=True)
        model_df, layer_data, num_layers, decision, windows, pca_variance = extract_combined(
            BASE, model_key, combined
        )
        model_df.to_csv(OUTPUT_DIR / f"{model_key}_prompt_table.csv", index=False, encoding="utf-8-sig")

        original_prompt = pd.read_csv(SOURCE_DIR / f"{model_key}_prompt_table.csv")
        original_delta = original_prompt["DeltaU"].to_numpy(dtype=float)
        reproduced_delta = model_df.loc[model_df["split"] == "discovery", "DeltaU"].to_numpy(dtype=float)
        pca_reproduction_corr = safe_corr(original_delta, reproduced_delta)

        frozen = selected_spec(summary, model_key)
        scan_rows = []
        selected_parts = None
        selected_features = None
        for window in windows:
            for k in K_LIST:
                features, _ = BASE.compute_feature_df(model_df, layer_data, k, window["layers"])
                features["split"] = model_df["split"].to_numpy()
                metrics, parts = fit_models_and_score(features)
                row = {
                    "model_key": model_key,
                    "window": window["window"],
                    "layers": json.dumps(window["layers"]),
                    "k": k,
                    **metrics,
                }
                scan_rows.append(row)
                if window["window"] == frozen["window"] and k == frozen["k"]:
                    selected_parts = parts
                    selected_features = features.copy()

        scan = pd.DataFrame(scan_rows).sort_values("composite_score", ascending=False).reset_index(drop=True)
        scan["rank"] = np.arange(1, len(scan) + 1)
        scan["rank_percentile"] = 100.0 * (len(scan) - scan["rank"] + 1) / len(scan)
        scan.to_csv(OUTPUT_DIR / f"{model_key}_prospective_window_scan.csv", index=False, encoding="utf-8-sig")
        if selected_parts is None or selected_features is None:
            raise AssertionError(f"Frozen selection not found for {model_key}.")
        selected_row = scan[(scan["window"] == frozen["window"]) & (scan["k"] == frozen["k"])].iloc[0]
        selected_features.to_csv(
            OUTPUT_DIR / f"{model_key}_frozen_window_features.csv", index=False, encoding="utf-8-sig"
        )

        model_rng = np.random.default_rng(SEED + 1000 * (model_index + 1))
        intervals = bootstrap_intervals(selected_parts, model_rng)
        nulls = permutation_nulls(selected_parts, model_rng)
        pd.DataFrame({"mechanism_macro_f1_null": nulls["mechanism_null"]}).to_csv(
            OUTPUT_DIR / f"{model_key}_mechanism_label_null.csv", index=False
        )
        pd.DataFrame({"downstream_corr_null": nulls["downstream_null"]}).to_csv(
            OUTPUT_DIR / f"{model_key}_downstream_target_null.csv", index=False
        )

        mechanism_null_95 = float(np.quantile(nulls["mechanism_null"], 0.95))
        downstream_null_95 = float(np.quantile(nulls["downstream_null"], 0.95))
        role_recurrence = bool(
            intervals["topology_separation_ci95"][0] > 0
            and selected_row["mechanism_macro_f1"] >= 0.55
            and selected_row["mechanism_macro_f1"] > mechanism_null_95
            and selected_row["downstream_corr"] > 0
            and selected_row["downstream_corr"] > downstream_null_95
        )
        location_recurrence = bool(selected_row["rank_percentile"] >= 75.0)
        result = {
            "model_key": model_key,
            "num_layers": int(num_layers),
            "decision_layers": decision,
            "frozen_window": frozen["window"],
            "frozen_layers": frozen["layers"],
            "frozen_k": frozen["k"],
            "pca_discovery_reproduction_corr": pca_reproduction_corr,
            "pca_explained_variance_ratio_discovery": pca_variance.tolist(),
            "topology_separation": float(selected_row["topology_separation"]),
            "topology_separation_ci95": intervals["topology_separation_ci95"],
            "mechanism_macro_f1": float(selected_row["mechanism_macro_f1"]),
            "mechanism_macro_f1_ci95": intervals["mechanism_macro_f1_ci95"],
            "mechanism_null_95": mechanism_null_95,
            "mechanism_empirical_p": empirical_p(nulls["mechanism_null"], selected_row["mechanism_macro_f1"]),
            "downstream_corr": float(selected_row["downstream_corr"]),
            "downstream_corr_ci95": intervals["downstream_corr_ci95"],
            "downstream_r2": float(selected_row["downstream_r2"]),
            "downstream_null_95": downstream_null_95,
            "downstream_empirical_p": empirical_p(nulls["downstream_null"], selected_row["downstream_corr"]),
            "composite_score": float(selected_row["composite_score"]),
            "candidate_count": int(len(scan)),
            "frozen_rank": int(selected_row["rank"]),
            "frozen_rank_percentile": float(selected_row["rank_percentile"]),
            "role_recurrence": role_recurrence,
            "window_location_recurrence": location_recurrence,
        }
        model_results.append(result)
        audit_rows.append(
            {
                "model_key": model_key,
                "pca_reproduction_corr": pca_reproduction_corr,
                "discovery_rows": int((model_df["split"] == "discovery").sum()),
                "confirmation_rows": int((model_df["split"] == "confirmation").sum()),
            }
        )

        del layer_data, model_df, selected_parts, selected_features
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    role_count = sum(result["role_recurrence"] for result in model_results)
    location_count = sum(result["window_location_recurrence"] for result in model_results)
    if role_count == 3:
        family_verdict = "three-checkpoint role recurrence"
    elif role_count == 2:
        family_verdict = "partial role recurrence (two of three checkpoints)"
    else:
        family_verdict = "failed three-checkpoint role recurrence"

    final = {
        "protocol_version": "v0.1",
        "seed": SEED,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "discovery_graphs": 36,
        "confirmation_graphs": 24,
        "discovery_labels": discovery_labels,
        "confirmation_labels": confirmation_labels,
        "n_bootstrap": N_BOOT,
        "n_null": N_NULL,
        "role_recurrence_count": int(role_count),
        "window_location_recurrence_count": int(location_count),
        "family_verdict": family_verdict,
        "models": model_results,
        "input_hashes": {
            "protocol_sha256": sha256(PROTOCOL),
            "discovery_prompt_dataset_sha256": sha256(SOURCE_DIR / "discovery_prompt_dataset.csv"),
            "model_summary_sha256": sha256(SOURCE_DIR / "discovery_model_summary.csv"),
            "base_script_sha256": sha256(BASE_SCRIPT),
        },
    }
    with (OUTPUT_DIR / "functional_window_confirmation_summary_v0_1.json").open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    pd.DataFrame(model_results).to_csv(
        OUTPUT_DIR / "functional_window_confirmation_summary_v0_1.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(audit_rows).to_csv(OUTPUT_DIR / "extraction_reproduction_audit.csv", index=False)

    lines = [
        "# Functional-window prospective confirmation v0.1",
        "",
        f"Verdict: **{family_verdict}**.",
        "",
        "| Checkpoint | Topology separation (95% CI) | Macro F1 | Label-null 95% | Downstream r | Target-null 95% | Frozen rank | Role | Location |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for result in model_results:
        lines.append(
            "| {model_key} | {topology_separation:.3f} [{t0:.3f}, {t1:.3f}] | "
            "{mechanism_macro_f1:.3f} | {mechanism_null_95:.3f} | {downstream_corr:.3f} | "
            "{downstream_null_95:.3f} | {frozen_rank}/{candidate_count} ({frozen_rank_percentile:.1f}%) | "
            "{role_recurrence} | {window_location_recurrence} |".format(
                t0=result["topology_separation_ci95"][0],
                t1=result["topology_separation_ci95"][1],
                **result,
            )
        )
    lines.extend(
        [
            "",
            "The confirmation set was not used for PCA fitting, feature scaling, classifier fitting, ridge fitting or window selection.",
            "A positive result is restricted to new graph instantiations within the same controlled task family.",
        ]
    )
    (OUTPUT_DIR / "functional_window_confirmation_summary_v0_1.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
