#!/usr/bin/env python
"""Held-out Qwen candidate-boundary control audit.

The script reuses the frozen public local-transition executor but changes the train-only policy
objective from DeltaU displacement to paired candidate-boundary displacement.
It never imports held-out outcomes into action or gate selection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


EXECUTOR_SOURCE = Path(__file__).resolve().parent / "local_transition_executor.py"
DEFAULT_OUT = Path(__file__).resolve().parent / "output"

FULL_ACTIONS = (
    (0.0, 0.0),
    (0.3, 0.3),
    (0.6, 0.3),
    (0.6, 0.6),
    (0.9, 0.3),
    (0.9, 0.6),
    (0.9, 0.9),
    (1.0, 0.6),
    (1.0, 0.9),
    (1.0, 1.2),
)


def load_executor(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("local_transition_frozen", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def action_label(action):
    return f"a{action[0]:.1f}_b{action[1]:.1f}"


def refit_delta_u_on_train(mod, base_df, train_idx, cfg, out_dir):
    """Remove the executor's transductive PCA fit and freeze the coordinate on train only."""
    columns = [f"dR_L{layer}" for layer in cfg.decision_layers]
    train_matrix = base_df.iloc[train_idx][columns].to_numpy(np.float32)
    all_matrix = base_df[columns].to_numpy(np.float32)
    pca = mod.PCA(n_components=1)
    pca.fit(train_matrix)
    delta_u = pca.transform(all_matrix)[:, 0]
    train_mechanism = base_df.iloc[train_idx]["mechanism"].to_numpy()
    train_delta = delta_u[train_idx]
    if np.mean(train_delta[train_mechanism == "stable"]) < np.mean(train_delta[train_mechanism == "closure"]):
        pca.components_[0] *= -1
        delta_u *= -1
    out = base_df.copy()
    out["DeltaU"] = delta_u.astype(np.float32)
    out.to_csv(out_dir / "baseline_features_trainfit_coordinate.csv", index=False)
    audit = {
        "fit_scope": "training graph groups only",
        "n_fit_prompts": int(len(train_idx)),
        "explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "decision_columns": columns,
        "heldout_used_for_pca_fit": False,
    }
    with open(out_dir / "deltaU_trainfit_audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print(f"[PCA-TRAIN] explained variance {audit['explained_variance_ratio']:.9f}", flush=True)
    return out, pca


def uniform_actions(df: pd.DataFrame, layers, action, label=None):
    label = label or action_label(action)
    rows = []
    for pid in df["prompt_id"].astype(str):
        for layer in layers:
            rows.append(
                {
                    "prompt_id": pid,
                    "layer": int(layer),
                    "alpha": float(action[0]),
                    "beta": float(action[1]),
                    "policy_conf": 1.0,
                    "action_label": label,
                }
            )
    return pd.DataFrame(rows)


def norm_tracing_hook_factory(mod):
    def make_factory(
        cfg,
        stable_ops,
        precursor_dirs,
        control_mode,
        W_device,
        action_lookup=None,
        trace_buffer=None,
        clean_ids_batch=None,
        conflict_ids_batch=None,
        prompt_ids_batch=None,
    ):
        pre_tensors = {
            int(layer): torch.tensor(vec, dtype=torch.float32, device=W_device.device)
            for layer, vec in precursor_dirs.items()
            if vec is not None
        }

        def hook_factory(layer):
            def hook(module, inputs, output):
                if control_mode == "none":
                    return output
                if isinstance(output, tuple):
                    h_out, rest = output[0], output[1:]
                else:
                    h_out, rest = output, None

                batch, seq, _ = h_out.shape
                pos = torch.full((batch,), seq - 1, dtype=torch.long, device=h_out.device)
                idx = torch.arange(batch, device=h_out.device)
                h_in_last = inputs[0][idx, pos, :]
                h_real = h_out[idx, pos, :]
                h_new = h_real
                alpha = torch.zeros((batch, 1), device=h_out.device, dtype=h_out.dtype)
                beta = torch.zeros((batch, 1), device=h_out.device, dtype=h_out.dtype)
                confs, labels = [], []

                if control_mode == "operator_only_fixed":
                    alpha[:] = cfg.fixed_alpha
                elif control_mode == "precursor_only_fixed":
                    beta[:] = cfg.fixed_beta
                elif control_mode == "same_combo_fixed":
                    alpha[:], beta[:] = cfg.fixed_alpha, cfg.fixed_beta
                elif (
                    control_mode.startswith("policy_")
                    or control_mode.startswith("shuffle_")
                    or control_mode == "real_predmech_policy"
                ):
                    for bi, pid in enumerate(prompt_ids_batch):
                        a, b, conf, label = action_lookup.get(
                            (str(pid), int(layer)),
                            (0.0, 0.0, 0.0, "missing_zero"),
                        )
                        alpha[bi, 0], beta[bi, 0] = a, b
                        confs.append(conf)
                        labels.append(label)
                elif control_mode == "answer":
                    beta[:] = cfg.fixed_beta
                else:
                    raise ValueError(control_mode)

                if layer in stable_ops and torch.max(alpha).item() != 0.0:
                    target = stable_ops[layer]["stable"].predict_torch(h_in_last.float())
                    target = target.to(h_real.dtype)
                    h_new = (1.0 - alpha) * h_new + alpha * target

                if layer in pre_tensors and torch.max(beta).item() != 0.0:
                    direction = pre_tensors[layer].to(h_new.device, h_new.dtype)
                    scale = h_new.float().std(dim=-1, keepdim=True).to(h_new.dtype)
                    h_new = h_new + beta * scale * direction[None, :].expand(batch, -1)

                base_norm = torch.norm(h_real.float(), dim=1).clamp_min(1e-8)
                new_norm = torch.norm(h_new.float(), dim=1)
                delta_ratio = torch.norm((h_new - h_real).float(), dim=1) / base_norm
                norm_ratio = new_norm / base_norm
                finite = torch.isfinite(h_new).all(dim=1)

                if trace_buffer is not None:
                    trace_buffer.append(
                        {
                            "control": control_mode,
                            "layer": int(layer),
                            "mean_alpha": float(alpha.float().mean().cpu()),
                            "mean_beta": float(beta.float().mean().cpu()),
                            "mean_policy_conf": float(np.mean(confs)) if confs else np.nan,
                            "most_common_action": max(set(labels), key=labels.count) if labels else "",
                            "batch_size": int(batch),
                            "mean_norm_ratio": float(norm_ratio.mean().cpu()),
                            "min_norm_ratio": float(norm_ratio.min().cpu()),
                            "max_norm_ratio": float(norm_ratio.max().cpu()),
                            "mean_delta_norm_ratio": float(delta_ratio.mean().cpu()),
                            "all_finite": bool(finite.all().cpu()),
                        }
                    )

                if torch.allclose(h_new, h_real):
                    return output
                h2 = h_out.clone()
                h2[idx, pos, :] = h_new
                return (h2,) + rest if rest is not None else h2

            return hook

        return hook_factory

    return make_factory


def run_grid(
    mod,
    model,
    tokenizer,
    train_df,
    cfg,
    pca,
    anchors,
    stable_ops,
    precursor_dirs,
    actions,
    out_dir,
):
    grid_dir = out_dir / "train_grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, action in enumerate(actions, 1):
        label = action_label(action)
        path = grid_dir / f"{label}.csv"
        if path.exists():
            cached = pd.read_csv(path)
            if len(cached) == len(train_df) and set(cached["prompt_id"].astype(str)) == set(train_df["prompt_id"].astype(str)):
                print(f"[GRID {i}/{len(actions)}] reuse {label}", flush=True)
                frames.append(cached)
                continue
        print(f"[GRID {i}/{len(actions)}] run {label}", flush=True)
        action_df = uniform_actions(train_df, cfg.operator_layers, action, label)
        result, _ = mod.forward_eval(
            model,
            tokenizer,
            train_df,
            cfg,
            pca,
            anchors,
            stable_ops,
            precursor_dirs,
            "policy_grid_boundary",
            action_df,
        )
        result["action_label"] = label
        result["alpha"], result["beta"] = action
        result["final_margin"] = result["final_clean_logit"] - result["final_conflict_logit"]
        result.to_csv(path, index=False)
        frames.append(result)
    grid = pd.concat(frames, ignore_index=True)
    grid.to_csv(out_dir / "train_action_grid.csv", index=False)
    return grid


def derive_train_labels(grid, base_train, out_dir):
    base = base_train[["prompt_id", "mechanism", "R_final_margin"]].rename(
        columns={"R_final_margin": "baseline_margin"}
    )
    merged = grid.merge(base, on=["prompt_id", "mechanism"], how="left")
    merged["margin_shift"] = merged["final_margin"] - merged["baseline_margin"]
    merged["dose"] = merged["alpha"] ** 2 + merged["beta"] ** 2
    merged["crossed"] = (merged["baseline_margin"] < 0) & (merged["final_margin"] >= 0)

    labels = []
    for pid, group in merged.groupby("prompt_id", sort=False):
        mechanism = str(group.iloc[0]["mechanism"])
        g = group.copy()
        if mechanism == "closure":
            g["utility"] = (
                g["margin_shift"]
                + 5.0 * g["crossed"].astype(float)
                - 0.03 * g["dose"]
            )
        else:
            g["utility"] = -g["margin_shift"].abs() - 0.10 * g["dose"]
        best = g.sort_values(["utility", "dose"], ascending=[False, True]).iloc[0]
        labels.append(
            {
                "prompt_id": pid,
                "mechanism": mechanism,
                "best_action_label": best["action_label"],
                "best_alpha": float(best["alpha"]),
                "best_beta": float(best["beta"]),
                "best_final_margin": float(best["final_margin"]),
                "best_margin_shift": float(best["margin_shift"]),
                "best_crossed": bool(best["crossed"]),
                "best_utility": float(best["utility"]),
            }
        )
    label_df = pd.DataFrame(labels)
    merged.to_csv(out_dir / "train_action_grid_merged.csv", index=False)
    label_df.to_csv(out_dir / "train_boundary_policy_labels.csv", index=False)
    return merged, label_df


def select_global_action(merged, actions, out_dir):
    rows = []
    for action in actions:
        label = action_label(action)
        group = merged[merged["action_label"] == label]
        closure = group[group["mechanism"] == "closure"]
        other = group[group["mechanism"] != "closure"]
        flips = (other["pair_choice"] != np.where(other["baseline_margin"] >= 0, "clean", "conflict"))
        rows.append(
            {
                "action_label": label,
                "alpha": action[0],
                "beta": action[1],
                "closure_crossings": int(closure["crossed"].sum()),
                "closure_crossing_rate": float(closure["crossed"].mean()),
                "closure_mean_margin_shift": float(closure["margin_shift"].mean()),
                "nonclosure_flip_rate": float(flips.mean()),
                "nonclosure_mean_abs_margin_shift": float(other["margin_shift"].abs().mean()),
                "dose": action[0] ** 2 + action[1] ** 2,
            }
        )
    audit = pd.DataFrame(rows)
    valid = audit[audit["nonclosure_flip_rate"] <= 0.05]
    pool = valid if len(valid) else audit
    best = pool.sort_values(
        ["closure_crossings", "closure_mean_margin_shift", "dose"],
        ascending=[False, False, True],
    ).iloc[0]
    audit["selected"] = audit["action_label"] == best["action_label"]
    audit.to_csv(out_dir / "train_global_action_selection.csv", index=False)
    return (float(best["alpha"]), float(best["beta"]))


def apply_confidence_gate(actions, predicted_features):
    probs = predicted_features[["p_mech_stable", "p_mech_competition", "p_mech_closure"]].to_numpy(float)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1) / math.log(3.0)
    gate_source = predicted_features[["prompt_id", "p_mech_closure"]].copy()
    gate_source["entropy"] = entropy
    prompt_gate = gate_source.groupby("prompt_id", as_index=False).agg(
        mean_p_closure=("p_mech_closure", "mean"),
        mean_entropy=("entropy", "mean"),
    )
    prompt_gate["gate_open"] = (
        (prompt_gate["mean_p_closure"] >= 0.50)
        & (prompt_gate["mean_entropy"] <= 0.90)
    )
    out = actions.merge(prompt_gate, on="prompt_id", how="left")
    closed = ~out["gate_open"].fillna(False)
    out.loc[closed, ["alpha", "beta"]] = 0.0
    out.loc[closed, "action_label"] = "a0.0_b0.0_confidence_gate"
    return out


def evaluate_control(
    mod,
    model,
    tokenizer,
    test_df,
    cfg,
    pca,
    anchors,
    stable_ops,
    precursor_dirs,
    control,
    actions,
    out_dir,
):
    path = out_dir / f"test_{control}.csv"
    trace_path = out_dir / f"trace_{control}.csv"
    if path.exists() and len(pd.read_csv(path)) == len(test_df):
        print(f"[TEST] reuse {control}", flush=True)
        trace = pd.DataFrame()
        if trace_path.exists() and trace_path.stat().st_size > 1:
            try:
                trace = pd.read_csv(trace_path)
            except pd.errors.EmptyDataError:
                trace = pd.DataFrame()
        return pd.read_csv(path), trace
    print(f"[TEST] run {control}", flush=True)
    result, trace = mod.forward_eval(
        model,
        tokenizer,
        test_df,
        cfg,
        pca,
        anchors,
        stable_ops,
        precursor_dirs,
        control,
        actions,
    )
    result["control"] = control
    result["final_margin"] = result["final_clean_logit"] - result["final_conflict_logit"]
    result.to_csv(path, index=False)
    trace.to_csv(trace_path, index=False)
    return result, trace


def evaluate_vocab_top1(
    mod,
    model,
    tokenizer,
    test_df,
    cfg,
    stable_ops,
    precursor_dirs,
    control,
    actions,
    out_dir,
):
    """Replay the intervention against the full vocabulary, not only the pair."""
    path = out_dir / f"vocab_top1_{control}.csv"
    if path.exists() and len(pd.read_csv(path)) == len(test_df):
        print(f"[TOP1] reuse {control}", flush=True)
        return pd.read_csv(path)

    print(f"[TOP1] run {control}", flush=True)
    W_device = mod.get_lm_head_weight(model).detach().float().to(model.device)
    action_lookup = mod.make_action_lookup(actions)
    layer_modules = mod.get_layers(model)
    hook_layers = sorted(set(cfg.operator_layers) | set(cfg.precursor_layers))
    rows = []
    texts = test_df["text"].tolist()
    clean_ids_all = test_df["clean_token_id"].to_numpy(np.int64)
    conflict_ids_all = test_df["conflict_token_id"].to_numpy(np.int64)
    prompt_ids_all = test_df["prompt_id"].astype(str).tolist()

    with torch.no_grad():
        for start in range(0, len(test_df), cfg.batch_size):
            end = min(len(test_df), start + cfg.batch_size)
            batch = test_df.iloc[start:end].reset_index(drop=True)
            inputs = tokenizer(
                texts[start:end],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_len,
            ).to(model.device)
            clean_ids = torch.tensor(clean_ids_all[start:end], dtype=torch.long, device=model.device)
            conflict_ids = torch.tensor(conflict_ids_all[start:end], dtype=torch.long, device=model.device)
            prompt_ids = prompt_ids_all[start:end]
            handles = []
            if control != "none":
                factory = mod.make_combo_hook_factory(
                    cfg=cfg,
                    stable_ops=stable_ops,
                    precursor_dirs=precursor_dirs,
                    control_mode=control,
                    W_device=W_device,
                    action_lookup=action_lookup,
                    trace_buffer=None,
                    clean_ids_batch=clean_ids,
                    conflict_ids_batch=conflict_ids,
                    prompt_ids_batch=prompt_ids,
                )
                for layer in hook_layers:
                    if layer < len(layer_modules):
                        handles.append(layer_modules[layer].register_forward_hook(factory(layer)))
            outputs = model(**inputs, use_cache=False)
            for handle in handles:
                handle.remove()
            pos = mod.last_positions(inputs["attention_mask"])
            idx = torch.arange(end - start, device=model.device)
            logits = outputs.logits[idx, pos, :].detach().float()
            top_ids = logits.argmax(dim=1)
            clean_logits = logits[idx, clean_ids]
            conflict_logits = logits[idx, conflict_ids]
            clean_ranks = (logits > clean_logits[:, None]).sum(dim=1) + 1
            conflict_ranks = (logits > conflict_logits[:, None]).sum(dim=1) + 1

            for bi, (_, row) in enumerate(batch.iterrows()):
                top_id = int(top_ids[bi].cpu())
                clean_id = int(clean_ids[bi].cpu())
                conflict_id = int(conflict_ids[bi].cpu())
                if top_id == clean_id:
                    top_class = "clean"
                elif top_id == conflict_id:
                    top_class = "conflict"
                else:
                    top_class = "other"
                rows.append(
                    {
                        "prompt_id": row["prompt_id"],
                        "graph_id": int(row["graph_id"]),
                        "condition": row["condition"],
                        "mechanism": row["mechanism"],
                        "control": control,
                        "clean_label": row["clean_label"],
                        "conflict_label": row["conflict_label"],
                        "clean_token_id": clean_id,
                        "conflict_token_id": conflict_id,
                        "top1_token_id": top_id,
                        "top1_token": tokenizer.decode([top_id]),
                        "top1_class": top_class,
                        "clean_rank": int(clean_ranks[bi].cpu()),
                        "conflict_rank": int(conflict_ranks[bi].cpu()),
                        "clean_minus_conflict_margin": float((clean_logits[bi] - conflict_logits[bi]).cpu()),
                    }
                )
            del outputs, logits, inputs
    result = pd.DataFrame(rows)
    result.to_csv(path, index=False)
    return result


def summarize_vocab_top1(top1_results, out_dir):
    baseline = top1_results["none"][["prompt_id", "top1_class", "top1_token_id"]].rename(
        columns={"top1_class": "baseline_top1_class", "top1_token_id": "baseline_top1_token_id"}
    )
    rows, merged_frames = [], []
    for control, result in top1_results.items():
        merged = result.merge(baseline, on="prompt_id", how="left")
        merged["strict_conflict_to_clean"] = (
            (merged["baseline_top1_class"] == "conflict") & (merged["top1_class"] == "clean")
        )
        merged["any_to_clean"] = (
            (merged["baseline_top1_class"] != "clean") & (merged["top1_class"] == "clean")
        )
        merged["top1_changed"] = merged["top1_token_id"] != merged["baseline_top1_token_id"]
        merged_frames.append(merged)
        closure = merged[merged["mechanism"] == "closure"]
        other = merged[merged["mechanism"] != "closure"]
        rows.append(
            {
                "control": control,
                "closure_n": len(closure),
                "closure_top1_clean": int((closure["top1_class"] == "clean").sum()),
                "closure_top1_conflict": int((closure["top1_class"] == "conflict").sum()),
                "closure_top1_other": int((closure["top1_class"] == "other").sum()),
                "closure_strict_conflict_to_clean": int(closure["strict_conflict_to_clean"].sum()),
                "closure_any_to_clean": int(closure["any_to_clean"].sum()),
                "nonclosure_top1_change_rate": float(other["top1_changed"].mean()),
                "nonclosure_top1_clean_loss_rate": float(
                    ((other["baseline_top1_class"] == "clean") & (other["top1_class"] != "clean")).mean()
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "vocab_top1_summary.csv", index=False)
    pd.concat(merged_frames, ignore_index=True).to_csv(out_dir / "vocab_top1_rows.csv", index=False)
    return summary


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def summarize(results, traces, base_test, action_tables, out_dir):
    base = base_test[["prompt_id", "mechanism", "DeltaU", "R_final_margin", "pair_choice"]].rename(
        columns={
            "DeltaU": "baseline_DeltaU",
            "R_final_margin": "baseline_margin",
            "pair_choice": "baseline_pair_choice",
        }
    )
    rows, merged_frames = [], []
    for control, result in results.items():
        merged = result.merge(base, on=["prompt_id", "mechanism"], how="left")
        merged["margin_shift"] = merged["final_margin"] - merged["baseline_margin"]
        merged["DeltaU_shift"] = merged["DeltaU"] - merged["baseline_DeltaU"]
        merged["pair_flip"] = merged["pair_choice"] != merged["baseline_pair_choice"]
        merged["crossed"] = (merged["baseline_margin"] < 0) & (merged["final_margin"] >= 0)
        merged["control"] = control
        merged_frames.append(merged)

        closure = merged[merged["mechanism"] == "closure"]
        other = merged[merged["mechanism"] != "closure"]
        k, n = int(closure["crossed"].sum()), len(closure)
        lo, hi = wilson(k, n)
        trace = traces.get(control, pd.DataFrame())
        if len(trace) and "mean_norm_ratio" in trace:
            weights = trace["batch_size"].to_numpy(float)
            mean_norm = float(np.average(trace["mean_norm_ratio"], weights=weights))
            max_norm = float(trace["max_norm_ratio"].max())
            all_finite = bool(trace["all_finite"].astype(str).str.lower().eq("true").all())
        else:
            mean_norm, max_norm, all_finite = 1.0, 1.0, True
        action_table = action_tables.get(control)
        nonzero_rate = float(((action_table["alpha"] != 0) | (action_table["beta"] != 0)).mean()) if action_table is not None else np.nan
        max_action_rate = float(((action_table["alpha"] == 1.0) & (action_table["beta"] == 1.2)).mean()) if action_table is not None else np.nan
        rows.append(
            {
                "control": control,
                "closure_n": n,
                "closure_crossings": k,
                "closure_crossing_rate": k / n if n else np.nan,
                "closure_crossing_ci95_low": lo,
                "closure_crossing_ci95_high": hi,
                "closure_mean_margin_shift": float(closure["margin_shift"].mean()),
                "closure_median_margin_shift": float(closure["margin_shift"].median()),
                "closure_max_final_margin": float(closure["final_margin"].max()),
                "closure_mean_DeltaU_shift": float(closure["DeltaU_shift"].mean()),
                "nonclosure_pair_flip_rate": float(other["pair_flip"].mean()),
                "nonclosure_mean_abs_margin_shift": float(other["margin_shift"].abs().mean()),
                "margin_specificity": float(closure["margin_shift"].mean() - other["margin_shift"].abs().mean()),
                "mean_norm_ratio": mean_norm,
                "max_norm_ratio": max_norm,
                "all_finite": all_finite,
                "nonzero_action_rate": nonzero_rate,
                "max_action_rate": max_action_rate,
            }
        )
    summary = pd.DataFrame(rows)
    merged_all = pd.concat(merged_frames, ignore_index=True)
    summary.to_csv(out_dir / "heldout_control_summary.csv", index=False)
    merged_all.to_csv(out_dir / "heldout_control_rows.csv", index=False)
    return summary, merged_all


def verdict(summary):
    primary = summary.set_index("control").loc["policy_boundary"]
    null = summary.set_index("control").loc["shuffle_boundary_labels"]
    fixed = summary.set_index("control").loc["policy_fixed_global"]
    safe = (
        primary["nonclosure_pair_flip_rate"] <= 0.05
        and bool(primary["all_finite"])
        and 0.67 <= primary["mean_norm_ratio"] <= 1.50
    )
    null_better = primary["closure_crossings"] > null["closure_crossings"]
    if primary["closure_crossing_rate"] >= 0.10 and primary["closure_median_margin_shift"] > 0 and safe and null_better:
        label = "PASS_STRONG_POLICY_BOUNDARY_CONTROL"
    elif primary["closure_crossings"] > 0 and safe and null_better:
        label = "PASS_PARTIAL_POLICY_BOUNDARY_CONTROL"
    elif fixed["closure_crossings"] > 0 and fixed["nonclosure_pair_flip_rate"] <= 0.05:
        label = "PASS_ACTUATOR_LEVERAGE_FIXED_ONLY"
    elif not safe:
        label = "UNSAFE_AT_EXPANDED_BUDGET"
    else:
        label = "BOUNDED_NEGATIVE_NO_CROSSING"
    return {
        "verdict": label,
        "primary_control": "policy_boundary",
        "primary_crossings": int(primary["closure_crossings"]),
        "primary_closure_n": int(primary["closure_n"]),
        "primary_crossing_rate": float(primary["closure_crossing_rate"]),
        "primary_mean_margin_shift": float(primary["closure_mean_margin_shift"]),
        "primary_nonclosure_flip_rate": float(primary["nonclosure_pair_flip_rate"]),
        "primary_mean_norm_ratio": float(primary["mean_norm_ratio"]),
        "shuffle_crossings": int(null["closure_crossings"]),
        "fixed_global_crossings": int(fixed["closure_crossings"]),
        "claim_boundary": (
            "Paired candidate-boundary control on a controlled relation-graph task only; "
            "not open-ended generation control, answer-correctness improvement, deployment control, "
            "a universal dynamical law, or a hidden-ontology claim."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mod = load_executor(EXECUTOR_SOURCE)
    cfg = mod.Config()
    cfg.save_dir = str(out_dir / "local_transition_runtime")
    cfg.n_graphs = 12 if args.smoke else 96
    cfg.batch_size = 4
    cfg.n_shuffles = 0
    cfg.include_answer_control = False
    actions = FULL_ACTIONS[:4] if args.smoke else FULL_ACTIONS
    cfg.policy_actions = tuple(actions)
    mod.CFG = cfg
    mod.make_combo_hook_factory = norm_tracing_hook_factory(mod)
    mod.set_seed(cfg.seed)

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "executor_source": str(EXECUTOR_SOURCE),
                "executor_config": asdict(cfg),
                "actions": actions,
                "smoke": args.smoke,
                "policy_objective": "train-only paired candidate-boundary margin",
                "confidence_gate_role": "mechanism-confidence and entropy gate only",
            },
            handle,
            indent=2,
        )

    print("[LOAD] model and tokenizer", flush=True)
    model, tokenizer = mod.load_model_and_tokenizer(cfg)
    dataset_dir = out_dir / "local_transition_runtime"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    df = mod.build_dataset(tokenizer, cfg, dataset_dir)
    base_df, hidden, _ = mod.extract_baseline(model, tokenizer, df, cfg, dataset_dir)
    train_idx, test_idx, split = mod.make_split(base_df, cfg, dataset_dir)
    split.to_csv(out_dir / "frozen_group_split.csv", index=False)
    base_df, pca = refit_delta_u_on_train(mod, base_df, train_idx, cfg, out_dir)

    clean = base_df[base_df["condition"] == "clean"].set_index("graph_id")
    anchors = {
        int(graph): {layer: float(row[f"R_L{layer}"]) for layer in cfg.decision_layers}
        for graph, row in clean.iterrows()
    }
    stable_ops = mod.fit_stable_operators(base_df, hidden, train_idx, cfg, dataset_dir)
    precursor_dirs, probes, velocity = mod.fit_precursor_and_probes(base_df, hidden, train_idx, cfg, dataset_dir)

    train_df = base_df.iloc[train_idx].reset_index(drop=True)
    test_df = base_df.iloc[test_idx].reset_index(drop=True)
    train_features = mod.base_state_features_for_samples(base_df, hidden, probes, velocity, train_idx, cfg)
    test_features = mod.base_state_features_for_samples(base_df, hidden, probes, velocity, test_idx, cfg)

    grid = run_grid(
        mod, model, tokenizer, train_df, cfg, pca, anchors,
        stable_ops, precursor_dirs, actions, out_dir,
    )
    merged_grid, labels = derive_train_labels(grid, train_df, out_dir)
    global_action = select_global_action(merged_grid, actions, out_dir)
    print(f"[TRAIN] selected global action {global_action}", flush=True)

    mech_clf, mech_audit = mod.fit_mechanism_classifier(train_features, cfg, dataset_dir)
    train_pred = mod.attach_predicted_mechanism(train_features, mech_clf)
    test_pred = mod.attach_predicted_mechanism(test_features, mech_clf)
    policy_clf, policy_audit = mod.fit_gain_policy(
        train_pred, labels, mod.POLICY_PRED_MECH_COLS, cfg,
        shuffle_policy_labels=False, shuffle_id=0,
    )
    boundary_actions = mod.policy_predict_actions(policy_clf, test_pred, mod.POLICY_PRED_MECH_COLS)
    guarded_actions = apply_confidence_gate(boundary_actions, test_pred)
    shuffle_clf, shuffle_audit = mod.fit_gain_policy(
        train_pred, labels, mod.POLICY_PRED_MECH_COLS, cfg,
        shuffle_policy_labels=True, shuffle_id=0,
    )
    shuffle_actions = mod.policy_predict_actions(shuffle_clf, test_pred, mod.POLICY_PRED_MECH_COLS)
    fixed_actions = uniform_actions(test_df, cfg.operator_layers, global_action, "training_global")

    action_tables = {
        "policy_fixed_global": fixed_actions,
        "policy_boundary": boundary_actions,
        "policy_boundary_guarded": guarded_actions,
        "shuffle_boundary_labels": shuffle_actions,
    }
    for name, table in action_tables.items():
        table.to_csv(out_dir / f"actions_{name}.csv", index=False)
    with open(out_dir / "classifier_audits.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"mechanism": mech_audit, "boundary_policy": policy_audit, "shuffle_policy": shuffle_audit},
            handle,
            indent=2,
        )

    controls = {
        "none": None,
        "same_combo_fixed": None,
        "policy_fixed_global": fixed_actions,
        "policy_boundary": boundary_actions,
        "policy_boundary_guarded": guarded_actions,
        "shuffle_boundary_labels": shuffle_actions,
    }
    results, traces = {}, {}
    for control, table in controls.items():
        result, trace = evaluate_control(
            mod, model, tokenizer, test_df, cfg, pca, anchors,
            stable_ops, precursor_dirs, control, table, out_dir,
        )
        results[control], traces[control] = result, trace

    summary, _ = summarize(results, traces, test_df, action_tables, out_dir)
    top1_controls = {
        "none": None,
        "same_combo_fixed": None,
        "policy_boundary": boundary_actions,
        "policy_boundary_guarded": guarded_actions,
        "shuffle_boundary_labels": shuffle_actions,
    }
    top1_results = {}
    for control, table in top1_controls.items():
        top1_results[control] = evaluate_vocab_top1(
            mod, model, tokenizer, test_df, cfg, stable_ops, precursor_dirs,
            control, table, out_dir,
        )
    top1_summary = summarize_vocab_top1(top1_results, out_dir)
    final = verdict(summary)
    primary_top1 = top1_summary.set_index("control").loc["policy_boundary"]
    guarded_top1 = top1_summary.set_index("control").loc["policy_boundary_guarded"]
    final.update(
        {
            "n_train_prompts": int(len(train_df)),
            "n_test_prompts": int(len(test_df)),
            "n_test_graphs": int(test_df["graph_id"].nunique()),
            "n_test_closure": int((test_df["mechanism"] == "closure").sum()),
            "selected_global_action": list(global_action),
            "primary_strict_conflict_to_clean_top1": int(primary_top1["closure_strict_conflict_to_clean"]),
            "guarded_strict_conflict_to_clean_top1": int(guarded_top1["closure_strict_conflict_to_clean"]),
            "primary_nonclosure_top1_change_rate": float(primary_top1["nonclosure_top1_change_rate"]),
            "guarded_nonclosure_top1_change_rate": float(guarded_top1["nonclosure_top1_change_rate"]),
        }
    )
    with open(out_dir / "verdict.json", "w", encoding="utf-8") as handle:
        json.dump(final, handle, indent=2)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
