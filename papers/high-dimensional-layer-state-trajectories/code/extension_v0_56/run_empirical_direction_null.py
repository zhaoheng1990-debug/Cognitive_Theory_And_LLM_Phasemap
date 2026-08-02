"""Norm-matched empirical-subspace random-direction null for the v0.52 actuator."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
PAPER_ROOT = HERE.parents[1]
EXECUTOR_SOURCE = CODE_ROOT / "extension_v0_52" / "local_transition_executor.py"
CONTROLLER_SOURCE = CODE_ROOT / "extension_v0_52" / "boundary_controller.py"
ATTRIBUTION_SOURCE = CODE_ROOT / "extension_v0_54" / "run_actuator_component_attribution.py"
BOUNDARY_SOURCE = PAPER_ROOT / "source_data" / "source_data" / "extension_v0_52" / "output_boundary_control"

MODEL_WINDOWS = {
    "qwen": {
        "operator_layers": (15, 16, 17, 18, 19),
        "precursor_layers": (17, 18, 19),
        "decision_layers": (20, 21, 22, 23, 24, 25),
        "path_env": "QWEN_MODEL_PATH",
    },
    "llama": {
        "operator_layers": (8, 9, 10, 11),
        "precursor_layers": (10, 11, 12),
        "decision_layers": (12, 13, 14, 15),
        "path_env": "LLAMA_MODEL_PATH",
    },
    "gemma": {
        "operator_layers": (13, 14, 15, 16, 17),
        "precursor_layers": (15, 16, 17),
        "decision_layers": (18, 19, 20, 21, 22, 23),
        "path_env": "GEMMA_MODEL_PATH",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_WINDOWS), default=list(MODEL_WINDOWS))
    parser.add_argument("--null-seeds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026080302)
    parser.add_argument("--subspace-rank", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def action_lookup(actions: pd.DataFrame) -> dict[tuple[str, int], tuple[float, float, float, str]]:
    result = {}
    for _, row in actions.iterrows():
        result[(str(row["prompt_id"]), int(row["layer"]))] = (
            float(row["alpha"]),
            float(row["beta"]),
            float(row.get("policy_conf", 0.0)),
            str(row.get("action_label", "")),
        )
    return result


def empirical_bases(hidden: dict[int, np.ndarray], train_index: np.ndarray, cfg, rank: int, seed: int) -> dict[tuple[str, int], np.ndarray]:
    bases = {}
    rng = np.random.default_rng(seed)
    layers = sorted(set(cfg.operator_layers) | set(cfg.precursor_layers))
    for layer in layers:
        updates = hidden[layer + 1][train_index].astype(np.float32) - hidden[layer][train_index].astype(np.float32)
        components = min(rank, updates.shape[0] - 1, updates.shape[1])
        if components < 2:
            raise RuntimeError(f"Insufficient empirical rank at layer {layer}")
        pca = PCA(n_components=components, svd_solver="full")
        pca.fit(updates)
        for component_name in ("operator", "precursor"):
            vectors = []
            for _ in range(8):
                coefficients = rng.normal(size=components).astype(np.float32)
                vector = pca.components_.T @ coefficients
                norm = float(np.linalg.norm(vector))
                if norm <= 1e-8:
                    raise RuntimeError("Degenerate empirical direction")
                vectors.append((vector / norm).astype(np.float32))
            bases[(component_name, layer)] = np.stack(vectors, axis=0)
    return bases


def orthogonal_random_vectors(base_candidates: torch.Tensor, exclusions: torch.Tensor) -> torch.Tensor:
    vectors = []
    for exclusion in exclusions:
        exclusion_norm_sq = torch.sum(exclusion.float() * exclusion.float()).clamp_min(1e-12)
        chosen = None
        for base in base_candidates:
            candidate = base.to(exclusion.device, exclusion.dtype)
            candidate = candidate - torch.sum(candidate.float() * exclusion.float()) / exclusion_norm_sq * exclusion
            norm = torch.norm(candidate.float())
            if float(norm) > 1e-6:
                chosen = candidate / norm.to(candidate.dtype)
                break
        if chosen is None:
            raise RuntimeError("Empirical subspace could not provide an orthogonal direction")
        vectors.append(chosen)
    return torch.stack(vectors, dim=0)


def random_direction_hook_factory(executor, bases: dict[tuple[str, int], np.ndarray]):
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
        if not control_mode.startswith("random_direction_"):
            raise ValueError(control_mode)
        real_directions = {
            int(layer): torch.tensor(vector, dtype=torch.float32, device=W_device.device)
            for layer, vector in precursor_dirs.items()
            if vector is not None
        }
        base_tensors = {
            key: torch.tensor(value, dtype=torch.float32, device=W_device.device)
            for key, value in bases.items()
        }

        def hook_factory(layer):
            def hook(module, inputs, output):
                if isinstance(output, tuple):
                    h_out, rest = output[0], output[1:]
                else:
                    h_out, rest = output, None
                batch, seq, _ = h_out.shape
                idx = torch.arange(batch, device=h_out.device)
                pos = torch.full((batch,), seq - 1, dtype=torch.long, device=h_out.device)
                h_input = inputs[0][idx, pos, :]
                h_real = h_out[idx, pos, :]
                alpha = torch.zeros((batch, 1), dtype=h_real.dtype, device=h_real.device)
                beta = torch.zeros((batch, 1), dtype=h_real.dtype, device=h_real.device)
                labels = []
                confidences = []
                for index, prompt_id in enumerate(prompt_ids_batch):
                    a, b, confidence, label = action_lookup.get((str(prompt_id), int(layer)), (0.0, 0.0, 0.0, "missing_zero"))
                    alpha[index, 0], beta[index, 0] = a, b
                    labels.append(label)
                    confidences.append(confidence)

                h_random = h_real
                h_structured = h_real
                op_structured_norm = torch.zeros(batch, device=h_real.device)
                pre_structured_norm = torch.zeros(batch, device=h_real.device)
                if layer in stable_ops and torch.max(alpha).item() != 0.0:
                    target = stable_ops[layer]["stable"].predict_torch(h_input.float()).to(h_real.dtype)
                    structured_delta = alpha * (target - h_real)
                    op_structured_norm = torch.norm(structured_delta.float(), dim=1)
                    random_unit = orthogonal_random_vectors(base_tensors[("operator", layer)], structured_delta)
                    h_random = h_random + op_structured_norm[:, None].to(h_real.dtype) * random_unit
                    h_structured = h_structured + structured_delta

                if layer in real_directions and torch.max(beta).item() != 0.0:
                    direction = real_directions[layer].to(h_real.device, h_real.dtype)
                    structured_scale = h_structured.float().std(dim=-1, keepdim=True).to(h_real.dtype)
                    structured_delta = beta * structured_scale * direction[None, :].expand(batch, -1)
                    pre_structured_norm = torch.norm(structured_delta.float(), dim=1)
                    exclusions = direction[None, :].expand(batch, -1)
                    random_unit = orthogonal_random_vectors(base_tensors[("precursor", layer)], exclusions)
                    h_random = h_random + pre_structured_norm[:, None].to(h_real.dtype) * random_unit
                    h_structured = h_structured + structured_delta

                base_norm = torch.norm(h_real.float(), dim=1).clamp_min(1e-8)
                structured_total = torch.norm((h_structured - h_real).float(), dim=1)
                random_total = torch.norm((h_random - h_real).float(), dim=1)
                if trace_buffer is not None:
                    trace_buffer.append(
                        {
                            "control": control_mode,
                            "layer": int(layer),
                            "batch_size": int(batch),
                            "mean_alpha": float(alpha.float().mean().cpu()),
                            "mean_beta": float(beta.float().mean().cpu()),
                            "mean_policy_conf": float(np.mean(confidences)),
                            "most_common_action": max(set(labels), key=labels.count),
                            "mean_operator_component_norm": float(op_structured_norm.mean().cpu()),
                            "mean_random_operator_component_norm": float(op_structured_norm.mean().cpu()),
                            "mean_precursor_component_norm": float(pre_structured_norm.mean().cpu()),
                            "mean_random_precursor_component_norm": float(pre_structured_norm.mean().cpu()),
                            "operator_component_norm_abs_error": 0.0,
                            "precursor_component_norm_abs_error": 0.0,
                            "mean_structured_total_ratio": float((structured_total / base_norm).mean().cpu()),
                            "mean_random_total_ratio": float((random_total / base_norm).mean().cpu()),
                            "mean_total_norm_ratio": float((torch.norm(h_random.float(), dim=1) / base_norm).mean().cpu()),
                            "all_finite": bool(torch.isfinite(h_random).all().cpu()),
                        }
                    )
                if torch.allclose(h_random, h_real):
                    return output
                replacement = h_out.clone()
                replacement[idx, pos, :] = h_random
                return (replacement,) + rest if rest is not None else replacement
            return hook
        return hook_factory
    return make_factory


def classify_outcomes(result: pd.DataFrame, baseline: pd.DataFrame, model_key: str, control: str, seed: int | None) -> pd.DataFrame:
    base = baseline[["prompt_id", "top1_token_id", "top1_class", "clean_minus_conflict_margin"]].rename(
        columns={"top1_token_id": "baseline_top1_token_id", "top1_class": "baseline_top1_class", "clean_minus_conflict_margin": "baseline_margin"}
    )
    out = result.merge(base, on="prompt_id", validate="one_to_one")
    out["strict_conflict_to_clean"] = out["baseline_top1_class"].eq("conflict") & out["top1_class"].eq("clean")
    out["nonclosure_top1_changed"] = ~out["top1_token_id"].eq(out["baseline_top1_token_id"])
    out["margin_shift"] = out["clean_minus_conflict_margin"] - out["baseline_margin"]
    out["model"] = model_key
    out["control"] = control
    out["random_seed"] = seed
    return out


def evaluate_or_load(attribution, executor, model, tokenizer, test, cfg, stable_ops, precursor_dirs, control, actions, out_dir):
    output_path = out_dir / f"vocab_top1_{control}.csv"
    trace_path = out_dir / f"trace_{control}.csv"
    if output_path.exists():
        result = pd.read_csv(output_path)
        if len(result) != len(test):
            raise RuntimeError(f"{control}: cached output has an unexpected row count")
        if trace_path.exists() and trace_path.stat().st_size > 3:
            trace = pd.read_csv(trace_path)
        else:
            trace = pd.DataFrame()
        print(f"[{cfg.model_key}] reuse {control}", flush=True)
        return result, trace
    return attribution.evaluate_vocab_top1_with_trace(
        executor, model, tokenizer, test, cfg, stable_ops, precursor_dirs, control, actions, out_dir
    )


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for (model_key, control, seed), frame in rows.groupby(["model", "control", "random_seed"], dropna=False, sort=True):
        closure = frame[frame["mechanism"].eq("closure")]
        nonclosure = frame[~frame["mechanism"].eq("closure")]
        crossings = int(closure["strict_conflict_to_clean"].sum())
        changes = int(nonclosure["nonclosure_top1_changed"].sum())
        summary_rows.append(
            {
                "model": model_key,
                "control": control,
                "random_seed": seed,
                "closure_n": int(len(closure)),
                "strict_conflict_to_clean": crossings,
                "closure_crossing_rate": float(crossings / len(closure)) if len(closure) else np.nan,
                "nonclosure_n": int(len(nonclosure)),
                "nonclosure_top1_changes": changes,
                "nonclosure_change_rate": float(changes / len(nonclosure)) if len(nonclosure) else np.nan,
                "specificity_count": int(crossings - changes),
                "specificity_rate": float(crossings / len(closure) - changes / len(nonclosure)) if len(closure) and len(nonclosure) else np.nan,
                "closure_mean_margin_shift": float(closure["margin_shift"].mean()),
            }
        )
    return pd.DataFrame(summary_rows)


def run_model(model_key: str, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    out_dir = args.output_root / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = BOUNDARY_SOURCE / model_key
    controller = load_module(f"controller_v056_{model_key}", CONTROLLER_SOURCE)
    executor = controller.load_executor(EXECUTOR_SOURCE)
    attribution = load_module(f"attribution_v056_{model_key}", ATTRIBUTION_SOURCE)
    cfg = executor.Config()
    cfg.model_key = model_key
    cfg.model_path = os.environ.get(MODEL_WINDOWS[model_key]["path_env"])
    if not cfg.model_path:
        raise RuntimeError(
            f"Set {MODEL_WINDOWS[model_key]['path_env']} to the local checkpoint path before running {model_key}."
        )
    cfg.save_dir = str(out_dir / "runtime")
    cfg.n_graphs = 96
    cfg.batch_size = 4
    cfg.n_shuffles = 0
    cfg.include_answer_control = False
    for field in ("operator_layers", "precursor_layers", "decision_layers"):
        setattr(cfg, field, MODEL_WINDOWS[model_key][field])
    executor.CFG = cfg
    executor.set_seed(cfg.seed)
    runtime = out_dir / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol": "v0.56 PROTOCOL.md",
        "model": model_key,
        "config": asdict(cfg),
        "executor_sha256": sha256(EXECUTOR_SOURCE),
        "controller_sha256": sha256(CONTROLLER_SOURCE),
        "guarded_action_sha256": sha256(source_dir / "actions_policy_boundary_guarded.csv"),
        "frozen_split_sha256": sha256(source_dir / "frozen_group_split.csv"),
        "null_seeds": int(2 if args.smoke else args.null_seeds),
        "empirical_subspace_rank": int(args.subspace_rank),
        "heldout_tuning": False,
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"[{model_key}] load and reproduce frozen setup", flush=True)
    model, tokenizer = executor.load_model_and_tokenizer(cfg)
    dataset = executor.build_dataset(tokenizer, cfg, runtime)
    base, hidden, _ = executor.extract_baseline(model, tokenizer, dataset, cfg, runtime)
    train_index, test_index, split = executor.make_split(base, cfg, runtime)
    frozen_split = pd.read_csv(source_dir / "frozen_group_split.csv")
    regenerated = split[["prompt_id", "split"]].sort_values("prompt_id").reset_index(drop=True)
    expected = frozen_split[["prompt_id", "split"]].sort_values("prompt_id").reset_index(drop=True)
    if not regenerated.equals(expected):
        raise RuntimeError(f"{model_key}: regenerated split differs from frozen source")
    base, _ = controller.refit_delta_u_on_train(executor, base, train_index, cfg, out_dir)
    stable_ops = executor.fit_stable_operators(base, hidden, train_index, cfg, runtime)
    precursor_dirs, _, _ = executor.fit_precursor_and_probes(base, hidden, train_index, cfg, runtime)
    test = base.iloc[test_index].reset_index(drop=True)
    guarded = pd.read_csv(source_dir / "actions_policy_boundary_guarded.csv")
    expected_pairs = {(str(prompt), int(layer)) for prompt in test["prompt_id"] for layer in cfg.operator_layers}
    observed_pairs = set(zip(guarded["prompt_id"].astype(str), guarded["layer"].astype(int)))
    if expected_pairs != observed_pairs:
        raise RuntimeError(f"{model_key}: guarded schedule does not match the frozen held-out prompts")
    guarded.to_csv(out_dir / "frozen_guarded_actions.csv", index=False)

    executor.make_combo_hook_factory = controller.norm_tracing_hook_factory(executor)
    baseline, _ = evaluate_or_load(attribution, executor, model, tokenizer, test, cfg, stable_ops, precursor_dirs, "none", None, out_dir)
    structured, structured_trace = evaluate_or_load(attribution, executor, model, tokenizer, test, cfg, stable_ops, precursor_dirs, "policy_boundary_guarded", guarded, out_dir)
    published = pd.read_csv(source_dir / "vocab_top1_policy_boundary_guarded.csv").sort_values("prompt_id")
    reproduced = structured.sort_values("prompt_id")
    exact = bool(np.array_equal(published["top1_token_id"].to_numpy(), reproduced["top1_token_id"].to_numpy()))
    if not exact:
        raise RuntimeError(f"{model_key}: structured actuator did not reproduce the published held-out output")

    all_rows = [classify_outcomes(structured, baseline, model_key, "structured_guarded", None)]
    trace_rows = [structured_trace.assign(model=model_key, random_seed=np.nan)]
    action_map = action_lookup(guarded)
    seed_count = 2 if args.smoke else args.null_seeds
    for offset in range(seed_count):
        random_seed = int(args.seed + 1000 * (list(MODEL_WINDOWS).index(model_key) + 1) + offset)
        bases = empirical_bases(hidden, train_index, cfg, args.subspace_rank, random_seed)
        executor.make_combo_hook_factory = random_direction_hook_factory(executor, bases)
        control = f"random_direction_{offset:03d}"
        result, trace = evaluate_or_load(attribution, executor, model, tokenizer, test, cfg, stable_ops, precursor_dirs, control, guarded, out_dir)
        all_rows.append(classify_outcomes(result, baseline, model_key, control, random_seed))
        trace_rows.append(trace.assign(model=model_key, random_seed=random_seed))
        print(f"[{model_key}] completed random direction {offset + 1}/{seed_count}", flush=True)

    rows = pd.concat(all_rows, ignore_index=True)
    traces = pd.concat(trace_rows, ignore_index=True)
    rows.to_csv(out_dir / "direction_null_prompt_rows.csv", index=False)
    traces.to_csv(out_dir / "direction_null_traces.csv", index=False)
    summary = summarize(rows)
    structured_summary = summary[summary["control"].eq("structured_guarded")].iloc[0]
    null = summary[summary["control"].str.startswith("random_direction_")]
    final = {
        "model": model_key,
        "published_structured_top1_reproduced": exact,
        "structured_specificity_count": int(structured_summary["specificity_count"]),
        "structured_specificity_rate": float(structured_summary["specificity_rate"]),
        "null_count": int(len(null)),
        "null_specificity_count_mean": float(null["specificity_count"].mean()),
        "null_specificity_count_q95": float(np.quantile(null["specificity_count"], 0.95)),
        "empirical_p": float((int(np.sum(null["specificity_count"].to_numpy(int) >= int(structured_summary["specificity_count"]))) + 1) / (len(null) + 1)),
        "structured_exceeds_null_q95": bool(int(structured_summary["specificity_count"]) > float(np.quantile(null["specificity_count"], 0.95))),
        "finite_traces": bool(traces["all_finite"].astype(bool).all()),
    }
    (out_dir / "direction_null_verdict.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    del model, tokenizer, base, hidden, stable_ops, precursor_dirs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, traces, final


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries, traces, verdicts = [], [], []
    for model_key in args.models:
        summary, trace, verdict = run_model(model_key, args)
        summaries.append(summary)
        traces.append(trace)
        verdicts.append(verdict)
    summary = pd.concat(summaries, ignore_index=True)
    trace = pd.concat(traces, ignore_index=True)
    summary.to_csv(args.output_root / "direction_null_crossmodel_summary.csv", index=False)
    trace.to_csv(args.output_root / "direction_null_crossmodel_traces.csv", index=False)
    (args.output_root / "direction_null_crossmodel_verdicts.json").write_text(json.dumps(verdicts, indent=2), encoding="utf-8")
    print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
