"""Independent recomputation checks for the v0.56 new-graph geometry audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 2026080303
N_PROMPTS = 240
N_GRAPHS = 24
N_SHUFFLES = 50
CONDITIONS = {
    "clean", "permuted", "redundant", "irrelevant", "weak_distractor",
    "competition_balanced", "direct_conflict", "closure_update",
    "closure_override", "exception_override",
}


def metric(states: np.ndarray) -> dict[str, np.ndarray]:
    unit = states.astype(np.float32, copy=False)
    unit /= np.maximum(np.linalg.norm(unit, axis=-1, keepdims=True), 1e-8)
    steps = unit[:, 1:] - unit[:, :-1]
    step_norm = np.linalg.norm(steps, axis=-1)
    chord = unit[:, -1] - unit[:, 0]
    chord /= np.maximum(np.linalg.norm(chord, axis=-1, keepdims=True), 1e-8)
    alignment = np.sum(steps * chord[:, None], axis=-1) / np.maximum(step_norm, 1e-8)
    path = np.sum(np.clip(1.0 - np.sum(unit[:, :-1] * unit[:, 1:], axis=-1), 0.0, None), axis=1)
    endpoint = np.clip(1.0 - np.sum(unit[:, 0] * unit[:, -1], axis=-1), 0.0, None)
    turn = np.sum(steps[:, :-1] * steps[:, 1:], axis=-1) / np.maximum(step_norm[:, :-1] * step_norm[:, 1:], 1e-8)
    return {
        "chord_alignment": np.mean(alignment, axis=1),
        "detour_ratio": path / np.maximum(endpoint, 1e-8),
        "turning_curvature": np.mean(np.arccos(np.clip(turn, -1.0, 1.0)), axis=1),
    }


def validate_model(root: Path, model_key: str, array_file: str, expected_shape: tuple[int, int, int]) -> dict[str, bool]:
    states = np.load(root / array_file)
    summary = pd.read_csv(root / "newgraph_geometry_summary.csv").set_index("model").loc[model_key]
    prompt_table = pd.read_csv(root / f"{model_key}_prompt_geometry.csv")
    null_table = pd.read_csv(root / f"{model_key}_shared_endpoint_shuffle_null.csv")
    checks = {
        f"{model_key}_shape": tuple(states.shape) == expected_shape,
        f"{model_key}_finite": bool(np.isfinite(states).all()),
        f"{model_key}_prompt_rows": len(prompt_table) == expected_shape[0],
        f"{model_key}_null_rows": len(null_table) == N_SHUFFLES,
    }
    real = metric(states)
    rng = np.random.default_rng(SEED)
    middle = np.arange(1, states.shape[1] - 1)
    shuffled = {name: [] for name in real}
    for _ in range(N_SHUFFLES):
        order = np.concatenate(([0], rng.permutation(middle), [states.shape[1] - 1]))
        values = metric(states[:, order])
        for name in shuffled:
            shuffled[name].append(float(np.mean(values[name])))
    for name, values in real.items():
        observed = float(np.mean(values))
        null_values = np.asarray(shuffled[name], dtype=float)
        direction = 1.0 if name == "chord_alignment" else -1.0
        expected = {
            f"real_{name}": observed,
            f"null_{name}_mean": float(np.mean(null_values)),
            f"null_{name}_sd": float(np.std(null_values, ddof=1)),
            f"real_minus_null_{name}": observed - float(np.mean(null_values)),
            f"null_standardized_{name}": direction * (observed - float(np.mean(null_values))) / max(float(np.std(null_values, ddof=1)), 1e-8),
            f"empirical_p_{name}": (1 + int(np.sum(direction * null_values >= direction * observed))) / (N_SHUFFLES + 1),
        }
        for column, value in expected.items():
            checks[f"{model_key}_{column}"] = bool(np.isclose(float(summary[column]), value, rtol=1e-6, atol=1e-7))
        checks[f"{model_key}_prompt_{name}"] = bool(np.allclose(prompt_table[name].to_numpy(float), values, rtol=1e-6, atol=1e-7))
        checks[f"{model_key}_null_{name}"] = bool(np.allclose(null_table[name].to_numpy(float), null_values, rtol=1e-6, atol=1e-7))
    return checks


def validate_processed_model(root: Path, model_key: str, expected_shape: tuple[int, int, int]) -> dict[str, bool]:
    """Verify released prompt metrics and shuffle summaries without raw arrays."""
    summary = pd.read_csv(root / "newgraph_geometry_summary.csv").set_index("model").loc[model_key]
    prompt_table = pd.read_csv(root / f"{model_key}_prompt_geometry.csv")
    null_table = pd.read_csv(root / f"{model_key}_shared_endpoint_shuffle_null.csv")
    checks = {
        f"{model_key}_released_prompt_rows": len(prompt_table) == expected_shape[0],
        f"{model_key}_released_null_rows": len(null_table) == N_SHUFFLES,
        f"{model_key}_released_layers": int(summary["n_layers"]) == expected_shape[1],
        f"{model_key}_released_hidden_dim": int(summary["hidden_dim"]) == expected_shape[2],
        f"{model_key}_released_finite": bool(
            np.isfinite(prompt_table[["chord_alignment", "detour_ratio", "turning_curvature"]].to_numpy(float)).all()
            and np.isfinite(null_table[["chord_alignment", "detour_ratio", "turning_curvature"]].to_numpy(float)).all()
        ),
    }
    for name in ("chord_alignment", "detour_ratio", "turning_curvature"):
        observed = float(prompt_table[name].mean())
        null_values = null_table[name].to_numpy(float)
        direction = 1.0 if name == "chord_alignment" else -1.0
        expected = {
            f"real_{name}": observed,
            f"null_{name}_mean": float(np.mean(null_values)),
            f"null_{name}_sd": float(np.std(null_values, ddof=1)),
            f"real_minus_null_{name}": observed - float(np.mean(null_values)),
            f"null_standardized_{name}": direction * (observed - float(np.mean(null_values))) / max(float(np.std(null_values, ddof=1)), 1e-8),
            f"empirical_p_{name}": (1 + int(np.sum(direction * null_values >= direction * observed))) / (N_SHUFFLES + 1),
        }
        for column, value in expected.items():
            checks[f"{model_key}_released_{column}"] = bool(
                np.isclose(float(summary[column]), value, rtol=1e-6, atol=1e-7)
            )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    root = args.output_root
    manifest = pd.read_csv(root / "newgraph_prompt_manifest.csv")
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    content_hash = hashlib.sha256("\n".join(manifest.text.tolist()).encode("utf-8")).hexdigest()
    checks: dict[str, bool] = {
        "manifest_rows": len(manifest) == N_PROMPTS,
        "manifest_graphs": manifest.graph_id.nunique() == N_GRAPHS,
        "manifest_conditions": set(manifest.condition) == CONDITIONS,
        "manifest_each_graph_complete": bool((manifest.groupby("graph_id").size() == len(CONDITIONS)).all()),
        "manifest_content_hash": metadata.get("prompt_content_sha256") == content_hash,
        "declared_shuffle_count": int(metadata.get("n_shared_endpoint_shuffles", -1)) == N_SHUFFLES,
        "raw_prompt_wrapping": metadata.get("prompt_wrapping") == "raw fixed prompt string in both runtimes",
    }
    raw_available = (root / "qwen_all_block_states.npy").exists() and (root / "gemma3_12b_all_block_states.npy").exists()
    if raw_available:
        checks.update(validate_model(root, "Qwen2.5-1.5B-Instruct", "qwen_all_block_states.npy", (240, 28, 1536)))
        checks.update(validate_model(root, "Gemma-3-12B-it-QAT-Q4_0", "gemma3_12b_all_block_states.npy", (240, 48, 3840)))
        captures = root / "gemma3_captures"
        checks["gemma_capture_file_count"] = len(list(captures.glob("*.f32"))) == N_PROMPTS and len(list(captures.glob("*.json"))) == N_PROMPTS
        verification_mode = "raw-state recomputation"
    else:
        checks.update(validate_processed_model(root, "Qwen2.5-1.5B-Instruct", (240, 28, 1536)))
        checks.update(validate_processed_model(root, "Gemma-3-12B-it-QAT-Q4_0", (240, 48, 3840)))
        checks["raw_arrays_excluded_from_public_mirror"] = True
        verification_mode = "processed-source verification; raw-state recomputation requires the archive intermediates"
    result = {
        "experiment": "v0_56_newgraph_geometry_confirmation",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "verification_mode": verification_mode,
        "checks": checks,
    }
    if args.report is not None:
        args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
