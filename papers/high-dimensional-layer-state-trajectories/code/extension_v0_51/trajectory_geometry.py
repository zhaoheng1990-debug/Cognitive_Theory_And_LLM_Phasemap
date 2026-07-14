"""Shared numerical definitions for ordered hidden-state trajectory controls."""

from __future__ import annotations

import numpy as np


EPS = 1e-8


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, EPS)


def trajectory_metrics(states: np.ndarray) -> dict[str, np.ndarray]:
    unit = normalize_rows(np.asarray(states, dtype=np.float32))
    steps = np.diff(unit, axis=1)
    step_norm = np.linalg.norm(steps, axis=-1)
    chord = unit[:, -1, :] - unit[:, 0, :]
    chord_unit = normalize_rows(chord)
    alignment = np.sum(steps * chord_unit[:, None, :], axis=-1) / np.maximum(step_norm, EPS)

    step_distance = 1.0 - np.sum(unit[:, :-1, :] * unit[:, 1:, :], axis=-1)
    path_length = np.sum(np.clip(step_distance, 0.0, None), axis=1)
    endpoint_distance = 1.0 - np.sum(unit[:, 0, :] * unit[:, -1, :], axis=-1)
    endpoint_distance = np.clip(endpoint_distance, 0.0, None)
    detour_ratio = path_length / np.maximum(endpoint_distance, EPS)

    turn_cos = np.sum(steps[:, :-1, :] * steps[:, 1:, :], axis=-1)
    turn_cos /= np.maximum(step_norm[:, :-1] * step_norm[:, 1:], EPS)
    turns = np.arccos(np.clip(turn_cos, -1.0, 1.0))
    return {
        "alignment_mean": np.mean(alignment, axis=1),
        "alignment_positive_fraction": np.mean(alignment > 0.0, axis=1),
        "path_length": path_length,
        "endpoint_distance": endpoint_distance,
        "detour_ratio": detour_ratio,
        "curvature_mean": np.mean(turns, axis=1),
    }


def summarize_against_endpoint_shuffles(
    states: np.ndarray, n_shuffles: int, rng: np.random.Generator
) -> tuple[dict[str, float], list[dict[str, float]]]:
    real = trajectory_metrics(states)
    real_means = {name: float(np.mean(values)) for name, values in real.items()}
    middle = np.arange(1, states.shape[1] - 1)
    null_rows: list[dict[str, float]] = []
    for shuffle_id in range(n_shuffles):
        order = np.concatenate(([0], rng.permutation(middle), [states.shape[1] - 1]))
        shuffled = trajectory_metrics(states[:, order, :])
        row = {"shuffle_id": float(shuffle_id)}
        row.update({name: float(np.mean(values)) for name, values in shuffled.items()})
        null_rows.append(row)
    return real_means, null_rows


def geometry_effect_summary(
    real_means: dict[str, float], null_rows: list[dict[str, float]]
) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric, real_value in real_means.items():
        null = np.asarray([row[metric] for row in null_rows], dtype=float)
        null_sd = float(np.std(null, ddof=1))
        summary[f"real_{metric}"] = real_value
        summary[f"shuffle_{metric}_mean"] = float(np.mean(null))
        summary[f"shuffle_{metric}_sd"] = null_sd
        summary[f"shuffle_{metric}_q95"] = float(np.quantile(null, 0.95))
        summary[f"real_vs_shuffle_{metric}_z"] = (
            (real_value - float(np.mean(null))) / null_sd if null_sd > 0 else np.nan
        )
        summary[f"real_exceeds_all_shuffles_{metric}"] = float(real_value > float(np.max(null)))
    summary["alignment_gain"] = (
        real_means["alignment_mean"]
        - float(np.mean([row["alignment_mean"] for row in null_rows]))
    )
    summary["detour_reduction_log2"] = float(
        np.log2(
            np.mean([row["detour_ratio"] for row in null_rows])
            / max(real_means["detour_ratio"], EPS)
        )
    )
    summary["curvature_reduction_log2"] = float(
        np.log2(
            np.mean([row["curvature_mean"] for row in null_rows])
            / max(real_means["curvature_mean"], EPS)
        )
    )
    return summary
