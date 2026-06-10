# -*- coding: utf-8 -*-
r"""
SEM MemoryUnit Bank Exporter

Purpose
-------
Create a real hidden-dim MemoryUnit bank for SEM-6H.2b:

    sem5c_outputs\sem5c_memory_bank.npz

The SEM-6H.2b live writer expects:

    family_ids
    mu_shape   [n_family, hidden_dim]   or shape_l7...shape_l19
    mu_commit  [n_family, hidden_dim]   or commit_l23...commit_l25

This exporter searches your local project outputs for full hidden-state trajectory
tables and builds:

    mu_shape  = mean hidden vector over L7-L19 per family
    mu_commit = mean hidden vector over L23-L25 per family

It refuses to export PCA/compressed vectors unless they are already full hidden_dim.

Default root
------------
C:\Users\ZH\Desktop\AGI\python_script

Run
---
conda activate dhrf_4080s
cd C:\Users\ZH\Desktop\AGI\python_script

# 1) Inventory likely files:
python GPT_sem_memory_bank_exporter.py --inventory_only

# 2) Try automatic export:
python GPT_sem_memory_bank_exporter.py

# 3) If auto search finds the wrong file, specify explicit input:
python GPT_sem_memory_bank_exporter.py ^
  --input_csv sem5c_outputs\YOUR_FULL_HIDDEN_FEATURES.csv ^
  --out_npz sem5c_outputs\sem5c_memory_bank.npz

Supported input formats
-----------------------

Long format, preferred:
    family, layer, h_0, h_1, ..., h_D
or:
    seed_family, layer, hidden_0, hidden_1, ..., hidden_D

Wide format:
    family, shape_l7_d0 ... shape_l19_dD, commit_l23_d0 ... commit_l25_dD
or:
    family, mu_shape_0 ... mu_shape_D, mu_commit_0 ... mu_commit_D

The output vectors must be hidden_dim, e.g. Qwen2.5-1.5B hidden_size is usually 1536.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


DEFAULT_ROOT = r"C:\Users\ZH\Desktop\AGI\python_script"
DEFAULT_OUT = rf"{DEFAULT_ROOT}\sem5c_outputs\sem5c_memory_bank.npz"

SHAPE_LAYERS = list(range(7, 20))
COMMIT_LAYERS = [23, 24, 25]

SEARCH_DIRS = [
    "sem5c_outputs",
    "sem5c2_outputs",
    "sem5_outputs",
    "sem6_outputs",
    "sem6e_outputs",
    "sem6f_outputs",
    "sem6g_outputs",
    "sem6h_outputs",
    "sem6h1b_outputs",
    ".",
]

FILENAME_HINTS = [
    "hidden", "trajectory", "features", "memory", "unit", "seed", "sem5", "sem6"
]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def find_family_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["family", "seed_family", "target_family", "family_id", "row_family"]
    nmap = {norm(c): c for c in df.columns}
    for c in candidates:
        if norm(c) in nmap:
            return nmap[norm(c)]
    for c in df.columns:
        if "family" in norm(c):
            return c
    return None


def find_layer_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["layer", "layer_id", "l", "layer_idx"]
    nmap = {norm(c): c for c in df.columns}
    for c in candidates:
        if norm(c) in nmap:
            return nmap[norm(c)]
    return None


def hidden_cols_long(df: pd.DataFrame) -> List[str]:
    cols = []
    patterns = [
        r"^h_?\d+$",
        r"^hidden_?\d+$",
        r"^hid_?\d+$",
        r"^vec_?\d+$",
        r"^x_?\d+$",
        r"^center_?\d+$",
    ]
    for c in df.columns:
        nc = norm(c)
        if any(re.match(p, nc) for p in patterns):
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)

    def key(c):
        m = re.search(r"(\d+)$", norm(c))
        return int(m.group(1)) if m else 0
    return sorted(cols, key=key)


def vector_cols_prefix(df: pd.DataFrame, prefixes: List[str]) -> List[str]:
    cols = []
    for c in df.columns:
        nc = norm(c)
        for p in prefixes:
            pp = norm(p)
            if re.match(rf"^{pp}_(?:d)?\d+$", nc):
                if pd.api.types.is_numeric_dtype(df[c]):
                    cols.append(c)
                break

    def key(c):
        m = re.search(r"(\d+)$", norm(c))
        return int(m.group(1)) if m else 0
    return sorted(cols, key=key)


def layer_vector_cols(df: pd.DataFrame, kind: str, layer: int) -> List[str]:
    cols = []
    for c in df.columns:
        nc = norm(c)
        if kind in nc and re.search(rf"(?:l|layer)_?{layer}(?:_|$)", nc) and re.search(r"(?:d)?\d+$", nc):
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)

    def key(c):
        m = re.search(r"(\d+)$", norm(c))
        return int(m.group(1)) if m else 0
    return sorted(cols, key=key)


def candidate_files(root: Path) -> List[Path]:
    files = []
    for d in SEARCH_DIRS:
        p = root / d
        if not p.exists():
            continue
        for ext in ["*.csv", "*.parquet", "*.pkl", "*.npz"]:
            files.extend(p.glob(ext))

    def score(p: Path):
        name = norm(p.name)
        s = sum(1 for h in FILENAME_HINTS if h in name)
        # prefer sem5c/memory/hidden
        if "sem5c" in name: s += 3
        if "memory" in name: s += 3
        if "hidden" in name: s += 3
        if "trajectory" in name: s += 2
        return -s, len(str(p))
    return sorted(set(files), key=score)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, nrows=None)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".pkl":
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported table file: {path}")


def inspect_file(path: Path, max_rows_preview: int = 3) -> Dict:
    info = {"path": str(path), "suffix": path.suffix, "usable": False, "reason": ""}
    try:
        if path.suffix.lower() == ".npz":
            npz = np.load(path, allow_pickle=True)
            keys = list(npz.keys())
            info["npz_keys"] = keys
            has_family = any("family" in norm(k) for k in keys) or any(k in keys for k in ["family_ids", "families"])
            has_shape = any("shape" in norm(k) for k in keys)
            has_commit = any("commit" in norm(k) for k in keys)
            info["usable"] = bool(has_family and (has_shape or has_commit))
            info["reason"] = "npz family+shape/commit" if info["usable"] else "npz lacks family and shape/commit keys"
            return info

        df = read_table(path)
        info["shape"] = df.shape
        info["columns_sample"] = list(df.columns[:30])
        fam = find_family_col(df)
        lyr = find_layer_col(df)
        hcols = hidden_cols_long(df)
        mu_shape = vector_cols_prefix(df, ["mu_shape", "shape_mu", "shape_center", "shape"])
        mu_commit = vector_cols_prefix(df, ["mu_commit", "commit_mu", "commit_center", "commit"])
        layer_shape_any = any(layer_vector_cols(df, "shape", l) for l in SHAPE_LAYERS)
        layer_commit_any = any(layer_vector_cols(df, "commit", l) for l in COMMIT_LAYERS)

        info["family_col"] = fam
        info["layer_col"] = lyr
        info["n_hidden_cols"] = len(hcols)
        info["n_mu_shape_cols"] = len(mu_shape)
        info["n_mu_commit_cols"] = len(mu_commit)
        info["has_layer_shape_cols"] = bool(layer_shape_any)
        info["has_layer_commit_cols"] = bool(layer_commit_any)

        if fam and lyr and len(hcols) >= 256:
            info["usable"] = True
            info["reason"] = f"long format: family={fam}, layer={lyr}, hidden_dim={len(hcols)}"
        elif fam and len(mu_shape) >= 256 and len(mu_commit) >= 256:
            info["usable"] = True
            info["reason"] = f"wide mu format: family={fam}, shape_dim={len(mu_shape)}, commit_dim={len(mu_commit)}"
        elif fam and layer_shape_any and layer_commit_any:
            info["usable"] = True
            info["reason"] = f"wide layer-specific format: family={fam}"
        else:
            info["reason"] = "not enough family/layer/hidden or mu_shape/mu_commit columns"
        return info
    except Exception as e:
        info["reason"] = f"error: {type(e).__name__}: {e}"
        return info


def load_from_long(df: pd.DataFrame, family_col: str, layer_col: str, hcols: List[str]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    sub = df[[family_col, layer_col] + hcols].copy()
    sub[layer_col] = pd.to_numeric(sub[layer_col], errors="coerce")
    sub = sub.dropna(subset=[family_col, layer_col])
    sub[layer_col] = sub[layer_col].astype(int)

    families = sorted(sub[family_col].astype(str).unique().tolist())
    shape_vecs = []
    commit_vecs = []

    for fam in families:
        fdf = sub[sub[family_col].astype(str) == fam]
        sdf = fdf[fdf[layer_col].isin(SHAPE_LAYERS)]
        cdf = fdf[fdf[layer_col].isin(COMMIT_LAYERS)]

        if len(sdf) == 0 or len(cdf) == 0:
            # skip incomplete family
            continue

        shape = sdf[hcols].to_numpy(dtype=np.float32).mean(axis=0)
        commit = cdf[hcols].to_numpy(dtype=np.float32).mean(axis=0)
        shape_vecs.append(shape)
        commit_vecs.append(commit)

    # Rebuild families after skips.
    good_fams = []
    shape_vecs = []
    commit_vecs = []
    for fam in families:
        fdf = sub[sub[family_col].astype(str) == fam]
        sdf = fdf[fdf[layer_col].isin(SHAPE_LAYERS)]
        cdf = fdf[fdf[layer_col].isin(COMMIT_LAYERS)]
        if len(sdf) == 0 or len(cdf) == 0:
            continue
        good_fams.append(fam)
        shape_vecs.append(sdf[hcols].to_numpy(dtype=np.float32).mean(axis=0))
        commit_vecs.append(cdf[hcols].to_numpy(dtype=np.float32).mean(axis=0))

    if not good_fams:
        raise ValueError("No families have both shape L7-L19 and commit L23-L25 rows.")
    return good_fams, np.stack(shape_vecs), np.stack(commit_vecs)


def load_from_wide(df: pd.DataFrame, family_col: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    families = df[family_col].astype(str).tolist()

    mu_shape = []
    for p in ["mu_shape", "shape_mu", "shape_center", "shape"]:
        mu_shape = vector_cols_prefix(df, [p])
        if mu_shape:
            break

    mu_commit = []
    for p in ["mu_commit", "commit_mu", "commit_center", "commit"]:
        mu_commit = vector_cols_prefix(df, [p])
        if mu_commit:
            break

    if mu_shape and mu_commit:
        return families, df[mu_shape].to_numpy(dtype=np.float32), df[mu_commit].to_numpy(dtype=np.float32)

    # Layer-specific: average across layers.
    shape_arrays = []
    for l in SHAPE_LAYERS:
        cols = layer_vector_cols(df, "shape", l)
        if cols:
            shape_arrays.append(df[cols].to_numpy(dtype=np.float32))
    commit_arrays = []
    for l in COMMIT_LAYERS:
        cols = layer_vector_cols(df, "commit", l)
        if cols:
            commit_arrays.append(df[cols].to_numpy(dtype=np.float32))

    if not shape_arrays or not commit_arrays:
        raise ValueError("No supported wide shape/commit columns found.")

    shape = np.mean(np.stack(shape_arrays, axis=1), axis=1)
    commit = np.mean(np.stack(commit_arrays, axis=1), axis=1)
    return families, shape, commit


def export_bank(input_path: Path, out_npz: Path, expected_dim: int = 0) -> Dict:
    if input_path.suffix.lower() == ".npz":
        # Validate and copy to output location under canonical name if already usable.
        npz = np.load(input_path, allow_pickle=True)
        keys = list(npz.keys())
        fam_key = None
        for k in keys:
            if norm(k) in {"family_ids", "families", "family", "seed_family"} or "family" in norm(k):
                fam_key = k
                break
        if fam_key is None:
            raise ValueError(f"NPZ missing family key: {keys}")

        shape_key = None
        commit_key = None
        for k in keys:
            nk = norm(k)
            if nk in {"mu_shape", "shape_mu", "shape_center", "shape", "mu_f_shape"}:
                shape_key = k
            if nk in {"mu_commit", "commit_mu", "commit_center", "commit", "mu_f_commit"}:
                commit_key = k
        if shape_key is None or commit_key is None:
            raise ValueError(f"NPZ missing mu_shape/mu_commit. Keys={keys}")

        families = npz[fam_key]
        shape = np.asarray(npz[shape_key], dtype=np.float32)
        commit = np.asarray(npz[commit_key], dtype=np.float32)
        if shape.ndim == 3:
            shape = shape.mean(axis=1)
        if commit.ndim == 3:
            commit = commit.mean(axis=1)
    else:
        df = read_table(input_path)
        fam = find_family_col(df)
        if fam is None:
            raise ValueError("Input table has no family column.")
        lyr = find_layer_col(df)
        hcols = hidden_cols_long(df)
        if lyr and len(hcols) >= 256:
            families, shape, commit = load_from_long(df, fam, lyr, hcols)
        else:
            families, shape, commit = load_from_wide(df, fam)

        families = np.array(families, dtype=object)

    if shape.shape != commit.shape:
        raise ValueError(f"shape and commit arrays have different shapes: {shape.shape} vs {commit.shape}")
    if expected_dim > 0 and shape.shape[-1] != expected_dim:
        raise ValueError(
            f"Exported dim={shape.shape[-1]} but expected_dim={expected_dim}. "
            "This likely means you selected PCA/compressed features, not full hidden vectors."
        )
    if shape.shape[-1] < 256:
        raise ValueError(
            f"Exported dim={shape.shape[-1]} is too small for hidden vectors. "
            "This looks like PCA/compressed features. Need full hidden_dim vectors."
        )

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        family_ids=np.array([str(x) for x in families], dtype=object),
        mu_shape=shape.astype(np.float32),
        mu_commit=commit.astype(np.float32),
        shape_layers=np.array(SHAPE_LAYERS, dtype=np.int32),
        commit_layers=np.array(COMMIT_LAYERS, dtype=np.int32),
        source_file=str(input_path),
    )

    return {
        "out_npz": str(out_npz),
        "source_file": str(input_path),
        "n_family": int(len(families)),
        "hidden_dim": int(shape.shape[-1]),
        "shape": list(shape.shape),
        "commit": list(commit.shape),
        "shape_layers": SHAPE_LAYERS,
        "commit_layers": COMMIT_LAYERS,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--input_csv", default="", help="Explicit input CSV/parquet/pkl/npz.")
    ap.add_argument("--out_npz", default=DEFAULT_OUT)
    ap.add_argument("--expected_dim", type=int, default=1536, help="Qwen2.5-1.5B hidden size is usually 1536. Use 0 to disable.")
    ap.add_argument("--inventory_only", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    if args.input_csv.strip():
        files = [Path(args.input_csv)]
    else:
        files = candidate_files(root)

    infos = []
    for p in files:
        infos.append(inspect_file(p))

    inv_path = root / "sem_memory_bank_inventory.json"
    with open(inv_path, "w", encoding="utf-8") as f:
        json.dump(infos, f, ensure_ascii=False, indent=2)

    print(f"[inventory] wrote: {inv_path}")
    print("\nTop candidate files:")
    for info in infos[:30]:
        flag = "USABLE" if info.get("usable") else "-----"
        print(f"{flag} | {info.get('path')} | {info.get('reason')}")

    if args.inventory_only:
        print("\n[inventory_only] Stop here. If no USABLE file appears, you need to export full hidden features from SEM-5C/5C.2 first.")
        return

    usable = [i for i in infos if i.get("usable")]
    if not usable:
        raise FileNotFoundError(
            "No usable source found for full hidden-dim MemoryUnit bank. "
            f"See inventory: {inv_path}\n"
            "You likely only have PCA/compressed outputs. Re-run SEM-5C feature extraction and save full hidden vectors per family/layer."
        )

    src = Path(usable[0]["path"])
    out = Path(args.out_npz)
    result = export_bank(src, out, expected_dim=args.expected_dim)
    print("\n[EXPORT DONE]")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nNext:")
    print("python GPT_sem6h2b_true_memoryunit_live_write.py --inspect_only")
    print("python GPT_sem6h2b_true_memoryunit_live_write.py --max_rows 120 --max_null_reps 2")


if __name__ == "__main__":
    main()
