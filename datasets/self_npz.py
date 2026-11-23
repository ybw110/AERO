"""
SelfA / SelfB processed npz loader.

Expected layout:
- <self_processed_root>/SelfA/domain_1200.npz
- <self_processed_root>/SelfA/domain_1800.npz
- <self_processed_root>/SelfB/domain_1200.npz
- <self_processed_root>/SelfB/domain_1800.npz

Each npz should contain:
- X: [N, L] or [N, 1, L]
- y: [N]
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from config import Route1Config


def _extract_domain_name(npz_path: Path) -> str:
    stem = npz_path.stem
    return stem[len("domain_") :] if stem.startswith("domain_") else stem


def _discover_domains(data_root: Path) -> List[str]:
    files = sorted(data_root.glob("domain_*.npz"))
    if not files:
        files = sorted(data_root.glob("*.npz"))
    return [_extract_domain_name(p) for p in files]


def _resolve_self_root(cfg: Route1Config, dataset_name: str) -> Path:
    if dataset_name == "SELF_A":
        return Path(cfg.self_processed_root) / cfg.self_a_subdir
    if dataset_name == "SELF_B":
        return Path(cfg.self_processed_root) / cfg.self_b_subdir
    raise ValueError(f"Unsupported self dataset: {dataset_name}")


def _coerce_shape(x: np.ndarray, seq_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    elif x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]
    elif x.ndim != 2:
        raise ValueError(f"Unexpected X shape: {x.shape}, expected [N,L] or [N,1,L].")

    if seq_len > 0 and x.shape[1] != seq_len:
        if x.shape[1] > seq_len:
            x = x[:, :seq_len]
        else:
            pad = np.zeros((x.shape[0], seq_len - x.shape[1]), dtype=np.float32)
            x = np.concatenate([x, pad], axis=1)
    return x.astype(np.float32)


def load_self_npz(cfg: Route1Config, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    dataset_name = dataset_name.upper()
    data_root = _resolve_self_root(cfg, dataset_name)
    if not data_root.is_dir():
        raise FileNotFoundError(f"{dataset_name} directory not found: {data_root}")

    domains = cfg.source_domains or _discover_domains(data_root)
    if not domains:
        raise FileNotFoundError(f"No npz files found under: {data_root}")

    domain_map = {d: i for i, d in enumerate(domains)}
    xs, ys, ds = [], [], []

    for d in domains:
        npz_path = data_root / f"domain_{d}.npz"
        if not npz_path.exists():
            fallback = data_root / f"{d}.npz"
            if fallback.exists():
                npz_path = fallback
            else:
                raise FileNotFoundError(f"Domain file missing for '{d}': {npz_path}")

        data = np.load(npz_path)
        if "X" not in data or "y" not in data:
            raise KeyError(f"Invalid npz file {npz_path}, required keys: X, y.")

        x = _coerce_shape(data["X"], cfg.sequence_length)
        y = np.asarray(data["y"], dtype=np.int64).reshape(-1)
        if len(x) != len(y):
            raise ValueError(f"X/y size mismatch in {npz_path}: len(X)={len(x)}, len(y)={len(y)}")

        xs.append(x)
        ys.append(y)
        ds.append(np.full(len(y), domain_map[d], dtype=np.int64))

    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ds), domain_map

