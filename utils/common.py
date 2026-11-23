"""
通用工具函数：
1. 随机种子；
2. 路径创建；
3. 分层划分索引。
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """设置随机种子，保证实验可重复。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> Path:
    """确保目录存在，不存在时自动创建。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def stratified_split_indices(
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    纯 numpy 实现分层划分，避免依赖 sklearn。

    返回:
    - train_idx
    - val_idx
    - test_idx
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 3:
            n_train = min(max(n_train, 1), n - 2)
            n_val = min(max(n_val, 1), n - n_train - 1)
        elif n == 2:
            n_train, n_val = 1, 0
        else:
            n_train, n_val = 1, 0

        tr = cls_idx[:n_train]
        va = cls_idx[n_train : n_train + n_val]
        te = cls_idx[n_train + n_val :]

        train_idx.append(tr)
        val_idx.append(va)
        test_idx.append(te)

    train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_idx) if val_idx else np.array([], dtype=np.int64)
    test_idx = np.concatenate(test_idx) if test_idx else np.array([], dtype=np.int64)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx
