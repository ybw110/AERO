"""
数据构建入口（dataset + dataloader）。

职责：
1. 根据 dataset_name 调用对应加载器；
2. 执行 known/unknown 标签映射；
3. 切分 train/val/test 并返回 DataLoader。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader

from config import Route1Config
from datasets.base_dataset import Route1Dataset
from datasets.cwru_jnu import load_cwru, load_jnu
from datasets.self_npz import load_self_npz
from utils.common import stratified_split_indices


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    known_map: Dict[int, int]
    domain_name_to_id: Dict[str, int]


@dataclass
class EvalBundle:
    test_loader: DataLoader
    known_map: Dict[int, int]
    domain_name_to_id: Dict[str, int]


def _map_labels(y: np.ndarray, known_map: Dict[int, int]) -> np.ndarray:
    """
    将原始标签映射为训练标签；未知类记为 -1。

    例子：
    - known_map={0:0,1:1,3:2}
    - 原始标签 2 不在 known_map 中，会被映射成 -1（开集未知）。
    """
    return np.array([known_map.get(int(v), -1) for v in y], dtype=np.int64)


def _normalize_by_train_stats(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Normalize by train split statistics to avoid leakage.
    """
    mean = float(x_train.mean())
    std = float(x_train.std())
    if std < 1e-8:
        std = 1.0
    x_train = ((x_train - mean) / std).astype(np.float32)
    x_val = ((x_val - mean) / std).astype(np.float32)
    x_test = ((x_test - mean) / std).astype(np.float32)
    return x_train, x_val, x_test, mean, std


def _normalize_per_sample(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((x - mean) / std).astype(np.float32)


def _load_raw_dataset(
    cfg: Route1Config,
    dataset_name: Optional[str] = None,
    domains: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    读取原始样本数组（不做切分），可按需覆盖数据集名与域列表。
    """
    old_dataset = cfg.dataset_name
    old_domains = cfg.source_domains
    if dataset_name is not None:
        cfg.dataset_name = dataset_name
    if domains is not None:
        cfg.source_domains = domains

    try:
        name = cfg.dataset_name.upper()
        if name == "CWRU":
            x, y, d, domain_map = load_cwru(cfg)
        elif name == "JNU":
            x, y, d, domain_map = load_jnu(cfg)
        elif name in {"SELF_A", "SELF_B"}:
            x, y, d, domain_map = load_self_npz(cfg, dataset_name=name)
        else:
            raise ValueError(f"Unsupported dataset_name: {cfg.dataset_name}")
        return x, y, d, domain_map
    finally:
        cfg.dataset_name = old_dataset
        cfg.source_domains = old_domains


def build_dataloaders(cfg: Route1Config) -> DataBundle:
    """
    统一构建 DataLoader。

    设计意图：
    1) 将“数据集差异”收敛到这里，训练脚本只依赖统一接口；
    2) train/val 仅用 known classes，避免开集样本污染监督训练；
    3) test 保留全类别，用于同时评估 closed-set 与 open-set。
    """
    x, y, d, domain_map = _load_raw_dataset(cfg)

    # 将原始语义标签压缩为 [0..C-1]，用于分类头输出维度对齐。
    # 例如机械场景 known=[0,1,3] -> {0:0,1:1,3:2}
    known_map = {orig: i for i, orig in enumerate(cfg.known_classes)}
    y_mapped = _map_labels(y, known_map)

    # 分层划分保证每个类别在 train/val/test 中都有代表性。
    tr_idx, va_idx, te_idx = stratified_split_indices(y, cfg.train_ratio, cfg.val_ratio, cfg.seed)

    # train/val 只保留已知类，test 保留全类别用于开集评估
    tr_mask = y_mapped[tr_idx] >= 0
    va_mask = y_mapped[va_idx] >= 0

    x_train = x[tr_idx][tr_mask]
    x_val = x[va_idx][va_mask]
    x_test = x[te_idx]

    # Self datasets keep train-stat normalization (same protocol as previous self experiments).
    if cfg.dataset_name.upper() in {"SELF_A", "SELF_B"}:
        if str(cfg.self_norm_mode).lower() == "per_sample":
            x_train = _normalize_per_sample(x_train)
            x_val = _normalize_per_sample(x_val)
            x_test = _normalize_per_sample(x_test)
            cfg.self_use_saved_norm = False
            cfg.self_norm_mean = 0.0
            cfg.self_norm_std = 1.0
        else:
            x_train, x_val, x_test, mean, std = _normalize_by_train_stats(x_train, x_val, x_test)
            cfg.self_norm_mean = mean
            cfg.self_norm_std = std
            cfg.self_use_saved_norm = True
    else:
        cfg.self_use_saved_norm = False

    # Route1Dataset 中包含 pseudo_env 字段，后续训练时会在线更新。
    train_ds = Route1Dataset(x_train, y[tr_idx][tr_mask], y_mapped[tr_idx][tr_mask], d[tr_idx][tr_mask])
    val_ds = Route1Dataset(x_val, y[va_idx][va_mask], y_mapped[va_idx][va_mask], d[va_idx][va_mask])
    test_ds = Route1Dataset(x_test, y[te_idx], y_mapped[te_idx], d[te_idx])

    # train 使用 shuffle + drop_last 保证 batch 统计更稳定；
    # val/test 不打乱，保证评估可复现。
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    return DataBundle(train_loader, val_loader, test_loader, known_map, domain_map)


def build_eval_loader_full(
    cfg: Route1Config,
    dataset_name: Optional[str] = None,
    domains: Optional[List[str]] = None,
    known_classes: Optional[List[int]] = None,
) -> EvalBundle:
    """
    构建“全量样本”测试集 loader（不做 train/val/test 切分）。

    用途：
    - 同域 OOD（单域全部类别评估）
    - 跨域/跨数据集外测
    """
    x, y, d, domain_map = _load_raw_dataset(cfg, dataset_name=dataset_name, domains=domains)
    effective_dataset = (dataset_name if dataset_name is not None else cfg.dataset_name).upper()

    if effective_dataset in {"SELF_A", "SELF_B"}:
        if str(cfg.self_norm_mode).lower() == "per_sample":
            x = _normalize_per_sample(x)
        elif cfg.self_use_saved_norm:
            mean = float(cfg.self_norm_mean)
            std = float(cfg.self_norm_std)
            if std < 1e-8:
                std = 1.0
            x = ((x - mean) / std).astype(np.float32)
        else:
            mean = float(x.mean())
            std = float(x.std())
            if std < 1e-8:
                std = 1.0
            x = ((x - mean) / std).astype(np.float32)

    kcls = known_classes if known_classes is not None else cfg.known_classes
    known_map = {orig: i for i, orig in enumerate(kcls)}
    y_mapped = _map_labels(y, known_map)
    test_ds = Route1Dataset(x, y, y_mapped, d)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    return EvalBundle(test_loader=test_loader, known_map=known_map, domain_name_to_id=domain_map)
