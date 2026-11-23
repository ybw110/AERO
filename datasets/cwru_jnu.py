"""
CWRU / JNU 机械数据加载器。

特点：
1. 统一输出窗口序列 + 标签 + 域标签；
2. 支持“拼接多工况形成非平稳流”的训练场景；
3. 与路线一训练框架完全对齐。
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.io

from config import Route1Config


# CWRU 文件映射，标签: 0-Normal, 1-Inner, 2-Ball, 3-Outer
CWRU_FILE_MAP: Dict[str, Dict[int, List[str]]] = {
    "0HP": {0: ["97.mat"], 1: ["105.mat", "169.mat", "209.mat"], 2: ["118.mat", "185.mat", "222.mat"], 3: ["130.mat", "197.mat", "234.mat"]},
    "1HP": {0: ["98.mat"], 1: ["106.mat", "170.mat", "210.mat"], 2: ["119.mat", "186.mat", "223.mat"], 3: ["131.mat", "198.mat", "235.mat"]},
    "2HP": {0: ["99.mat"], 1: ["107.mat", "171.mat", "211.mat"], 2: ["120.mat", "187.mat", "224.mat"], 3: ["132.mat", "199.mat", "236.mat"]},
    "3HP": {0: ["100.mat"], 1: ["108.mat", "172.mat", "212.mat"], 2: ["121.mat", "188.mat", "225.mat"], 3: ["133.mat", "200.mat", "237.mat"]},
}


def _standardize(x: np.ndarray) -> np.ndarray:
    """标准化到零均值单位方差。"""
    mean = float(x.mean())
    std = float(x.std())
    if std < 1e-8:
        std = 1.0
    return ((x - mean) / std).astype(np.float32)


def _windowize(signal: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    """按滑动窗口切片。"""
    n = (len(signal) - seq_len) // stride + 1
    if n <= 0:
        return np.zeros((0, seq_len), dtype=np.float32)
    return np.stack([signal[i * stride : i * stride + seq_len] for i in range(n)], axis=0).astype(np.float32)


def _load_cwru_signal(path: str) -> np.ndarray:
    """读取 CWRU .mat 中的 DE_time 信号。"""
    mat = scipy.io.loadmat(path)
    for k in mat.keys():
        if k.endswith("DE_time"):
            return mat[k].flatten().astype(np.float32)
    raise ValueError(f"Cannot find DE_time in {path}")


def _load_jnu_signal(path: str) -> np.ndarray:
    """读取 JNU csv 最后一列振动信号。"""
    df = pd.read_csv(path, header=None)
    return df.iloc[:, -1].values.astype(np.float32)


def load_cwru(cfg: Route1Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    加载 CWRU 为统一格式：
    - x: [N, L]
    - y: [N]
    - domain_id: [N]
    """
    normal_root = os.path.join(cfg.cwru_root, "Normal Baseline Data")
    fault_root = os.path.join(cfg.cwru_root, "12k DriveEndBearingFault Data")
    # 在路线一中，每个负载档可视为一个真实环境（domain）。
    domains = cfg.source_domains or ["0HP", "1HP", "2HP", "3HP"]
    domain_map = {d: i for i, d in enumerate(domains)}

    xs, ys, ds = [], [], []
    for d in domains:
        file_map = CWRU_FILE_MAP[d]
        for y, files in sorted(file_map.items()):
            for f in files:
                root = normal_root if y == 0 else fault_root
                path = os.path.join(root, f)
                if not os.path.exists(path):
                    continue
                # 先标准化再切片，降低不同采集文件间幅值尺度差异。
                sig = _standardize(_load_cwru_signal(path))
                wins = _windowize(sig, cfg.sequence_length, cfg.stride)
                if len(wins) == 0:
                    continue
                xs.append(wins)
                ys.append(np.full(len(wins), y, dtype=np.int64))
                ds.append(np.full(len(wins), domain_map[d], dtype=np.int64))

    if not xs:
        raise FileNotFoundError("CWRU 数据为空，请检查路径和文件完整性。")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ds), domain_map


def load_jnu(cfg: Route1Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    加载 JNU 数据。
    文件命名约定:
    - 前缀 n/ib/tb/ob 对应标签 0/1/2/3
    - 文件名包含转速 600/800/1000
    """
    if not os.path.isdir(cfg.jnu_root):
        raise FileNotFoundError(f"JNU 目录不存在: {cfg.jnu_root}")

    # JNU 中转速可以自然视为不同运行工况（domain）。
    domains = cfg.source_domains or ["600", "800", "1000"]
    domain_map = {d: i for i, d in enumerate(domains)}
    prefix_to_y = {"n": 0, "ib": 1, "tb": 2, "ob": 3}

    all_files = os.listdir(cfg.jnu_root)
    xs, ys, ds = [], [], []
    for d in domains:
        for prefix, y in prefix_to_y.items():
            # 文件名中包含前缀(类别) + 转速(域)。
            matched = [f for f in all_files if f.startswith(prefix) and d in f and f.endswith(".csv")]
            for f in matched:
                path = os.path.join(cfg.jnu_root, f)
                sig = _standardize(_load_jnu_signal(path))
                wins = _windowize(sig, cfg.sequence_length, cfg.stride)
                if len(wins) == 0:
                    continue
                xs.append(wins)
                ys.append(np.full(len(wins), y, dtype=np.int64))
                ds.append(np.full(len(wins), domain_map[d], dtype=np.int64))

    if not xs:
        raise FileNotFoundError("JNU 数据为空，请检查路径和文件完整性。")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ds), domain_map
