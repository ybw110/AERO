"""
统一样本数据结构定义。

每个样本返回字段:
1. x: 输入信号 [1, L]
2. y_mapped: 训练标签（已知类映射后）
3. domain_id: 真实域标签（如负载档位、记录ID）
4. pseudo_env: 动态路由推断的伪环境标签（可在线更新）
5. y_original: 原始标签（用于开集评估）
6. index: 样本索引
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class Route1Dataset(Dataset):
    """
    路线一统一数据集封装（教学版）。

    为什么不用“只返回 x,y”：
    - 训练阶段需要 y_mapped（已知类映射标签）；
    - 分析阶段需要 y_original（原始语义标签）；
    - 路由阶段需要 pseudo_env（可在线更新）；
    - 可视化阶段需要 domain_id（真实域标签）和 index（全局索引）。
    """

    def __init__(
        self,
        x: np.ndarray,
        y_original: np.ndarray,
        y_mapped: np.ndarray,
        domain_id: np.ndarray,
    ) -> None:
        self.x = x.astype(np.float32)
        self.y_original = y_original.astype(np.int64)
        self.y_mapped = y_mapped.astype(np.int64)
        self.domain_id = domain_id.astype(np.int64)

        # 伪环境标签初始全 0，训练中会被算法迭代更新。
        self.pseudo_env = np.zeros(len(self.x), dtype=np.int64)
        self.indices = np.arange(len(self.x), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        # 输出顺序在整个项目中保持固定，避免 train/test 脚本错位读取。
        return (
            torch.from_numpy(self.x[idx]).unsqueeze(0),
            torch.tensor(self.y_mapped[idx], dtype=torch.long),
            torch.tensor(self.domain_id[idx], dtype=torch.long),
            torch.tensor(self.pseudo_env[idx], dtype=torch.long),
            torch.tensor(self.y_original[idx], dtype=torch.long),
            torch.tensor(self.indices[idx], dtype=torch.long),
        )

    def set_pseudo_env_by_index(self, labels: np.ndarray, indices: np.ndarray) -> None:
        """根据全局索引更新伪环境标签。"""
        labels = np.asarray(labels)
        indices = np.asarray(indices)
        for i, idx in enumerate(indices):
            idx = int(idx)
            if 0 <= idx < len(self.pseudo_env):
                self.pseudo_env[idx] = int(labels[i])
