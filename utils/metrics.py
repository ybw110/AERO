"""
评估指标工具（纯 numpy，无 sklearn 依赖）：
1. 准确率
2. AUROC
3. AUPR（正类为 unknown）
4. FPR95（TPR=95% 时的 FPR）
"""

import numpy as np


def accuracy_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    """根据 logits 和真实标签计算准确率。"""
    if len(y_true) == 0:
        return 0.0
    pred = logits.argmax(axis=1)
    return float((pred == y_true).mean())


def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    计算二分类 AUROC。

    参数:
    - y_true: 0/1 标签，1 表示正类（这里通常表示 unknown）
    - y_score: 分数，越大越偏向正类
    """
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos_rank_sum = ranks[y_true == 1].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _binary_pr_curve(y_true: np.ndarray, y_score: np.ndarray):
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted == 1).astype(np.float64)
    fp = np.cumsum(y_sorted == 0).astype(np.float64)
    n_pos = float((y_true == 1).sum())
    if n_pos <= 0:
        return np.array([1.0]), np.array([0.0])
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(n_pos, 1e-12)

    # 补一个起点：recall=0, precision=1
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return precision, recall


def binary_aupr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    AUPR（unknown 作为正类）。
    """
    y_true = y_true.astype(np.int64)
    if (y_true == 1).sum() == 0 or (y_true == 0).sum() == 0:
        return float("nan")
    precision, recall = _binary_pr_curve(y_true, y_score)
    # recall 单调不减，直接梯形积分
    aupr = np.trapz(precision, recall)
    return float(aupr)


def binary_fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, target_tpr: float = 0.95) -> float:
    """
    FPR@TPR（默认 FPR95）。
    y_true=1 为正类（unknown），score 越大越偏正类。
    """
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    n_pos = float((y_true == 1).sum())
    n_neg = float((y_true == 0).sum())
    if n_pos <= 0 or n_neg <= 0:
        return float("nan")

    # 所有唯一阈值（降序）+ 末端
    thresholds = np.unique(y_score)[::-1]
    best = float("nan")
    for th in thresholds:
        pred_pos = y_score >= th
        tp = float(np.sum((pred_pos == 1) & (y_true == 1)))
        fp = float(np.sum((pred_pos == 1) & (y_true == 0)))
        tpr = tp / n_pos
        fpr = fp / n_neg
        if tpr >= target_tpr:
            best = fpr
            break

    if np.isnan(best):
        # 达不到目标 tpr，按最宽松阈值近似（全判正）
        return 1.0
    return float(best)
