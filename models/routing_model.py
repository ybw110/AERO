"""
路线一核心模型（教学版注释）。

模块组成：
1) Encoder: 提取潜在特征 z
2) Experts: M 个重构专家，分别尝试重构输入
3) Router: 通过重构误差得到 soft 路由概率 q
4) Classifier: 已知类分类头
5) Domain Discriminator + GRL: 对抗式环境不变学习

------------------------------------------------------------
关键公式
------------------------------------------------------------
给定样本 x_i：
1) z_i = Encoder(x_i)
2) \hat{x}_{i,m} = Expert_m(z_i)
3) e_{i,m} = MSE(x_i, \hat{x}_{i,m})
4) q_{i,m} = softmax(-e_{i,m} / T)
5) L_rec = mean_i sum_m q_{i,m} * e_{i,m}
------------------------------------------------------------
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReverseFn(torch.autograd.Function):
    """
    GRL（梯度反转层）。

    前向：y = x（恒等）
    反向：dL/dx = -alpha * dL/dy
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """对输入张量应用 GRL。"""
    return GradientReverseFn.apply(x, alpha)


class ConvEncoder1D(nn.Module):
    """
    一维卷积编码器。

    输入: [B, 1, L]
    输出: [B, D]
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).squeeze(-1)
        return self.proj(h)


class MLPDecoderExpert(nn.Module):
    """
    单个重构专家。

    输入: z [B, D]
    输出: reconstruction [B, L]
    """

    def __init__(self, latent_dim: int, seq_len: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 2 * latent_dim),
            nn.GELU(),
            nn.Linear(2 * latent_dim, seq_len),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ReconstructionRoutingModel(nn.Module):
    """
    路线一主模型。

    forward 输出字段：
    - z: 潜在特征 [B, D]
    - recons: 所有专家重构 [B, M, L]
    - mse_per_expert: 每个专家误差 [B, M]
    - routing_prob: 路由概率 [B, M]
    - class_logits: 分类输出 [B, C]
    - domain_logits: 域输出 [B, M]
    """

    def __init__(self, seq_len: int, latent_dim: int, m_max: int, num_known_classes: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.m_max = m_max

        self.encoder = ConvEncoder1D(latent_dim)
        self.experts = nn.ModuleList([MLPDecoderExpert(latent_dim, seq_len) for _ in range(m_max)])
        self.classifier = nn.Linear(latent_dim, num_known_classes)
        self.domain_disc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, m_max),
        )

    def forward(self, x: torch.Tensor, temp: float, grl_alpha: float) -> Dict[str, torch.Tensor]:
        # Step-1: 编码得到 z
        z = self.encoder(x)  # [B, D]

        # Step-2: M 个专家并行重构
        recons = torch.stack([e(z) for e in self.experts], dim=1)  # [B, M, L]

        # Step-3: 计算每个专家的重构误差
        target = x.squeeze(1)  # [B, L]
        mse = ((recons - target.unsqueeze(1)) ** 2).mean(dim=2)  # [B, M]

        # Step-4: 动态路由概率 q
        # temp 越小，q 越尖锐（更接近硬路由）
        routing_logits = -mse / max(temp, 1e-6)
        q = F.softmax(routing_logits, dim=1)  # [B, M]

        # Step-5: 混合重构（用于可视化/分析）
        mix_recon = (q.unsqueeze(-1) * recons).sum(dim=1)  # [B, L]

        # Step-6: 分类与对抗分支
        class_logits = self.classifier(z)  # [B, C]
        domain_logits = self.domain_disc(grad_reverse(z, grl_alpha))  # [B, M]

        return {
            "z": z,
            "recons": recons,
            "mix_recon": mix_recon,
            "mse_per_expert": mse,
            "routing_prob": q,
            "class_logits": class_logits,
            "domain_logits": domain_logits,
        }
