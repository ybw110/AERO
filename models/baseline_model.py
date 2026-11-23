"""
基线模型定义。

包含：
1) ClassifierNet: 仅分类头（ERM/JTT/CVAR/GROUPDRO）
2) DiversifyNet: 分类 + 对抗域分支（DIVERSIFY）
3) AMINetNet: 互信息引导单域泛化风格（简化统一版）
4) DACNNet: 对抗对比域扩展风格（简化统一版）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.routing_model import ConvEncoder1D, MLPDecoderExpert, grad_reverse


class ClassifierNet(nn.Module):
    """标准分类网络：Encoder + Classifier。"""

    def __init__(self, latent_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = ConvEncoder1D(latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        logits = self.classifier(z)
        return {"z": z, "class_logits": logits}


class DiversifyNet(nn.Module):
    """Diversify 风格基线：分类 + 对抗域分支。"""

    def __init__(self, latent_dim: int, num_classes: int, num_domains: int) -> None:
        super().__init__()
        self.encoder = ConvEncoder1D(latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)
        self.domain_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, num_domains),
        )

    def forward(self, x: torch.Tensor, grl_alpha: float = 0.0):
        z = self.encoder(x)
        class_logits = self.classifier(z)
        domain_logits = self.domain_head(grad_reverse(z, grl_alpha))
        return {"z": z, "class_logits": class_logits, "domain_logits": domain_logits}


class AFRDNNet(nn.Module):
    """
    AFRDN-style lightweight baseline:
    - shared encoder + classifier
    - reconstruction branch
    - pseudo-environment discriminator trained with GRL
    """

    def __init__(self, latent_dim: int, seq_len: int, num_classes: int, num_envs: int) -> None:
        super().__init__()
        self.encoder = ConvEncoder1D(latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)
        self.decoder = MLPDecoderExpert(latent_dim, seq_len)
        self.domain_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, num_envs),
        )

    def forward(self, x: torch.Tensor, grl_alpha: float = 0.0):
        z = self.encoder(x)
        class_logits = self.classifier(z)
        recon = self.decoder(z)
        domain_logits = self.domain_head(grad_reverse(z, grl_alpha))
        return {
            "z": z,
            "class_logits": class_logits,
            "recon": recon,
            "domain_logits": domain_logits,
        }


class _AMIDomainGenerator(nn.Module):
    """AMINet 风格域生成器（轻量化）。"""

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2))
        self.enc2 = nn.Sequential(nn.Conv1d(16, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.enc3 = nn.Sequential(nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2))
        self.dec1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="linear", align_corners=False), nn.Conv1d(64, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="linear", align_corners=False), nn.Conv1d(32, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ReLU())
        self.dec3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="linear", align_corners=False), nn.Conv1d(16, 1, 3, padding=1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        z = self.enc3(e2)
        d1 = self.dec1(z)
        d2 = self.dec2(d1 + e2)
        out = self.dec3(d2 + e1)
        return x + out


class _AMIFeatureExtractor(nn.Module):
    """AMINet 风格特征提取器。"""

    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=64, stride=1, padding=32),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 32, kernel_size=16, stride=1, padding=8),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.fc_proj = nn.Linear(64 * 64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.fc_proj(h)


class AMINetNet(nn.Module):
    """AMINet 统一接口版本。"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.gen = _AMIDomainGenerator()
        self.feat = _AMIFeatureExtractor(out_dim=256)
        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor):
        x_plus = self.gen(x)
        z = self.feat(x)
        z_plus = self.feat(x_plus)
        class_logits = self.classifier(z)
        class_logits_plus = self.classifier(z_plus)
        return {
            "z": z,
            "z_plus": z_plus,
            "class_logits": class_logits,
            "class_logits_plus": class_logits_plus,
        }


class _DACNFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(1, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 3, 1, 1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 128, 3, 1, 1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3(self.conv2(self.conv1(x)))


class _DACNAdaIN(nn.Module):
    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm1d(channels, affine=False)
        self.fc_gamma = nn.Sequential(nn.Linear(1, channels), nn.ReLU(), nn.Linear(channels, channels))
        self.fc_beta = nn.Sequential(nn.Linear(1, channels), nn.ReLU(), nn.Linear(channels, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        n1 = (0.05 + (1.95 - 0.05) * torch.rand(b, 1, device=x.device))
        n2 = torch.randn(b, 1, device=x.device)
        gamma = self.fc_gamma(n1).unsqueeze(2)
        beta = self.fc_beta(n2).unsqueeze(2)
        return self.norm(x) * gamma + beta


class _DACNInvariantExtractor(nn.Module):
    def __init__(self, in_dim: int = 128 * 128, out_dim: int = 256) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, out_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DACNNet(nn.Module):
    """DACN 统一接口版本（对抗分支使用 GRL 简化）。"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.F = _DACNFeatureExtractor()
        self.H = _DACNAdaIN(channels=128)
        self.G = _DACNInvariantExtractor(in_dim=128 * 128, out_dim=256)
        self.C = nn.Linear(256, num_classes)
        self.D = nn.Sequential(
            nn.Linear(256 * num_classes, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _op(z: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        prob = F.softmax(logits, dim=1)
        op = torch.bmm(prob.unsqueeze(2), z.unsqueeze(1))
        return op.view(op.size(0), -1)

    def forward(self, x: torch.Tensor, grl_alpha: float = 0.0):
        f_real = self.F(x)
        f_fake = self.H(f_real)

        z_real = self.G(f_real)
        z_fake = self.G(f_fake)
        logits_real = self.C(z_real)
        logits_fake = self.C(z_fake)

        op_real = self._op(z_real, logits_real)
        op_fake = self._op(z_fake, logits_fake)
        d_real = self.D(grad_reverse(op_real, grl_alpha))
        d_fake = self.D(grad_reverse(op_fake, grl_alpha))

        return {
            "z": z_real,
            "z_fake": z_fake,
            "class_logits": logits_real,
            "class_logits_fake": logits_fake,
            "domain_logits_real": d_real,
            "domain_logits_fake": d_fake,
        }
