"""Unified baseline trainer for protocol-aligned comparison."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from config import Route1Config
from datasets.builder import DataBundle
from utils.metrics import binary_aupr, binary_auroc, binary_fpr_at_tpr


def _unknown_score(logits: torch.Tensor, cfg: Route1Config) -> np.ndarray:
    score_type = str(getattr(cfg, "ood_score_type", "msp")).lower()
    if score_type == "energy":
        t = max(float(getattr(cfg, "ood_energy_temp", 1.0)), 1e-6)
        score = -t * torch.logsumexp(logits / t, dim=1)
    elif score_type == "maxlogit":
        score = -logits.max(dim=1).values
    else:
        prob = torch.softmax(logits, dim=1)
        score = 1.0 - prob.max(dim=1).values
    return score.detach().cpu().numpy()


def _numpy_kmeans(x: np.ndarray, k: int, n_iter: int = 20, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return np.zeros((0,), dtype=np.int64)

    k = max(1, min(k, n))
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)

    for _ in range(n_iter):
        d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = x[mask].mean(axis=0)
    return labels


def _supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    features = F.normalize(features, dim=1)
    logits = torch.matmul(features, features.T) / max(float(temperature), 1e-6)

    b = features.size(0)
    eye = torch.eye(b, device=features.device, dtype=torch.bool)
    labels = labels.view(-1, 1)
    pos = torch.eq(labels, labels.T) & (~eye)

    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~eye).float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)

    pos_count = pos.float().sum(dim=1)
    mean_log_prob_pos = (pos.float() * log_prob).sum(dim=1) / (pos_count + 1e-6)
    return -mean_log_prob_pos.mean()


class BaselineTrainer:
    """Single training/evaluation implementation for all baselines."""

    def __init__(self, model: nn.Module, cfg: Route1Config, device: str) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.method = cfg.baseline_method.upper()

        self._jtt_weights: Optional[np.ndarray] = None
        self._groupdro_q: Optional[torch.Tensor] = None
        self._eiil_env_weights: Optional[np.ndarray] = None

    def _forward(self, x: torch.Tensor):
        if self.method in {"DIVERSIFY", "DACN", "AFRDN", "WPA_PDG"}:
            return self.model(x, grl_alpha=0.0)
        return self.model(x)

    def _forward_with_grl(self, x: torch.Tensor, grl_alpha: float):
        if self.method in {"DIVERSIFY", "DACN", "AFRDN", "WPA_PDG"}:
            return self.model(x, grl_alpha=grl_alpha)
        return self.model(x)

    @torch.no_grad()
    def _cluster_from_feature_vectors(self, feature_blocks, index_blocks, k: int, seed_delta: int = 0) -> np.ndarray:
        feats = np.concatenate(feature_blocks) if feature_blocks else np.zeros((0, 1), dtype=np.float32)
        idxs = np.concatenate(index_blocks) if index_blocks else np.zeros((0,), dtype=np.int64)
        labels = _numpy_kmeans(feats, k=max(1, int(k)), n_iter=20, seed=int(self.cfg.seed + seed_delta))
        return labels.astype(np.int64), idxs.astype(np.int64)

    @torch.no_grad()
    def init_diversify_pseudo_domains(self, bundle: DataBundle) -> None:
        self.model.eval()
        feats, idxs = [], []
        for x, _, _, _, _, idx in bundle.train_loader:
            x = x.to(self.device)
            out = self._forward_with_grl(x, grl_alpha=0.0)
            feats.append(out["z"].detach().cpu().numpy())
            idxs.append(idx.numpy())
        labels, idx = self._cluster_from_feature_vectors(feats, idxs, k=self.cfg.diversify_k, seed_delta=1)
        bundle.train_loader.dataset.set_pseudo_env_by_index(labels, idx)

    @torch.no_grad()
    def init_wpa_pdg_pseudo_domains(self, bundle: DataBundle) -> None:
        # Same protocol as DIVERSIFY: generate pseudo environments from feature clustering.
        self.init_diversify_pseudo_domains(bundle)

    @torch.no_grad()
    def init_irm_pseudo_domains(self, bundle: DataBundle) -> None:
        self.model.eval()
        feats, idxs = [], []
        for x, _, _, _, _, idx in bundle.train_loader:
            x = x.to(self.device)
            out = self._forward_with_grl(x, grl_alpha=0.0)
            feats.append(out["z"].detach().cpu().numpy())
            idxs.append(idx.numpy())
        labels, idx = self._cluster_from_feature_vectors(feats, idxs, k=self.cfg.irm_num_env, seed_delta=2)
        bundle.train_loader.dataset.set_pseudo_env_by_index(labels, idx)

    @torch.no_grad()
    def init_afrdn_pseudo_domains(self, bundle: DataBundle) -> None:
        self.model.eval()
        feats, idxs = [], []
        for x, _, _, _, _, idx in bundle.train_loader:
            x = x.to(self.device)
            out = self._forward_with_grl(x, grl_alpha=0.0)
            target = x.squeeze(1)
            rec = out["recon"]
            rec_err = ((rec - target) ** 2).mean(dim=1, keepdim=True)
            feat = torch.cat([out["z"], rec_err], dim=1)
            feats.append(feat.detach().cpu().numpy())
            idxs.append(idx.numpy())
        labels, idx = self._cluster_from_feature_vectors(feats, idxs, k=self.cfg.afrdn_num_env, seed_delta=3)
        bundle.train_loader.dataset.set_pseudo_env_by_index(labels, idx)

    @torch.no_grad()
    def prepare_eiil_weights(self, bundle: DataBundle) -> None:
        self.model.eval()
        loss_vec, idx_vec = [], []
        for x, y, _, _, _, idx in bundle.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            out = self._forward_with_grl(x, grl_alpha=0.0)
            per = F.cross_entropy(out["class_logits"], y, reduction="none")
            loss_vec.append(per.detach().cpu().numpy().reshape(-1, 1))
            idx_vec.append(idx.numpy())

        labels, idxs = self._cluster_from_feature_vectors(loss_vec, idx_vec, k=self.cfg.eiil_num_env, seed_delta=4)
        bundle.train_loader.dataset.set_pseudo_env_by_index(labels, idxs)

        # Group-wise reweighting: harder inferred environment gets larger weight.
        env_losses = np.zeros(max(1, int(self.cfg.eiil_num_env)), dtype=np.float32)
        env_counts = np.zeros_like(env_losses)
        all_losses = np.concatenate(loss_vec, axis=0).reshape(-1)
        for l, e in zip(all_losses, labels):
            env_losses[int(e)] += float(l)
            env_counts[int(e)] += 1.0
        env_means = env_losses / np.maximum(env_counts, 1.0)
        if float(env_means.mean()) > 0:
            norm = env_means / float(env_means.mean())
        else:
            norm = np.ones_like(env_means)
        self._eiil_env_weights = (1.0 + (self.cfg.eiil_reweight - 1.0) * norm).astype(np.float32)

    def _erm_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, y)

    def _cvar_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        per = F.cross_entropy(logits, y, reduction="none")
        k = max(1, int(np.ceil(len(per) * self.cfg.cvar_alpha)))
        topk = torch.topk(per, k=k, largest=True).values
        return topk.mean()

    def _groupdro_loss(self, logits: torch.Tensor, y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        unique_groups = torch.unique(g)
        group_losses = []
        for gid in unique_groups:
            mask = g == gid
            if mask.any():
                group_losses.append(F.cross_entropy(logits[mask], y[mask]))

        if not group_losses:
            return F.cross_entropy(logits, y)

        gl = torch.stack(group_losses)
        if self._groupdro_q is None or len(self._groupdro_q) != len(gl):
            self._groupdro_q = torch.ones(len(gl), device=gl.device) / len(gl)

        self._groupdro_q = self._groupdro_q * torch.exp(self.cfg.groupdro_eta * gl.detach())
        self._groupdro_q = self._groupdro_q / self._groupdro_q.sum()
        return torch.sum(self._groupdro_q * gl)

    def _irm_penalty(self, logits: torch.Tensor, y: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        unique_groups = torch.unique(groups)
        if len(unique_groups) == 0:
            return torch.tensor(0.0, device=logits.device)
        scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
        penalties = []
        for gid in unique_groups:
            mask = groups == gid
            if int(mask.sum().item()) < 2:
                continue
            g_loss = F.cross_entropy(logits[mask] * scale, y[mask])
            grad = torch.autograd.grad(g_loss, [scale], create_graph=True)[0]
            penalties.append(torch.sum(grad**2))
        if not penalties:
            return torch.tensor(0.0, device=logits.device)
        return torch.stack(penalties).mean()

    def _mixup_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        alpha = max(float(self.cfg.mixup_alpha), 1e-6)
        lam = float(np.random.beta(alpha, alpha))
        perm = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1.0 - lam) * x[perm]
        out = self.model(mixed_x)
        logits = out["class_logits"]
        return lam * F.cross_entropy(logits, y) + (1.0 - lam) * F.cross_entropy(logits, y[perm])

    def _mixstyle_loss(self, out: Dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        cls_main = F.cross_entropy(out["class_logits"], y)
        if np.random.rand() > float(self.cfg.mixstyle_prob):
            return cls_main
        z = out["z"]
        if z.size(0) < 2:
            return cls_main
        alpha = max(float(self.cfg.mixstyle_alpha), 1e-6)
        lam = float(np.random.beta(alpha, alpha))
        perm = torch.randperm(z.size(0), device=z.device)
        z_mix = lam * z + (1.0 - lam) * z[perm]
        logits_mix = self.model.classifier(z_mix)
        cls_mix = F.cross_entropy(logits_mix, y)
        return 0.5 * (cls_main + cls_mix)

    def _wpa_augment(self, x: torch.Tensor) -> torch.Tensor:
        # Lightweight wavelet-packet-style augmentation via random frequency sub-band gains.
        fft_x = torch.fft.rfft(x.squeeze(1), dim=1)
        b, nfreq = fft_x.shape
        n_bands = 4
        band = max(1, nfreq // n_bands)
        gains = torch.ones((b, nfreq), device=x.device, dtype=x.dtype)
        for k in range(n_bands):
            left = k * band
            right = nfreq if k == (n_bands - 1) else min(nfreq, (k + 1) * band)
            if left >= right:
                continue
            if float(torch.rand(1).item()) < 0.7:
                g = torch.empty((b, 1), device=x.device, dtype=x.dtype).uniform_(0.6, 1.4)
                gains[:, left:right] = g
        aug = torch.fft.irfft(fft_x * gains.to(fft_x.dtype), n=x.size(-1), dim=1)
        return aug.unsqueeze(1)

    def train_one_epoch(self, bundle: DataBundle, optimizer: torch.optim.Optimizer, epoch: int = 1) -> Dict[str, float]:
        self.model.train()
        meter = {
            "loss": 0.0,
            "cls": 0.0,
            "adv": 0.0,
            "cvar": 0.0,
            "groupdro": 0.0,
            "jtt_weighted": 0.0,
            "supcon": 0.0,
            "rec": 0.0,
            "irm": 0.0,
            "mix": 0.0,
        }
        steps = 0

        for x, y, d, pseudo_env, _, idx in bundle.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            d = d.to(self.device)
            pseudo_env = pseudo_env.to(self.device)
            idx_np = idx.numpy()

            if self.method == "DIVERSIFY":
                out = self._forward_with_grl(x, grl_alpha=self.cfg.diversify_grl_alpha)
                cls_loss = F.cross_entropy(out["class_logits"], y)
                adv_loss = F.cross_entropy(out["domain_logits"], pseudo_env)
                loss = cls_loss + self.cfg.diversify_adv_weight * adv_loss
                meter["cls"] += float(cls_loss.item())
                meter["adv"] += float(adv_loss.item())
            elif self.method == "WPA_PDG":
                x_aug = self._wpa_augment(x)
                out_main = self._forward_with_grl(x, grl_alpha=self.cfg.diversify_grl_alpha)
                out_aug = self._forward_with_grl(x_aug, grl_alpha=self.cfg.diversify_grl_alpha)
                cls_main = F.cross_entropy(out_main["class_logits"], y)
                cls_aug = F.cross_entropy(out_aug["class_logits"], y)
                adv_loss = F.cross_entropy(out_main["domain_logits"], pseudo_env)
                loss = cls_main + self.cfg.wpa_aug_weight * cls_aug + self.cfg.wpa_adv_weight * adv_loss
                meter["cls"] += float((0.5 * (cls_main + cls_aug)).item())
                meter["adv"] += float(adv_loss.item())
                meter["mix"] += float(cls_aug.item())
            elif self.method == "AFRDN":
                out = self._forward_with_grl(x, grl_alpha=self.cfg.diversify_grl_alpha)
                cls_loss = F.cross_entropy(out["class_logits"], y)
                rec_loss = F.mse_loss(out["recon"], x.squeeze(1))
                adv_loss = F.cross_entropy(out["domain_logits"], pseudo_env)
                loss = cls_loss + self.cfg.afrdn_rec_weight * rec_loss + self.cfg.afrdn_adv_weight * adv_loss
                meter["cls"] += float(cls_loss.item())
                meter["rec"] += float(rec_loss.item())
                meter["adv"] += float(adv_loss.item())
            elif self.method == "AMINET":
                out = self.model(x)
                cls_main = F.cross_entropy(out["class_logits"], y)
                cls_plus = F.cross_entropy(out["class_logits_plus"], y)
                z_cat = torch.cat([out["z"], out["z_plus"]], dim=0)
                y_cat = torch.cat([y, y], dim=0)
                supcon = _supcon_loss(z_cat, y_cat)
                loss = cls_main + cls_plus + self.cfg.aminet_supcon_weight * supcon
                meter["cls"] += float((0.5 * (cls_main + cls_plus)).item())
                meter["supcon"] += float(supcon.item())
            elif self.method == "DACN":
                out = self._forward_with_grl(x, grl_alpha=self.cfg.diversify_grl_alpha)
                cls_real = F.cross_entropy(out["class_logits"], y)
                cls_fake = F.cross_entropy(out["class_logits_fake"], y)

                z_cat = torch.cat([out["z"], out["z_fake"]], dim=0)
                y_cat = torch.cat([y, y], dim=0)
                supcon = _supcon_loss(z_cat, y_cat)

                d_real = out["domain_logits_real"].view(-1)
                d_fake = out["domain_logits_fake"].view(-1)
                label_real = torch.ones_like(d_real)
                label_fake = torch.zeros_like(d_fake)
                adv = F.binary_cross_entropy(d_real, label_real) + F.binary_cross_entropy(d_fake, label_fake)

                loss = cls_real + cls_fake + self.cfg.dacn_supcon_weight * supcon + self.cfg.dacn_adv_weight * adv
                meter["cls"] += float((0.5 * (cls_real + cls_fake)).item())
                meter["supcon"] += float(supcon.item())
                meter["adv"] += float(adv.item())
            elif self.method == "MIXUP":
                loss = self._mixup_loss(x, y)
                meter["cls"] += float(loss.item())
                meter["mix"] += float(loss.item())
            else:
                out = self._forward(x)
                logits = out["class_logits"]

                if self.method == "ERM":
                    loss = self._erm_loss(logits, y)
                    meter["cls"] += float(loss.item())
                elif self.method == "MIXSTYLE":
                    loss = self._mixstyle_loss(out, y)
                    meter["cls"] += float(loss.item())
                    meter["mix"] += float(loss.item())
                elif self.method == "CVAR_DRO":
                    per = F.cross_entropy(logits, y, reduction="none")
                    cvar = self._cvar_loss(logits, y)
                    loss = cvar
                    meter["cls"] += float(per.mean().item())
                    meter["cvar"] += float(cvar.item())
                elif self.method == "GROUPDRO":
                    per = F.cross_entropy(logits, y, reduction="none")
                    gd = self._groupdro_loss(logits, y, d)
                    loss = gd
                    meter["cls"] += float(per.mean().item())
                    meter["groupdro"] += float(gd.item())
                elif self.method == "JTT":
                    per = F.cross_entropy(logits, y, reduction="none")
                    if self._jtt_weights is None:
                        w = torch.ones_like(per)
                    else:
                        w = torch.from_numpy(self._jtt_weights[idx_np]).to(per.device, dtype=per.dtype)
                    loss = (per * w).mean()
                    meter["cls"] += float(per.mean().item())
                    meter["jtt_weighted"] += float(loss.item())
                elif self.method == "IRM":
                    cls = F.cross_entropy(logits, y)
                    irm = self._irm_penalty(logits, y, pseudo_env)
                    loss = cls + self.cfg.irm_penalty_weight * irm
                    meter["cls"] += float(cls.item())
                    meter["irm"] += float(irm.item())
                elif self.method == "EIIL":
                    per = F.cross_entropy(logits, y, reduction="none")
                    if self._eiil_env_weights is None:
                        loss = per.mean()
                    else:
                        w_np = self._eiil_env_weights.astype(np.float32)
                        env_np = pseudo_env.detach().cpu().numpy().astype(np.int64)
                        env_np = np.clip(env_np, 0, len(w_np) - 1)
                        w = torch.from_numpy(w_np[env_np]).to(per.device, dtype=per.dtype)
                        loss = (per * w).mean()
                        meter["groupdro"] += float(w.mean().item())
                    meter["cls"] += float(per.mean().item())
                else:
                    raise ValueError(f"Unsupported method: {self.method}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            optimizer.step()

            meter["loss"] += float(loss.item())
            steps += 1

        for k in meter:
            meter[k] /= max(steps, 1)
        return meter

    @torch.no_grad()
    def prepare_jtt_weights(self, bundle: DataBundle) -> None:
        self.model.eval()
        n = len(bundle.train_loader.dataset)
        self._jtt_weights = np.ones(n, dtype=np.float32)

        for x, y, _, _, _, idx in bundle.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            out = self._forward_with_grl(x, grl_alpha=0.0)
            pred = out["class_logits"].argmax(dim=1)
            err = (pred != y).detach().cpu().numpy().astype(np.float32)
            idx_np = idx.numpy()
            self._jtt_weights[idx_np] = 1.0 + err * (self.cfg.jtt_upweight - 1.0)

    @torch.no_grad()
    def eval_known_acc(self, loader) -> float:
        self.model.eval()
        correct, total = 0, 0
        for x, y, _, _, _, _ in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            out = self._forward_with_grl(x, grl_alpha=0.0)
            pred = out["class_logits"].argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
        return float(correct / max(total, 1))

    @torch.no_grad()
    def eval_open_set(self, loader) -> Dict[str, np.ndarray]:
        self.model.eval()
        scores, unknown_flags = [], []
        known_correct, known_total = 0, 0

        for x, y_mapped, _, _, _, _ in loader:
            x = x.to(self.device)
            y_mapped = y_mapped.to(self.device)
            out = self._forward_with_grl(x, grl_alpha=0.0)
            logits = out["class_logits"]

            scores.append(_unknown_score(logits, self.cfg))
            unknown_flags.append((y_mapped.cpu().numpy() < 0).astype(np.int64))

            known_mask = y_mapped >= 0
            if known_mask.any():
                pred = logits[known_mask].argmax(dim=1)
                gt = y_mapped[known_mask]
                known_correct += int((pred == gt).sum().item())
                known_total += int(gt.numel())

        scores = np.concatenate(scores) if scores else np.array([])
        unknown_flags = np.concatenate(unknown_flags) if unknown_flags else np.array([])

        auroc = float("nan")
        aupr = float("nan")
        fpr95 = float("nan")
        if len(np.unique(unknown_flags)) == 2:
            auroc = binary_auroc(unknown_flags, scores)
            aupr = binary_aupr(unknown_flags, scores)
            fpr95 = binary_fpr_at_tpr(unknown_flags, scores, target_tpr=0.95)

        return {
            "test_known_acc": float(known_correct / max(known_total, 1)),
            "test_open_auroc": auroc,
            "test_open_aupr": aupr,
            "test_open_fpr95": fpr95,
            "scores": scores,
            "is_unknown": unknown_flags,
        }
