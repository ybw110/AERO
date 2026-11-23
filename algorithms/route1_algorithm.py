from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from config import Route1Config
from datasets.builder import DataBundle
from utils.metrics import binary_aupr, binary_auroc, binary_fpr_at_tpr


class Route1Algorithm:
    """
    Route1 (current version):
    - Unlabeled environment inference via reconstruction routing.
    - Alternating optimization in joint stage:
      (1) EI step: optimize routing/reconstruction objectives (natural collapse enabled)
      (2) Worst step: optimize domain discriminator with pseudo environments
      (3) Invariant step: DANN update on feature/classifier with pseudo environments
    - OOD score supports MSP / Energy / MaxLogit.
    """

    def __init__(self, model: nn.Module, cfg: Route1Config, device: str) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device

        self.ema_mse = None
        self._stage_cache = None

    def _set_head_trainable(self, enable: bool) -> None:
        for p in self.model.classifier.parameters():
            p.requires_grad = enable
        for p in self.model.domain_disc.parameters():
            p.requires_grad = enable

    def _set_stage(self, epoch: int) -> bool:
        pretrain = epoch <= self.cfg.route1_pretrain_epochs
        if self._stage_cache != pretrain:
            self._set_head_trainable(enable=not pretrain)
            self._stage_cache = pretrain
        return pretrain

    def _set_requires_grad(self, modules: Iterable[nn.Module], enable: bool) -> None:
        for m in modules:
            for p in m.parameters():
                p.requires_grad = enable

    def _scheduled_keep_k(self, epoch: int) -> int:
        if epoch < self.cfg.sparsify_start_epoch:
            return self.cfg.m_max
        start = self.cfg.sparsify_start_epoch
        end = max(self.cfg.sparsify_end_epoch, start + 1)
        t = min(max((epoch - start) / (end - start), 0.0), 1.0)
        k = round(self.cfg.m_max - t * (self.cfg.m_max - self.cfg.target_k))
        return int(max(self.cfg.target_k, min(self.cfg.m_max, k)))

    def _active_mask_from_ema(self, keep_k: int, device: torch.device) -> torch.Tensor:
        if self.ema_mse is None:
            return torch.ones(self.cfg.m_max, device=device)
        order = np.argsort(self.ema_mse)
        active = order[:keep_k]
        mask_np = np.zeros(self.cfg.m_max, dtype=np.float32)
        mask_np[active] = 1.0
        return torch.from_numpy(mask_np).to(device)

    @staticmethod
    def _msp_unknown_score(logits: torch.Tensor) -> np.ndarray:
        prob = F.softmax(logits, dim=1)
        score = 1.0 - prob.max(dim=1).values
        return score.detach().cpu().numpy()

    @staticmethod
    def _energy_unknown_score(logits: torch.Tensor, temp: float) -> np.ndarray:
        t = max(float(temp), 1e-6)
        score = -t * torch.logsumexp(logits / t, dim=1)
        return score.detach().cpu().numpy()

    @staticmethod
    def _maxlogit_unknown_score(logits: torch.Tensor) -> np.ndarray:
        # MaxLogit is an ID score (higher=more ID). We negate it into unknownness.
        score = -logits.max(dim=1).values
        return score.detach().cpu().numpy()

    def _unknown_score(self, logits: torch.Tensor) -> np.ndarray:
        score_type = str(getattr(self.cfg, "ood_score_type", "msp")).lower()
        if score_type == "energy":
            return self._energy_unknown_score(logits, getattr(self.cfg, "ood_energy_temp", 1.0))
        if score_type == "maxlogit":
            return self._maxlogit_unknown_score(logits)
        return self._msp_unknown_score(logits)

    def train_one_epoch(
        self,
        bundle: DataBundle,
        optimizer: torch.optim.Optimizer,
        epoch: int = 1,
    ) -> Tuple[Dict[str, float], np.ndarray]:
        self.model.train()
        pretrain = self._set_stage(epoch)

        meter = {
            "loss": 0.0,
            "cls": 0.0,
            "rec": 0.0,
            "adv": 0.0,
            "bal": 0.0,
            "ent": 0.0,
            "col": 0.0,
            "tmp": 0.0,
        }
        steps = 0

        usage_sum = np.zeros(self.cfg.m_max, dtype=np.float64)
        all_idx, all_env = [], []

        disable_natural = bool(getattr(self.cfg, "ablation_disable_natural_collapse", False))
        disable_conf_weight = bool(getattr(self.cfg, "ablation_disable_confidence_weighting", False))
        disable_worst_step = bool(getattr(self.cfg, "ablation_disable_worst_step", False))
        disable_recon_routing = bool(getattr(self.cfg, "ablation_disable_reconstruction_routing", False))
        soft_only_routing = bool(getattr(self.cfg, "ablation_soft_only_routing", False))
        equal_capacity_no_env = bool(getattr(self.cfg, "ablation_equal_capacity_no_env", False))

        progress = min(max((epoch - 1) / max(self.cfg.epochs - 1, 1), 0.0), 1.0)
        conf_thr = self.cfg.conf_threshold_start + (self.cfg.conf_threshold_end - self.cfg.conf_threshold_start) * progress
        hard_after = max(self.cfg.routing_hard_after_epoch, self.cfg.route1_pretrain_epochs + 2)

        # Progressive decay for balance regularization to support natural collapse.
        decay_progress = min(max((epoch - 1) / max(self.cfg.balance_decay_epochs, 1), 0.0), 1.0)
        bal_factor = max(self.cfg.balance_min_factor, 1.0 - decay_progress)
        if equal_capacity_no_env:
            balance_weight = 0.0
            collapse_weight = 0.0
        elif disable_natural:
            balance_weight = self.cfg.loss_w_balance
            collapse_weight = 0.0
        else:
            balance_weight = self.cfg.loss_w_balance * bal_factor
            collapse_weight = self.cfg.loss_w_collapse

        feat_modules = [self.model.encoder, self.model.experts]
        cls_module = [self.model.classifier]
        dis_module = [self.model.domain_disc]

        for x, y, _, _, _, idx in bundle.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            idx = idx.to(self.device)
            idx_np = idx.detach().cpu().numpy()

            # ===== Step 1: Environment Inference (EI) =====
            self._set_requires_grad(feat_modules, True)
            self._set_requires_grad(cls_module, False)
            self._set_requires_grad(dis_module, False)

            out_ei = self.model(x, temp=self.cfg.routing_temp, grl_alpha=0.0)
            mse = out_ei["mse_per_expert"]

            if equal_capacity_no_env:
                q = torch.full((x.size(0), self.cfg.m_max), 1.0 / self.cfg.m_max, device=x.device, dtype=mse.dtype)
                train_q = q
            elif disable_recon_routing:
                q = torch.full((x.size(0), self.cfg.m_max), 1.0 / self.cfg.m_max, device=x.device, dtype=mse.dtype)
                rand_assign = torch.randint(0, self.cfg.m_max, (x.size(0),), device=x.device)
                train_q = F.one_hot(rand_assign, num_classes=self.cfg.m_max).float()
            else:
                batch_mse = mse.detach().mean(dim=0).cpu().numpy()
                if self.ema_mse is None:
                    self.ema_mse = batch_mse.copy()
                else:
                    m = self.cfg.ema_mse_momentum
                    self.ema_mse = m * self.ema_mse + (1.0 - m) * batch_mse

                if disable_natural:
                    active_mask = torch.ones(self.cfg.m_max, device=mse.device)
                else:
                    keep_k = self._scheduled_keep_k(epoch)
                    active_mask = self._active_mask_from_ema(keep_k, mse.device)

                masked_logits = -mse / max(self.cfg.routing_temp, 1e-6)
                masked_logits = masked_logits + torch.log(active_mask.unsqueeze(0) + 1e-8)
                q = F.softmax(masked_logits, dim=1)

                if soft_only_routing:
                    train_q = q
                elif epoch >= hard_after:
                    noise_scale = self.cfg.routing_noise_std / max(float(epoch), 1.0)
                    noisy_mse = mse + noise_scale * torch.randn_like(mse)
                    noisy_mse = noisy_mse + (1.0 - active_mask.unsqueeze(0)) * 1e6
                    assign = noisy_mse.argmin(dim=1)
                    train_q = F.one_hot(assign, num_classes=self.cfg.m_max).float()
                else:
                    train_q = q

            mean_q = train_q.mean(dim=0)
            pseudo_env = train_q.argmax(dim=1)

            rec_loss = (train_q * mse).sum(dim=1).mean()
            if equal_capacity_no_env or disable_recon_routing:
                bal_loss = torch.tensor(0.0, device=self.device)
                route_entropy = torch.tensor(0.0, device=self.device)
                collapse_reg = torch.tensor(0.0, device=self.device)
            else:
                bal_loss = ((mean_q - (1.0 / self.cfg.m_max)) ** 2).sum()
                route_entropy = -(q * torch.log(q + 1e-8)).sum(dim=1).mean()
                collapse_reg = torch.sum(torch.sqrt(mean_q + 1e-8))

            cls_proxy = torch.tensor(0.0, device=self.device)
            if equal_capacity_no_env:
                cls_proxy = F.cross_entropy(out_ei["class_logits"], y)
                loss_ei = self.cfg.loss_w_rec * rec_loss + self.cfg.loss_w_cls * cls_proxy
            else:
                loss_ei = (
                    self.cfg.loss_w_rec * rec_loss
                    + balance_weight * bal_loss
                    + self.cfg.loss_w_entropy * route_entropy
                    + collapse_weight * collapse_reg
                )

            optimizer.zero_grad()
            loss_ei.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            optimizer.step()

            cls_loss = torch.tensor(0.0, device=self.device)
            worst_loss = torch.tensor(0.0, device=self.device)
            inv_adv_loss = torch.tensor(0.0, device=self.device)
            loss_inv = torch.tensor(0.0, device=self.device)
            if equal_capacity_no_env:
                cls_loss = cls_proxy.detach()

            # ===== Step 2 & 3: Alternating worst/invariant updates (joint only) =====
            if not pretrain:
                if equal_capacity_no_env:
                    # Keep three-step computational structure, but remove pseudo-env inference and invariance learning.
                    self._set_requires_grad(feat_modules, False)
                    self._set_requires_grad(cls_module, False)
                    self._set_requires_grad(dis_module, True)

                    out_worst = self.model(x, temp=self.cfg.routing_temp, grl_alpha=0.0)
                    dummy_env = torch.zeros_like(pseudo_env)
                    worst_loss = F.cross_entropy(out_worst["domain_logits"], dummy_env)

                    optimizer.zero_grad()
                    worst_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    optimizer.step()

                    self._set_requires_grad(feat_modules, True)
                    self._set_requires_grad(cls_module, True)
                    self._set_requires_grad(dis_module, False)

                    out_inv = self.model(x, temp=self.cfg.routing_temp, grl_alpha=0.0)
                    cls_loss = F.cross_entropy(out_inv["class_logits"], y)
                    inv_adv_loss = torch.tensor(0.0, device=self.device)
                    loss_inv = self.cfg.loss_w_cls * cls_loss

                    optimizer.zero_grad()
                    loss_inv.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    optimizer.step()
                else:
                    # Step 2: worst-case domain characterization (update D only).
                    if not disable_worst_step:
                        self._set_requires_grad(feat_modules, False)
                        self._set_requires_grad(cls_module, False)
                        self._set_requires_grad(dis_module, True)

                        out_worst = self.model(x, temp=self.cfg.routing_temp, grl_alpha=0.0)
                        worst_loss = F.cross_entropy(out_worst["domain_logits"], pseudo_env)

                        optimizer.zero_grad()
                        worst_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                        optimizer.step()

                    # Step 3: invariant learning (update F/C with fixed D under GRL).
                    self._set_requires_grad(feat_modules, True)
                    self._set_requires_grad(cls_module, True)
                    self._set_requires_grad(dis_module, False)

                    out_inv = self.model(x, temp=self.cfg.routing_temp, grl_alpha=self.cfg.grl_alpha)
                    cls_loss = F.cross_entropy(out_inv["class_logits"], y)

                    conf = q.max(dim=1).values
                    if disable_conf_weight:
                        conf_w = torch.ones_like(conf)
                    else:
                        conf_w = torch.clamp((conf - conf_thr) / max(1.0 - conf_thr, 1e-6), min=0.0)
                        if float(conf_w.sum().item()) < 1e-8:
                            conf_w = torch.ones_like(conf_w)

                    ce_per = F.cross_entropy(out_inv["domain_logits"], pseudo_env, reduction="none")
                    inv_adv_loss = (ce_per * conf_w.detach()).sum() / (conf_w.detach().sum() + 1e-8)

                    loss_inv = self.cfg.loss_w_cls * cls_loss + self.cfg.loss_w_adv * inv_adv_loss

                    optimizer.zero_grad()
                    loss_inv.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    optimizer.step()

            # Restore all trainable for next batch bookkeeping/forward safety.
            self._set_requires_grad(feat_modules, True)
            self._set_requires_grad(cls_module, True)
            self._set_requires_grad(dis_module, True)

            usage_sum += mean_q.detach().cpu().numpy()
            all_idx.append(idx_np)
            all_env.append(pseudo_env.detach().cpu().numpy())

            total_loss = loss_ei + worst_loss + loss_inv
            meter["loss"] += float(total_loss.item())
            meter["cls"] += float(cls_loss.item())
            meter["rec"] += float(rec_loss.item())
            meter["adv"] += float((worst_loss + inv_adv_loss).item())
            meter["bal"] += float(bal_loss.item())
            meter["ent"] += float(route_entropy.item())
            meter["col"] += float(collapse_reg.item())
            meter["tmp"] += 0.0
            steps += 1

        if all_idx:
            bundle.train_loader.dataset.set_pseudo_env_by_index(np.concatenate(all_env), np.concatenate(all_idx))

        for k in meter:
            meter[k] /= max(steps, 1)
        usage = usage_sum / max(steps, 1)
        return meter, usage

    @torch.no_grad()
    def eval_known_acc(self, loader) -> float:
        self.model.eval()
        correct, total = 0, 0
        for x, y, _, _, _, _ in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            logits = self.model(x, temp=1.0, grl_alpha=0.0)["class_logits"]
            pred = logits.argmax(dim=1)
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
            out = self.model(x, temp=1.0, grl_alpha=0.0)
            logits = out["class_logits"]

            scores.append(self._unknown_score(logits))
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
        if len(np.unique(unknown_flags)) == 2 and scores.size > 0:
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
