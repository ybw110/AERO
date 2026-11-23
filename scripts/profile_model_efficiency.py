#!/usr/bin/env python3
"""Profile Params / FLOPs / Latency for trained models under one scenario case dir.

By default it profiles *deploy scope* (inference subgraph only), which is usually what
should be reported for edge deployment.
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn

# Ensure project root is importable when executing from scripts/.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import Route1Config
from models import AFRDNNet, AMINetNet, DACNNet, ClassifierNet, DiversifyNet, ReconstructionRoutingModel


METHOD_DIRS = ["route1", "erm", "jtt", "cvar_dro", "groupdro", "diversify", "aminet", "dacn", "afrdn", "wpa_pdg"]


class _EncoderClassifierDeploy(nn.Module):
    def __init__(self, encoder: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.classifier(z)


class _AMINetDeploy(nn.Module):
    def __init__(self, feat: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.feat = feat
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.feat(x)
        return self.classifier(z)


class _DACNDeploy(nn.Module):
    def __init__(self, F: nn.Module, G: nn.Module, C: nn.Module) -> None:
        super().__init__()
        self.F = F
        self.G = G
        self.C = C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.F(x)
        z = self.G(f)
        return self.C(z)


class _FullModelWrapper(nn.Module):
    """Wrap full training-time forward into logits-only forward for profiling."""

    def __init__(self, model: nn.Module, method: str) -> None:
        super().__init__()
        self.model = model
        self.method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "ROUTE1":
            out = self.model(x, temp=1.0, grl_alpha=0.0)
            return out["class_logits"]
        if self.method in {"DIVERSIFY", "DACN", "WPA_PDG", "AFRDN"}:
            out = self.model(x, grl_alpha=0.0)
            return out["class_logits"]
        out = self.model(x)
        return out["class_logits"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Profile Params/FLOPs/Latency")
    p.add_argument("--case_dir", type=str, required=True, help="Case dir containing method subdirs")
    p.add_argument("--methods", type=str, default=",".join(METHOD_DIRS), help="Comma-separated method dirs")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--profile_scope", type=str, default="deploy", choices=["deploy", "full"])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=-1, help="Override sequence length; default from ckpt cfg")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--output_csv", type=str, default="")
    p.add_argument("--output_md", type=str, default="")
    p.add_argument("--round_digits", type=int, default=4)
    return p.parse_args()


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but CUDA is not available.")
    return device


def _build_cfg(raw_cfg: Dict) -> Route1Config:
    cfg = Route1Config()
    for k, v in raw_cfg.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _infer_method(state_dict: Dict[str, torch.Tensor], cfg: Route1Config, method_dir: str) -> str:
    key = method_dir.strip().lower()
    fixed = {
        "route1": "ROUTE1",
        "erm": "ERM",
        "jtt": "JTT",
        "cvar_dro": "CVAR_DRO",
        "groupdro": "GROUPDRO",
        "diversify": "DIVERSIFY",
        "aminet": "AMINET",
        "dacn": "DACN",
        "afrdn": "AFRDN",
        "wpa_pdg": "WPA_PDG",
        "wpa-pdg": "WPA_PDG",
    }
    if key in fixed:
        return fixed[key]

    keys = list(state_dict.keys())
    if any(k.startswith("experts.") for k in keys):
        return "ROUTE1"
    if any(k.startswith("domain_head.") for k in keys):
        return "DIVERSIFY"
    if any(k.startswith("gen.") for k in keys):
        return "AMINET"
    if any(k.startswith("F.") for k in keys) and any(k.startswith("D.") for k in keys):
        return "DACN"
    m = str(getattr(cfg, "baseline_method", "ERM")).upper()
    return m if m else "ERM"


def _build_model(method: str, cfg: Route1Config, seq_len: int) -> nn.Module:
    if method == "ROUTE1":
        return ReconstructionRoutingModel(
            seq_len=seq_len,
            latent_dim=cfg.latent_dim,
            m_max=cfg.m_max,
            num_known_classes=cfg.num_known_classes,
        )
    if method == "DIVERSIFY":
        return DiversifyNet(cfg.latent_dim, cfg.num_known_classes, cfg.diversify_k)
    if method == "WPA_PDG":
        return DiversifyNet(cfg.latent_dim, cfg.num_known_classes, cfg.diversify_k)
    if method == "AFRDN":
        return AFRDNNet(cfg.latent_dim, seq_len, cfg.num_known_classes, cfg.afrdn_num_env)
    if method == "AMINET":
        return AMINetNet(cfg.num_known_classes)
    if method == "DACN":
        return DACNNet(cfg.num_known_classes)
    return ClassifierNet(cfg.latent_dim, cfg.num_known_classes)


def _build_profile_module(base_model: nn.Module, method: str, scope: str) -> nn.Module:
    if scope == "full":
        return _FullModelWrapper(base_model, method)

    # deploy scope: keep only inference subgraph that is actually used at deployment
    if method == "ROUTE1":
        return _EncoderClassifierDeploy(base_model.encoder, base_model.classifier)
    if method == "DIVERSIFY":
        return _EncoderClassifierDeploy(base_model.encoder, base_model.classifier)
    if method == "WPA_PDG":
        return _EncoderClassifierDeploy(base_model.encoder, base_model.classifier)
    if method == "AFRDN":
        return _EncoderClassifierDeploy(base_model.encoder, base_model.classifier)
    if method == "AMINET":
        return _AMINetDeploy(base_model.feat, base_model.classifier)
    if method == "DACN":
        return _DACNDeploy(base_model.F, base_model.G, base_model.C)
    # ERM/JTT/CVAR_DRO/GROUPDRO
    return _EncoderClassifierDeploy(base_model.encoder, base_model.classifier)


def _count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


class _FlopCounter:
    """Simple forward-hook FLOPs estimator (Conv1d + Linear)."""

    def __init__(self) -> None:
        self.total_flops = 0.0
        self.handles = []

    def _conv1d_hook(self, module: nn.Conv1d, inputs, output) -> None:
        x = inputs[0]
        if not isinstance(output, torch.Tensor):
            return
        b = int(x.shape[0])
        out_c = int(output.shape[1])
        out_l = int(output.shape[2])
        k = int(module.kernel_size[0])
        in_c = int(module.in_channels)
        groups = int(module.groups)
        per_out = 2.0 * (k * (in_c / groups))
        if module.bias is not None:
            per_out += 1.0
        self.total_flops += float(b * out_c * out_l * per_out)

    def _linear_hook(self, module: nn.Linear, inputs, output) -> None:
        if not isinstance(output, torch.Tensor):
            return
        in_feat = int(module.in_features)
        out_feat = int(module.out_features)
        out_numel = int(output.numel())
        n = out_numel // max(out_feat, 1)
        per_out = 2.0 * in_feat
        if module.bias is not None:
            per_out += 1.0
        self.total_flops += float(n * out_feat * per_out)

    def register(self, model: nn.Module) -> None:
        for m in model.modules():
            if isinstance(m, nn.Conv1d):
                self.handles.append(m.register_forward_hook(self._conv1d_hook))
            elif isinstance(m, nn.Linear):
                self.handles.append(m.register_forward_hook(self._linear_hook))

    def clear(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


def _estimate_flops(model: nn.Module, x: torch.Tensor, device: str) -> float:
    counter = _FlopCounter()
    counter.register(model)
    model.eval()
    with torch.no_grad():
        _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
    flops = counter.total_flops
    counter.clear()
    return float(flops)


def _measure_latency_ms(model: nn.Module, x: torch.Tensor, device: str, warmup: int, iters: int) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(max(warmup, 0)):
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(max(iters, 1)):
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    latency_ms_per_batch = (t1 - t0) * 1000.0 / max(iters, 1)
    return float(latency_ms_per_batch)


def _write_csv(rows: List[Dict], out_csv: Path) -> None:
    fields = [
        "method",
        "method_dir",
        "profile_scope",
        "checkpoint",
        "device",
        "batch_size",
        "seq_len",
        "params",
        "params_m",
        "trainable_params",
        "trainable_params_m",
        "flops",
        "flops_m",
        "latency_ms_per_batch",
        "latency_ms_per_sample",
        "warmup",
        "iters",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _write_md(rows: List[Dict], out_md: Path, d: int) -> None:
    lines = []
    lines.append("| Method | Scope | Params (M) | FLOPs (M) | Latency (ms) |")
    lines.append("|---|---|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['profile_scope']} | {float(r['params_m']):.{d}f} | {float(r['flops_m']):.{d}f} | {float(r['latency_ms_per_batch']):.{d}f} |"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]
    case_dir = Path(args.case_dir)
    if not case_dir.is_absolute():
        case_dir = (root_dir / case_dir).resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"case_dir not found: {case_dir}")

    device = _resolve_device(args.device)
    method_dirs = [m.strip().lower() for m in args.methods.split(",") if m.strip()]

    if args.output_csv:
        out_csv = Path(args.output_csv).resolve()
    else:
        out_csv = case_dir / f"efficiency_profile_{args.profile_scope}.csv"

    if args.output_md:
        out_md = Path(args.output_md).resolve()
    else:
        out_md = case_dir / f"efficiency_profile_{args.profile_scope}.md"

    rows: List[Dict] = []
    for mdir in method_dirs:
        ckpt = case_dir / mdir / "train" / "best_model.pt"
        if not ckpt.exists():
            print(f"[Skip] checkpoint missing: {ckpt}")
            continue

        raw = torch.load(ckpt, map_location="cpu")
        cfg = _build_cfg(raw.get("cfg", {}))
        state_dict = raw.get("model", {})
        method = _infer_method(state_dict, cfg, mdir)

        seq_len = args.seq_len if args.seq_len > 0 else int(getattr(cfg, "sequence_length", 1024))

        full_model = _build_model(method, cfg, seq_len=seq_len)
        full_model.load_state_dict(state_dict, strict=True)
        full_model.to(device)

        profile_model = _build_profile_module(full_model, method, args.profile_scope).to(device)

        params, trainable = _count_params(profile_model)

        x = torch.randn(args.batch_size, 1, seq_len, device=device)
        flops = _estimate_flops(profile_model, x, device=device)
        lat_batch = _measure_latency_ms(profile_model, x, device=device, warmup=args.warmup, iters=args.iters)
        lat_sample = lat_batch / max(args.batch_size, 1)

        row = {
            "method": method,
            "method_dir": mdir,
            "profile_scope": args.profile_scope,
            "checkpoint": str(ckpt),
            "device": device,
            "batch_size": int(args.batch_size),
            "seq_len": seq_len,
            "params": int(params),
            "params_m": float(params) / 1e6,
            "trainable_params": int(trainable),
            "trainable_params_m": float(trainable) / 1e6,
            "flops": float(flops),
            "flops_m": float(flops) / 1e6,
            "latency_ms_per_batch": float(lat_batch),
            "latency_ms_per_sample": float(lat_sample),
            "warmup": int(args.warmup),
            "iters": int(args.iters),
        }
        rows.append(row)
        print(
            f"[OK] {method:<10} scope={args.profile_scope:<6} "
            f"params={row['params_m']:.4f}M flops={row['flops_m']:.2f}M latency={row['latency_ms_per_batch']:.3f}ms"
        )

    if not rows:
        raise RuntimeError("No valid checkpoints profiled.")

    order = {
        name: i
        for i, name in enumerate(
            ["ROUTE1", "ERM", "JTT", "CVAR_DRO", "GROUPDRO", "DIVERSIFY", "AMINET", "DACN", "AFRDN", "WPA_PDG"]
        )
    }
    rows = sorted(rows, key=lambda r: order.get(str(r["method"]), 999))

    _write_csv(rows, out_csv)
    _write_md(rows, out_md, d=max(args.round_digits, 0))
    print(f"[DONE] CSV: {out_csv}")
    print(f"[DONE] MD : {out_md}")


if __name__ == "__main__":
    main()
