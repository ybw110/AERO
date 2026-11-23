"""
Baseline 测试入口脚本。

功能：
1. 加载 baseline checkpoint；
2. 支持域覆盖/跨数据集覆盖评估；
3. 输出 known acc 与 open-set AUROC。
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch

from algorithms import BaselineTrainer
from config import Route1Config
from datasets import build_dataloaders, build_eval_loader_full
from models import AFRDNNet, AMINetNet, DACNNet, ClassifierNet, DiversifyNet
from utils.common import ensure_dir
from utils.visualization import plot_open_set_hist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Baseline Testing")
    p.add_argument("--ckpt", type=str, required=True, help="best_model.pt 路径")
    p.add_argument("--output", type=str, default="")
    p.add_argument(
        "--method",
        type=str,
        default="",
        choices=[
            "",
            "ERM",
            "JTT",
            "CVAR_DRO",
            "GROUPDRO",
            "DIVERSIFY",
            "AMINET",
            "DACN",
            "AFRDN",
            "WPA_PDG",
        ],
    )
    p.add_argument("--dataset_override", type=str, default="", choices=["", "CWRU", "JNU", "SELF_A", "SELF_B"])
    p.add_argument("--domains_override", type=str, default="", help="逗号分隔，如 3HP 或 600,800")
    p.add_argument("--self_processed_root", type=str, default="", help="Self npz 根目录（包含 SelfA/SelfB 子目录）")
    p.add_argument("--self_a_subdir", type=str, default="", help="SelfA 子目录名（默认 SelfA）")
    p.add_argument("--self_b_subdir", type=str, default="", help="SelfB 子目录名（默认 SelfB）")
    p.add_argument("--self_norm_mode", type=str, default="", choices=["", "train_global", "per_sample"])
    p.add_argument("--ood_score_type", type=str, default="", choices=["", "msp", "energy", "maxlogit"])
    p.add_argument("--ood_energy_temp", type=float, default=-1.0)
    p.add_argument("--known_classes", type=str, default="", help="逗号分隔，如 0,1,3")
    p.add_argument("--unknown_classes", type=str, default="", help="逗号分隔，如 2")
    p.add_argument("--full_eval", action="store_true", help="使用全量数据评估（不做切分）")
    return p.parse_args()


def _parse_int_csv(text: str) -> Optional[List[int]]:
    if not text.strip():
        return None
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def _parse_str_csv(text: str) -> Optional[List[str]]:
    if not text.strip():
        return None
    return [v.strip() for v in text.split(",") if v.strip()]


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = raw["cfg"]
    cfg = Route1Config()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.output.strip():
        cfg.output_dir = str(Path(args.output).resolve())
    out_dir = ensure_dir(cfg.output_dir)

    if args.dataset_override.strip():
        cfg.dataset_name = args.dataset_override.strip()
    if args.self_processed_root.strip():
        cfg.self_processed_root = str(Path(args.self_processed_root).resolve())
    if args.self_a_subdir.strip():
        cfg.self_a_subdir = args.self_a_subdir.strip()
    if args.self_b_subdir.strip():
        cfg.self_b_subdir = args.self_b_subdir.strip()
    if args.self_norm_mode.strip():
        cfg.self_norm_mode = args.self_norm_mode.strip()
    if args.ood_score_type.strip():
        cfg.ood_score_type = args.ood_score_type.strip()
    if args.ood_energy_temp > 0:
        cfg.ood_energy_temp = args.ood_energy_temp
    doms = _parse_str_csv(args.domains_override)
    if doms is not None:
        cfg.source_domains = doms
    known = _parse_int_csv(args.known_classes)
    unknown = _parse_int_csv(args.unknown_classes)
    if known is not None:
        cfg.known_classes = known
    if unknown is not None:
        cfg.unknown_classes = unknown

    method = args.method.strip().upper() if args.method.strip() else str(cfg.baseline_method).upper()
    if method not in {"ERM", "JTT", "CVAR_DRO", "GROUPDRO", "DIVERSIFY", "AMINET", "DACN", "AFRDN", "WPA_PDG"}:
        raise ValueError(f"Unsupported baseline method for testing: {method}")

    if args.full_eval:
        eval_bundle = build_eval_loader_full(cfg)
        test_loader = eval_bundle.test_loader
    else:
        bundle = build_dataloaders(cfg)
        test_loader = bundle.test_loader

    if method == "DIVERSIFY":
        model = DiversifyNet(cfg.latent_dim, cfg.num_known_classes, cfg.diversify_k).to(cfg.device)
    elif method == "WPA_PDG":
        model = DiversifyNet(cfg.latent_dim, cfg.num_known_classes, cfg.diversify_k).to(cfg.device)
    elif method == "AFRDN":
        model = AFRDNNet(cfg.latent_dim, cfg.sequence_length, cfg.num_known_classes, cfg.afrdn_num_env).to(cfg.device)
    elif method == "AMINET":
        model = AMINetNet(cfg.num_known_classes).to(cfg.device)
    elif method == "DACN":
        model = DACNNet(cfg.num_known_classes).to(cfg.device)
    else:
        model = ClassifierNet(cfg.latent_dim, cfg.num_known_classes).to(cfg.device)
    model.load_state_dict(raw["model"], strict=True)

    trainer = BaselineTrainer(model, cfg, cfg.device)
    trainer.method = method
    test_info = trainer.eval_open_set(test_loader)

    plot_open_set_hist(test_info["scores"], test_info["is_unknown"], Path(out_dir) / "test_open_set_histogram.png")
    with open(Path(out_dir) / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": method,
                "test_known_acc": test_info["test_known_acc"],
                "test_open_auroc": test_info["test_open_auroc"],
                "test_open_aupr": test_info.get("test_open_aupr", float("nan")),
                "test_open_fpr95": test_info.get("test_open_fpr95", float("nan")),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"method={method}")
    print(f"test_known_acc={test_info['test_known_acc']:.6f}")
    print(f"test_open_auroc={test_info['test_open_auroc']}")
    print(f"test_open_aupr={test_info.get('test_open_aupr', float('nan'))}")
    print(f"test_open_fpr95={test_info.get('test_open_fpr95', float('nan'))}")
    print(f"结果已写入: {out_dir}")


if __name__ == "__main__":
    main()


