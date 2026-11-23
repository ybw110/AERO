"""
Baseline 缁熶竴璁粌鍏ュ彛銆?
鏀寔鏂规硶锛?- ERM
- JTT
- CVAR_DRO
- GROUPDRO (oracle)
- DIVERSIFY (杞婚噺澶嶇幇鐗?
- DIVERSIFY_EXTERNAL (璋冪敤椤圭洰鍐呭師濮?diversify 浠ｇ爜)
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import Adam

from algorithms import BaselineTrainer
from config import Route1Config
from datasets import build_dataloaders
from models import AFRDNNet, AMINetNet, DACNNet, ClassifierNet, DiversifyNet
from utils.common import ensure_dir, set_seed
from utils.visualization import plot_loss_curves, plot_open_set_hist, save_history_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Baseline Training")
    p.add_argument(
        "--method",
        type=str,
        default="ERM",
        choices=[
            "ERM",
            "JTT",
            "CVAR_DRO",
            "GROUPDRO",
            "DIVERSIFY",
            "AMINET",
            "DACN",
            "AFRDN",
            "WPA_PDG",
            "DIVERSIFY_EXTERNAL",
        ],
    )
    p.add_argument("--dataset", type=str, default="CWRU", choices=["CWRU", "JNU", "SELF_A", "SELF_B"])
    p.add_argument("--domains", type=str, default="")
    p.add_argument("--self_processed_root", type=str, default="", help="Self npz 根目录（包含 SelfA/SelfB 子目录）")
    p.add_argument("--self_a_subdir", type=str, default="", help="SelfA 子目录名（默认 SelfA）")
    p.add_argument("--self_b_subdir", type=str, default="", help="SelfB 子目录名（默认 SelfB）")
    p.add_argument("--self_norm_mode", type=str, default="", choices=["", "train_global", "per_sample"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=-1.0)
    p.add_argument("--weight_decay", type=float, default=-1.0)
    p.add_argument("--ood_score_type", type=str, default="", choices=["", "msp", "energy", "maxlogit"])
    p.add_argument("--ood_energy_temp", type=float, default=-1.0)
    p.add_argument("--output", type=str, default="")
    p.add_argument("--known_classes", type=str, default="", help="閫楀彿鍒嗛殧锛屽 0,1,3")
    p.add_argument("--unknown_classes", type=str, default="", help="閫楀彿鍒嗛殧锛屽 2")
    return p.parse_args()


def build_cfg(args: argparse.Namespace, project_root: Path) -> Route1Config:
    cfg = Route1Config()
    cfg.baseline_method = args.method
    cfg.seed = args.seed
    cfg.dataset_name = args.dataset
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    if args.domains.strip():
        cfg.source_domains = [x.strip() for x in args.domains.split(",") if x.strip()]
    if args.self_processed_root.strip():
        cfg.self_processed_root = args.self_processed_root.strip()
    if args.self_a_subdir.strip():
        cfg.self_a_subdir = args.self_a_subdir.strip()
    if args.self_b_subdir.strip():
        cfg.self_b_subdir = args.self_b_subdir.strip()
    if args.self_norm_mode.strip():
        cfg.self_norm_mode = args.self_norm_mode.strip()
    if args.output.strip():
        cfg.output_dir = str(Path(args.output).resolve())
    else:
        cfg.output_dir = f"浠ｇ爜/route1_framework/outputs_{args.method.lower()}_{args.dataset.lower()}"
    if args.known_classes.strip():
        cfg.known_classes = [int(v.strip()) for v in args.known_classes.split(",") if v.strip()]
    if args.unknown_classes.strip():
        cfg.unknown_classes = [int(v.strip()) for v in args.unknown_classes.split(",") if v.strip()]
    if args.lr > 0:
        cfg.lr = args.lr
    if args.weight_decay >= 0:
        cfg.weight_decay = args.weight_decay
    if args.ood_score_type.strip():
        cfg.ood_score_type = args.ood_score_type.strip()
    if args.ood_energy_temp > 0:
        cfg.ood_energy_temp = args.ood_energy_temp
    cfg.resolve_paths(project_root)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def run_diversify_external(cfg: Route1Config, project_root: Path) -> None:
    """
    璋冪敤宸叉湁鐨?diversify 鍘熷鑴氭湰锛圕WRU/JNU锛夈€?    """
    if cfg.dataset_name.upper() not in {"CWRU", "JNU"}:
        raise ValueError("DIVERSIFY_EXTERNAL currently supports only CWRU/JNU.")

    script = project_root / "浠ｇ爜" / "瀹為獙CWRU+JNU+OOD" / "diversify_train.py"
    if not script.exists():
        raise FileNotFoundError(f"鏈壘鍒板師濮?diversify 鑴氭湰: {script}")

    cmd = ["python", str(script)]
    print("杩愯鍘熷 Diversify:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(script.parent))


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    cfg = build_cfg(args, project_root)
    out_dir = ensure_dir(cfg.output_dir)
    set_seed(cfg.seed)

    if cfg.baseline_method.upper() == "DIVERSIFY_EXTERNAL":
        run_diversify_external(cfg, project_root)
        return

    bundle = build_dataloaders(cfg)
    method = cfg.baseline_method.upper()

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

    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    trainer = BaselineTrainer(model, cfg, cfg.device)

    history: List[Dict[str, float]] = []
    best_val = -1.0
    best_path = out_dir / "best_model.pt"

    if method == "JTT":
        print(f"[JTT] Stage-1 ERM for {cfg.jtt_stage1_epochs} epochs")
        for ep in range(1, cfg.jtt_stage1_epochs + 1):
            # 涓存椂褰撲綔 ERM 璁粌
            trainer.method = "ERM"
            meter = trainer.train_one_epoch(bundle, optimizer, epoch=ep)
            val_acc = trainer.eval_known_acc(bundle.val_loader)
            print(f"[JTT stage1 {ep}/{cfg.jtt_stage1_epochs}] loss={meter['loss']:.4f} val={val_acc:.4f}")
        trainer.prepare_jtt_weights(bundle)
        trainer.method = "JTT"

    for ep in range(1, cfg.epochs + 1):
        if method == "DIVERSIFY":
            trainer.init_diversify_pseudo_domains(bundle)
        elif method == "WPA_PDG":
            trainer.init_wpa_pdg_pseudo_domains(bundle)
        elif method == "AFRDN":
            trainer.init_afrdn_pseudo_domains(bundle)

        meter = trainer.train_one_epoch(bundle, optimizer, epoch=ep)
        val_acc = trainer.eval_known_acc(bundle.val_loader)
        test_info = trainer.eval_open_set(bundle.test_loader)

        row = {
            "epoch": ep,
            "train_loss": meter["loss"],
            "train_cls": meter.get("cls", 0.0),
            "train_adv": meter.get("adv", 0.0),
            "train_cvar": meter.get("cvar", 0.0),
            "train_groupdro": meter.get("groupdro", 0.0),
            "train_jtt_weighted": meter.get("jtt_weighted", 0.0),
            "val_known_acc": val_acc,
            "test_known_acc": test_info["test_known_acc"],
            "test_open_auroc": test_info["test_open_auroc"],
            "test_open_aupr": test_info.get("test_open_aupr", float("nan")),
            "test_open_fpr95": test_info.get("test_open_fpr95", float("nan")),
            "train_rec": meter.get("rec", 0.0),
            "train_irm": meter.get("irm", 0.0),
            "train_mix": meter.get("mix", 0.0),
        }
        history.append(row)

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, best_path)

        print(
            f"[{method} {ep:03d}/{cfg.epochs}] "
            f"loss={row['train_loss']:.4f} val={row['val_known_acc']:.4f} "
            f"test={row['test_known_acc']:.4f} auroc={row['test_open_auroc']} "
            f"aupr={row['test_open_aupr']} fpr95={row['test_open_fpr95']} "
            f"rec={row['train_rec']:.4f} irm={row['train_irm']:.4f} mix={row['train_mix']:.4f}"
        )

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    save_history_csv(history, out_dir / "metrics.csv")
    plot_loss_curves(history, out_dir / "loss_curves.png", title=f"{method} Training Curves")

    final_test = trainer.eval_open_set(bundle.test_loader)
    plot_open_set_hist(final_test["scores"], final_test["is_unknown"], out_dir / "open_set_histogram.png")
    print(f"Baseline training finished. Output dir: {out_dir}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()


