"""
娴嬭瘯鍏ュ彛鑴氭湰銆?
鍔熻兘锛?1. 鍔犺浇璁粌濂界殑 checkpoint锛?2. 鍦ㄦ祴璇曢泦涓婅緭鍑?known accuracy 涓?open-set AUROC锛?3. 鐢熸垚寮€闆嗛噸鏋勮宸浘銆?
鏁欏璇存槑锛?- `test.py` 涓嶄細鏇存柊妯″瀷鍙傛暟锛屼粎鍋氬墠鍚戣瘎浼帮紱
- 鑻ヤ綘瑕佹瘮杈冨涓?checkpoint锛屽彲閲嶅鎵ц鏈剼鏈苟鎸囧畾涓嶅悓 --output锛?- 寤鸿濮嬬粓閰嶅悎 `test_metrics.json` 淇濆瓨缁撴灉锛岄伩鍏嶆墜鎶勬棩蹇楄宸€?"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch

from algorithms import Route1Algorithm
from config import Route1Config
from datasets import build_dataloaders, build_eval_loader_full
from models import ReconstructionRoutingModel
from utils.common import ensure_dir
from utils.visualization import plot_open_set_hist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Route1 Testing")
    p.add_argument("--ckpt", type=str, required=True, help="best_model.pt 璺緞")
    p.add_argument("--output", type=str, default="")
    p.add_argument("--dataset_override", type=str, default="", choices=["", "CWRU", "JNU", "SELF_A", "SELF_B"])
    p.add_argument("--domains_override", type=str, default="", help="閫楀彿鍒嗛殧锛屽 3HP 鎴?600,800")
    p.add_argument("--self_processed_root", type=str, default="", help="Self npz 鏍圭洰褰曪紙鍚嚜鏁扮粍 SelfA/SelfB 瀛愮洰褰曪級")
    p.add_argument("--self_a_subdir", type=str, default="", help="SelfA 瀛愮洰褰曞悕锛堥粯璁?SelfA锛?")
    p.add_argument("--self_b_subdir", type=str, default="", help="SelfB 瀛愮洰褰曞悕锛堥粯璁?SelfB锛?")
    p.add_argument("--self_norm_mode", type=str, default="", choices=["", "train_global", "per_sample"])
    p.add_argument("--ood_score_type", type=str, default="", choices=["", "msp", "energy", "maxlogit"])
    p.add_argument("--ood_energy_temp", type=float, default=-1.0)
    p.add_argument("--known_classes", type=str, default="", help="閫楀彿鍒嗛殧锛屽 0,1,3")
    p.add_argument("--unknown_classes", type=str, default="", help="閫楀彿鍒嗛殧锛屽 2")
    p.add_argument("--full_eval", action="store_true", help="浣跨敤鍏ㄩ噺鏁版嵁璇勪及锛堜笉鍋氬垏鍒嗭級锛岀敤浜庡煙澶?璺ㄦ暟鎹泦娴嬭瘯")
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

    if args.full_eval:
        eval_bundle = build_eval_loader_full(cfg)
        test_loader = eval_bundle.test_loader
    else:
        bundle = build_dataloaders(cfg)
        test_loader = bundle.test_loader
    seq_len = cfg.sequence_length
    model = ReconstructionRoutingModel(
        seq_len=seq_len,
        latent_dim=cfg.latent_dim,
        m_max=cfg.m_max,
        num_known_classes=cfg.num_known_classes,
    ).to(cfg.device)
    model.load_state_dict(raw["model"], strict=True)

    algo = Route1Algorithm(model, cfg, cfg.device)
    test_info = algo.eval_open_set(test_loader)

    plot_open_set_hist(test_info["scores"], test_info["is_unknown"], Path(out_dir) / "test_open_set_histogram.png")
    with open(Path(out_dir) / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_known_acc": test_info["test_known_acc"],
                "test_open_auroc": test_info["test_open_auroc"],
                "test_open_aupr": test_info.get("test_open_aupr", float("nan")),
                "test_open_fpr95": test_info.get("test_open_fpr95", float("nan")),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"test_known_acc={test_info['test_known_acc']:.6f}")
    print(f"test_open_auroc={test_info['test_open_auroc']}")
    print(f"test_open_aupr={test_info.get('test_open_aupr', float('nan'))}")
    print(f"test_open_fpr95={test_info.get('test_open_fpr95', float('nan'))}")
    print(f"Results written to: {out_dir}")


if __name__ == "__main__":
    main()


