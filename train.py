"""
璁粌鍏ュ彛鑴氭湰锛堟暀瀛︾増娉ㄩ噴锛夈€?
闃呰寤鸿锛堢涓€娆＄湅浠ｇ爜锛夛細
1. 鍏堢湅 `build_cfg`锛屼簡瑙ｅ疄楠屽弬鏁版潵婧愶紱
2. 鍐嶇湅 `build_dataloaders` 鐨勮繑鍥炵粨鏋勶紱
3. 閲嶇偣鐪?epoch 寰幆閲岀殑鍥涗釜鎸囨爣锛?   - train_cls / train_rec / train_adv / active_k
4. 鏈€鍚庣湅杈撳嚭鏂囦欢锛坢etrics.json + fig1 + fig3锛夈€?"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.optim import Adam

from algorithms import Route1Algorithm
from config import Route1Config
from datasets import build_dataloaders
from models import ReconstructionRoutingModel
from utils.common import ensure_dir, set_seed
from utils.visualization import (
    plot_dynamic_collapse,
    plot_loss_curves,
    plot_open_set_hist,
    save_history_csv,
)


def parse_args() -> argparse.Namespace:
    """
    鍛戒护琛屽弬鏁般€?
    鍙傛暟璇存槑锛?    - dataset: 鏁版嵁闆嗗悕绉?    - domains: 鍩熷垪琛紝閫楀彿鍒嗛殧锛堝彲閫夛級
    - epochs/batch_size/m_max: 鏍稿績璁粌瓒呭弬鏁?    - output: 杈撳嚭鐩綍
    """
    p = argparse.ArgumentParser("Route1 Training")
    p.add_argument("--dataset", type=str, default="CWRU", choices=["CWRU", "JNU", "SELF_A", "SELF_B"])
    p.add_argument("--domains", type=str, default="", help="閫楀彿鍒嗛殧锛屽 0HP,1HP,2HP,3HP 鎴?100,101")
    p.add_argument("--self_processed_root", type=str, default="", help="Self npz 鏍圭洰褰曪紙鍚嚜鏁扮粍 SelfA/SelfB 瀛愮洰褰曪級")
    p.add_argument("--self_a_subdir", type=str, default="", help="SelfA 瀛愮洰褰曞悕锛堥粯璁?SelfA锛?")
    p.add_argument("--self_b_subdir", type=str, default="", help="SelfB 瀛愮洰褰曞悕锛堥粯璁?SelfB锛?")
    p.add_argument("--self_norm_mode", type=str, default="", choices=["", "train_global", "per_sample"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--m_max", type=int, default=15)
    p.add_argument("--target_k", type=int, default=-1, help="Route1 目标活跃专家数（自然坍缩目标）")
    p.add_argument("--lr", type=float, default=-1.0)
    p.add_argument("--weight_decay", type=float, default=-1.0)
    p.add_argument("--route1_pretrain_epochs", type=int, default=-1)
    p.add_argument("--loss_w_adv", type=float, default=-1.0)
    p.add_argument("--grl_alpha", type=float, default=-1.0)
    p.add_argument("--routing_temp", type=float, default=-1.0)
    p.add_argument("--ood_score_type", type=str, default="", choices=["", "msp", "energy", "maxlogit"])
    p.add_argument("--ood_energy_temp", type=float, default=-1.0)
    p.add_argument(
        "--ablation_variant",
        type=str,
        default="NONE",
        choices=[
            "NONE",
            "WO_NATURAL_COLLAPSE",
            "WO_CONFIDENCE_WEIGHTING",
            "WO_WORST_STEP",
            "WO_RECONSTRUCTION_ROUTING",
            "SOFT_ONLY_ROUTING",
            "EQCAP_NO_ENV",
        ],
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="")
    p.add_argument("--known_classes", type=str, default="", help="閫楀彿鍒嗛殧锛屽 0,1,3")
    p.add_argument("--unknown_classes", type=str, default="", help="閫楀彿鍒嗛殧锛屽 2")
    return p.parse_args()


def build_cfg(args: argparse.Namespace, project_root: Path) -> Route1Config:
    """
    灏嗗懡浠よ鍙傛暟鍐欏叆閰嶇疆瀵硅薄銆?
    鏁欏鐐癸細
    - cfg 鏄€滃崟涓€鐪熷疄鏉ユ簮锛坰ingle source of truth锛夆€濓紝璁粌鍜屾祴璇曢兘渚濊禆瀹冿紱
    - 璺緞缁熶竴鍦ㄨ繖閲?resolve锛屽彲閬垮厤宸ヤ綔鐩綍鍙樺寲瀵艰嚧鐨勮矾寰勯敊璇€?    """
    cfg = Route1Config()
    cfg.dataset_name = args.dataset
    cfg.seed = args.seed
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.m_max = args.m_max
    if args.target_k > 0:
        cfg.target_k = args.target_k
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
    if args.known_classes.strip():
        cfg.known_classes = [int(v.strip()) for v in args.known_classes.split(",") if v.strip()]
    if args.unknown_classes.strip():
        cfg.unknown_classes = [int(v.strip()) for v in args.unknown_classes.split(",") if v.strip()]
    if args.lr > 0:
        cfg.lr = args.lr
    if args.weight_decay >= 0:
        cfg.weight_decay = args.weight_decay
    if args.route1_pretrain_epochs >= 0:
        cfg.route1_pretrain_epochs = args.route1_pretrain_epochs
    if args.loss_w_adv >= 0:
        cfg.loss_w_adv = args.loss_w_adv
    if args.grl_alpha >= 0:
        cfg.grl_alpha = args.grl_alpha
    if args.routing_temp > 0:
        cfg.routing_temp = args.routing_temp
    if args.ood_score_type.strip():
        cfg.ood_score_type = args.ood_score_type.strip()
    if args.ood_energy_temp > 0:
        cfg.ood_energy_temp = args.ood_energy_temp

    # Ablation switch (Route1 only)
    variant = str(args.ablation_variant).upper().strip()
    if variant == "WO_NATURAL_COLLAPSE":
        cfg.ablation_disable_natural_collapse = True
    elif variant == "WO_CONFIDENCE_WEIGHTING":
        cfg.ablation_disable_confidence_weighting = True
    elif variant == "WO_WORST_STEP":
        cfg.ablation_disable_worst_step = True
    elif variant == "WO_RECONSTRUCTION_ROUTING":
        cfg.ablation_disable_reconstruction_routing = True
    elif variant == "SOFT_ONLY_ROUTING":
        cfg.ablation_soft_only_routing = True
    elif variant == "EQCAP_NO_ENV":
        cfg.ablation_equal_capacity_no_env = True

    cfg.resolve_paths(project_root)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ECG 榛樿璇箟锛欰AMI 5 绫?    # 宸茬煡锛歂/S/V/F -> [0,1,2,3]
    # 鏈煡锛歈 -> [4]
    return cfg


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    cfg = build_cfg(args, project_root)
    out_dir = ensure_dir(cfg.output_dir)
    set_seed(cfg.seed)

    # -------------------- 1) 鏁版嵁 --------------------
    # builder 鍐呴儴瀹屾垚锛?    # - 鏁版嵁璇诲彇
    # - known/unknown 鏄犲皠
    # - 鍒嗗眰鍒囧垎
    # - DataLoader 鏋勫缓
    bundle = build_dataloaders(cfg)

    # -------------------- 2) 妯″瀷 --------------------
    seq_len = cfg.sequence_length
    model = ReconstructionRoutingModel(
        seq_len=seq_len,
        latent_dim=cfg.latent_dim,
        m_max=cfg.m_max,
        num_known_classes=cfg.num_known_classes,
    ).to(cfg.device)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    algo = Route1Algorithm(model, cfg, cfg.device)

    history: List[Dict[str, float]] = []
    usage_history = []
    best_val = -1.0
    best_auroc = -1.0
    best_path = out_dir / "best_model.pt"

    # -------------------- 3) 璁粌寰幆 --------------------
    # 姣忚疆杈撳嚭锛?    # - train_loss 鍙婂叾鍒嗛」锛坈ls/rec/adv锛?    # - val_known_acc锛堢敤浜庨€夋渶浼樻ā鍨嬶級
    # - test_known_acc/test_open_auroc锛堟硾鍖栦笌寮€闆嗭級
    # - active_k锛堜笓瀹惰嚜鐒舵敹鏁涘悗鐨勬暟閲忥級
    for ep in range(1, cfg.epochs + 1):
        train_meter, usage = algo.train_one_epoch(bundle, optimizer, epoch=ep)
        val_acc = algo.eval_known_acc(bundle.val_loader)
        test_info = algo.eval_open_set(bundle.test_loader)

        active_k = int((usage > cfg.expert_active_threshold).sum())
        row = {
            "epoch": ep,
            "train_loss": train_meter["loss"],
            "train_cls": train_meter["cls"],
            "train_rec": train_meter["rec"],
            "train_adv": train_meter["adv"],
            "train_ent": train_meter.get("ent", 0.0),
            "train_col": train_meter.get("col", 0.0),
            "train_tmp": train_meter.get("tmp", 0.0),
            "val_known_acc": val_acc,
            "test_known_acc": test_info["test_known_acc"],
            "test_open_auroc": test_info["test_open_auroc"],
            "test_open_aupr": test_info.get("test_open_aupr", float("nan")),
            "test_open_fpr95": test_info.get("test_open_fpr95", float("nan")),
            "active_k": active_k,
        }
        history.append(row)
        usage_history.append(usage)

        cur_auroc = float(test_info.get("test_open_auroc", float("nan")))
        if (val_acc > best_val) or (val_acc == best_val and cur_auroc > best_auroc):
            best_val = val_acc
            best_auroc = cur_auroc
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, best_path)

        stage = "pretrain" if ep <= cfg.route1_pretrain_epochs else "joint"
        print(
            f"[{ep:03d}/{cfg.epochs}] "
            f"loss={row['train_loss']:.4f} cls={row['train_cls']:.4f} rec={row['train_rec']:.4f} adv={row['train_adv']:.4f} "
            f"ent={row['train_ent']:.4f} col={row['train_col']:.4f} tmp={row['train_tmp']:.4f} | "
            f"val={row['val_known_acc']:.4f} test={row['test_known_acc']:.4f} "
            f"auroc={row['test_open_auroc']} aupr={row['test_open_aupr']} fpr95={row['test_open_fpr95']} K={active_k}"
            f" stage={stage}"
        )

    # -------------------- 4) 缁撴灉淇濆瓨 --------------------
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    save_history_csv(history, out_dir / "metrics.csv")
    plot_loss_curves(history, out_dir / "loss_curves.png", title="Route1 Training Curves")

    # Fig1锛氫笓瀹惰嚜鐒跺潔缂╁浘
    usage_arr = np.stack(usage_history, axis=0) if usage_history else np.zeros((0, cfg.m_max))
    np.save(out_dir / "expert_usage_history.npy", usage_arr)
    plot_dynamic_collapse(usage_arr, out_dir / "fig1_dynamic_routing_collapse.png")

    # Fig3锛氬紑闆嗚宸洿鏂瑰浘
    final_test = algo.eval_open_set(bundle.test_loader)
    plot_open_set_hist(final_test["scores"], final_test["is_unknown"], out_dir / "fig3_open_set_histogram.png")

    print(f"Training finished. Output dir: {out_dir}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()


