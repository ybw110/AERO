"""
场景化 benchmark 脚本（支持 A/B/C 试验矩阵）。

A: 同域 OOD（单域训练 + 单域全量测试）
B: 跨域泛化（多域训练 + 留一域全量测试）
C: 跨数据集迁移（源数据集训练 + 目标数据集全量测试）
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Scenario Benchmark")
    p.add_argument("--scenario", type=str, required=True, choices=["A_WITHIN_DOMAIN", "B_CROSS_DOMAIN", "C_CROSS_DATASET"])
    p.add_argument(
        "--methods",
        type=str,
        default="ROUTE1,ERM,JTT,CVAR_DRO,GROUPDRO,DIVERSIFY,AMINET,DACN,AFRDN,WPA_PDG",
    )

    p.add_argument("--dataset", type=str, default="CWRU", choices=["CWRU", "JNU", "SELF_A", "SELF_B"])
    p.add_argument("--domains", type=str, default="")

    p.add_argument("--train_dataset", type=str, default="CWRU", choices=["CWRU", "JNU", "SELF_A", "SELF_B"])
    p.add_argument("--test_dataset", type=str, default="JNU", choices=["CWRU", "JNU", "SELF_A", "SELF_B"])
    p.add_argument("--train_domains", type=str, default="")
    p.add_argument("--test_domains", type=str, default="")
    p.add_argument("--self_processed_root", type=str, default="", help="Self npz 根目录（包含 SelfA/SelfB 子目录）")
    p.add_argument("--self_a_subdir", type=str, default="", help="SelfA 子目录名（默认 SelfA）")
    p.add_argument("--self_b_subdir", type=str, default="", help="SelfB 子目录名（默认 SelfB）")
    p.add_argument("--self_norm_mode", type=str, default="", choices=["", "train_global", "per_sample"])

    p.add_argument("--known_classes", type=str, default="0,1,3")
    p.add_argument("--unknown_classes", type=str, default="2")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--m_max", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=-1.0)
    p.add_argument("--weight_decay", type=float, default=-1.0)
    p.add_argument("--route1_pretrain_epochs", type=int, default=-1)
    p.add_argument("--loss_w_adv", type=float, default=-1.0)
    p.add_argument("--grl_alpha", type=float, default=-1.0)
    p.add_argument("--routing_temp", type=float, default=-1.0)
    p.add_argument("--ood_score_type", type=str, default="", choices=["", "msp", "energy", "maxlogit"])
    p.add_argument("--ood_energy_temp", type=float, default=-1.0)
    p.add_argument("--exp_name", type=str, default="scenario_bench")
    return p.parse_args()


def _run_cmd(cmd: List[str], cwd: Path) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd))


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _default_domains(dataset: str) -> List[str]:
    m = {
        "CWRU": ["0HP", "1HP", "2HP", "3HP"],
        "JNU": ["600", "800", "1000"],
        "SELF_A": ["1200", "1800"],
        "SELF_B": ["1200", "1800"],
    }
    return m[dataset]


def _parse_domains(text: str, fallback_dataset: str) -> List[str]:
    if text.strip():
        return [x.strip() for x in text.split(",") if x.strip()]
    return _default_domains(fallback_dataset)


def _build_cases(args: argparse.Namespace) -> List[Dict[str, str]]:
    cases: List[Dict[str, str]] = []

    if args.scenario == "A_WITHIN_DOMAIN":
        domains = _parse_domains(args.domains, args.dataset)
        for d in domains:
            cases.append(
                {
                    "case_name": f"A_{args.dataset}_{d}",
                    "train_dataset": args.dataset,
                    "train_domains": d,
                    "test_dataset": args.dataset,
                    "test_domains": d,
                }
            )

    elif args.scenario == "B_CROSS_DOMAIN":
        domains = _parse_domains(args.domains, args.dataset)
        if len(domains) < 2:
            raise ValueError("B_CROSS_DOMAIN 需要至少 2 个域。")
        for heldout in domains:
            train_list = [d for d in domains if d != heldout]
            cases.append(
                {
                    "case_name": f"B_{args.dataset}_train_{'-'.join(train_list)}_test_{heldout}",
                    "train_dataset": args.dataset,
                    "train_domains": ",".join(train_list),
                    "test_dataset": args.dataset,
                    "test_domains": heldout,
                }
            )

    elif args.scenario == "C_CROSS_DATASET":
        train_domains = _parse_domains(args.train_domains, args.train_dataset)
        test_domains = _parse_domains(args.test_domains, args.test_dataset)
        cases.append(
            {
                "case_name": f"C_{args.train_dataset}_to_{args.test_dataset}",
                "train_dataset": args.train_dataset,
                "train_domains": ",".join(train_domains),
                "test_dataset": args.test_dataset,
                "test_domains": ",".join(test_domains),
            }
        )

    return cases


def _write_summary(rows: List[Dict], out_dir: Path) -> None:
    fields = [
        "case",
        "method",
        "train_dataset",
        "train_domains",
        "test_dataset",
        "test_domains",
        "test_known_acc",
        "test_open_auroc",
        "test_open_aupr",
        "test_open_fpr95",
        "train_dir",
        "eval_dir",
    ]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    md = [
        "| Case | Method | Test Acc | Test AUROC | Test AUPR | Test FPR95 | Train Dataset | Test Dataset |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['case']} | {r['method']} | {float(r['test_known_acc']):.4f} | {float(r['test_open_auroc']):.4f} | "
            f"{float(r['test_open_aupr']):.4f} | {float(r['test_open_fpr95']):.4f} | "
            f"{r['train_dataset']}({r['train_domains']}) | {r['test_dataset']}({r['test_domains']}) |"
        )
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")

    tex = [
        "\\begin{tabular}{llcccc}",
        "\\hline",
        "Case & Method & Test Acc & Test AUROC & Test AUPR & Test FPR95 \\\\",
        "\\hline",
    ]
    for r in rows:
        tex.append(
            f"{r['case']} & {r['method']} & {float(r['test_known_acc']):.4f} & {float(r['test_open_auroc']):.4f} & "
            f"{float(r['test_open_aupr']):.4f} & {float(r['test_open_fpr95']):.4f} \\\\"
        )
    tex += ["\\hline", "\\end{tabular}"]
    (out_dir / "summary.tex").write_text("\n".join(tex), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    methods = [m.strip().upper() for m in args.methods.split(",") if m.strip()]
    cases = _build_cases(args)

    out_root = base_dir / "outputs_scenarios" / args.exp_name
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for c in cases:
        case_dir = out_root / c["case_name"]
        case_dir.mkdir(parents=True, exist_ok=True)

        for m in methods:
            train_dir = case_dir / m.lower() / "train"
            eval_dir = case_dir / m.lower() / "eval"
            train_dir.mkdir(parents=True, exist_ok=True)
            eval_dir.mkdir(parents=True, exist_ok=True)

            # 1) train
            common_train = [
                "--dataset",
                c["train_dataset"],
                "--domains",
                c["train_domains"],
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--seed",
                str(args.seed),
                "--known_classes",
                args.known_classes,
                "--unknown_classes",
                args.unknown_classes,
                "--output",
                str(train_dir),
            ]
            if args.self_processed_root.strip():
                common_train += ["--self_processed_root", args.self_processed_root]
            if args.self_a_subdir.strip():
                common_train += ["--self_a_subdir", args.self_a_subdir]
            if args.self_b_subdir.strip():
                common_train += ["--self_b_subdir", args.self_b_subdir]
            if args.self_norm_mode.strip():
                common_train += ["--self_norm_mode", args.self_norm_mode]
            if args.lr > 0:
                common_train += ["--lr", str(args.lr)]
            if args.weight_decay >= 0:
                common_train += ["--weight_decay", str(args.weight_decay)]
            if args.ood_score_type.strip():
                common_train += ["--ood_score_type", args.ood_score_type]
            if args.ood_energy_temp > 0:
                common_train += ["--ood_energy_temp", str(args.ood_energy_temp)]
            if m == "ROUTE1":
                route1_extra: List[str] = []
                if args.route1_pretrain_epochs >= 0:
                    route1_extra += ["--route1_pretrain_epochs", str(args.route1_pretrain_epochs)]
                if args.loss_w_adv >= 0:
                    route1_extra += ["--loss_w_adv", str(args.loss_w_adv)]
                if args.grl_alpha >= 0:
                    route1_extra += ["--grl_alpha", str(args.grl_alpha)]
                if args.routing_temp > 0:
                    route1_extra += ["--routing_temp", str(args.routing_temp)]
                cmd_train = ["python", "train.py", "--m_max", str(args.m_max)] + route1_extra + common_train
            elif m in {"ERM", "JTT", "CVAR_DRO", "GROUPDRO", "DIVERSIFY", "AMINET", "DACN", "AFRDN", "WPA_PDG"}:
                cmd_train = ["python", "train_baseline.py", "--method", m] + common_train
            else:
                print(f"Skip unsupported method: {m}")
                continue
            _run_cmd(cmd_train, cwd=base_dir)

            # 2) eval (full_eval + dataset/domain override)
            ckpt = train_dir / "best_model.pt"
            if not ckpt.exists():
                print(f"Skip eval, checkpoint missing: {ckpt}")
                continue

            common_eval = [
                "--ckpt",
                str(ckpt),
                "--output",
                str(eval_dir),
                "--dataset_override",
                c["test_dataset"],
                "--domains_override",
                c["test_domains"],
                "--known_classes",
                args.known_classes,
                "--unknown_classes",
                args.unknown_classes,
                "--full_eval",
            ]
            if args.self_processed_root.strip():
                common_eval += ["--self_processed_root", args.self_processed_root]
            if args.self_a_subdir.strip():
                common_eval += ["--self_a_subdir", args.self_a_subdir]
            if args.self_b_subdir.strip():
                common_eval += ["--self_b_subdir", args.self_b_subdir]
            if args.self_norm_mode.strip():
                common_eval += ["--self_norm_mode", args.self_norm_mode]
            if args.ood_score_type.strip():
                common_eval += ["--ood_score_type", args.ood_score_type]
            if args.ood_energy_temp > 0:
                common_eval += ["--ood_energy_temp", str(args.ood_energy_temp)]
            if m == "ROUTE1":
                cmd_eval = ["python", "test.py"] + common_eval
            else:
                cmd_eval = ["python", "test_baseline.py", "--method", m] + common_eval
            _run_cmd(cmd_eval, cwd=base_dir)

            metrics = _read_json(eval_dir / "test_metrics.json")
            row = {
                "case": c["case_name"],
                "method": m,
                "train_dataset": c["train_dataset"],
                "train_domains": c["train_domains"],
                "test_dataset": c["test_dataset"],
                "test_domains": c["test_domains"],
                "test_known_acc": metrics.get("test_known_acc", float("nan")),
                "test_open_auroc": metrics.get("test_open_auroc", float("nan")),
                "test_open_aupr": metrics.get("test_open_aupr", float("nan")),
                "test_open_fpr95": metrics.get("test_open_fpr95", float("nan")),
                "train_dir": str(train_dir),
                "eval_dir": str(eval_dir),
            }
            rows.append(row)

    _write_summary(rows, out_root)
    print(f"Scenario benchmark finished. Output dir: {out_root}")
    print(f"- {out_root / 'summary.csv'}")
    print(f"- {out_root / 'summary.md'}")
    print(f"- {out_root / 'summary.tex'}")


if __name__ == "__main__":
    main()

