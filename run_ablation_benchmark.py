"""
Route1 ablation benchmark runner.

Ablation variants:
1) WO_NATURAL_COLLAPSE
2) WO_CONFIDENCE_WEIGHTING
3) WO_WORST_STEP
4) WO_RECONSTRUCTION_ROUTING
5) SOFT_ONLY_ROUTING
6) EQCAP_NO_ENV
"""

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Route1 Ablation Benchmark")
    p.add_argument("--scenario", type=str, default="B_AND_C", choices=["B_CROSS_DOMAIN", "C_CROSS_DATASET", "B_AND_C"])
    p.add_argument(
        "--variants",
        type=str,
        default="FULL,WO_NATURAL_COLLAPSE,WO_CONFIDENCE_WEIGHTING,WO_WORST_STEP,WO_RECONSTRUCTION_ROUTING,SOFT_ONLY_ROUTING,EQCAP_NO_ENV",
        help="Comma separated ablation list.",
    )

    p.add_argument("--dataset", type=str, default="CWRU", choices=["CWRU", "JNU", "SELF_A", "SELF_B"])
    p.add_argument("--domains", type=str, default="")
    p.add_argument("--train_dataset", type=str, default="CWRU", choices=["CWRU", "JNU", "SELF_A", "SELF_B"])
    p.add_argument("--test_dataset", type=str, default="JNU", choices=["CWRU", "JNU", "SELF_A", "SELF_B"])
    p.add_argument("--train_domains", type=str, default="")
    p.add_argument("--test_domains", type=str, default="")

    p.add_argument("--known_classes", type=str, default="0,1,3")
    p.add_argument("--unknown_classes", type=str, default="2")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--m_max", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=str, default="", help="Comma separated seeds, e.g. 41,42,43")
    p.add_argument("--lr", type=float, default=-1.0)
    p.add_argument("--weight_decay", type=float, default=-1.0)
    p.add_argument("--route1_pretrain_epochs", type=int, default=-1)
    p.add_argument("--loss_w_adv", type=float, default=-1.0)
    p.add_argument("--grl_alpha", type=float, default=-1.0)
    p.add_argument("--routing_temp", type=float, default=-1.0)
    p.add_argument("--ood_score_type", type=str, default="", choices=["", "msp", "energy", "maxlogit"])
    p.add_argument("--ood_energy_temp", type=float, default=-1.0)

    p.add_argument("--self_processed_root", type=str, default="")
    p.add_argument("--self_a_subdir", type=str, default="")
    p.add_argument("--self_b_subdir", type=str, default="")
    p.add_argument("--self_norm_mode", type=str, default="", choices=["", "train_global", "per_sample"])

    p.add_argument("--exp_name", type=str, default="ablation_route1")
    return p.parse_args()


def _run(cmd: List[str], cwd: Path) -> None:
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

    if args.scenario in {"B_CROSS_DOMAIN", "B_AND_C"}:
        domains = _parse_domains(args.domains, args.dataset)
        if len(domains) < 2:
            raise ValueError("B_CROSS_DOMAIN requires >=2 domains.")
        for heldout in domains:
            train_list = [d for d in domains if d != heldout]
            cases.append(
                {
                    "scenario": "B_CROSS_DOMAIN",
                    "case_name": f"B_{args.dataset}_train_{'-'.join(train_list)}_test_{heldout}",
                    "train_dataset": args.dataset,
                    "train_domains": ",".join(train_list),
                    "test_dataset": args.dataset,
                    "test_domains": heldout,
                }
            )

    if args.scenario in {"C_CROSS_DATASET", "B_AND_C"}:
        train_domains = _parse_domains(args.train_domains, args.train_dataset)
        test_domains = _parse_domains(args.test_domains, args.test_dataset)
        cases.append(
            {
                "scenario": "C_CROSS_DATASET",
                "case_name": f"C_{args.train_dataset}_to_{args.test_dataset}",
                "train_dataset": args.train_dataset,
                "train_domains": ",".join(train_domains),
                "test_dataset": args.test_dataset,
                "test_domains": ",".join(test_domains),
            }
        )

    return cases


def _norm_variant(v: str) -> str:
    t = v.strip().upper()
    if t in {"", "BASE", "ROUTE1", "FULL", "NONE"}:
        return "FULL"
    return t


def _variant_to_train_arg(v: str) -> str:
    if v == "FULL":
        return "NONE"
    return v


def _parse_seeds(args: argparse.Namespace) -> List[int]:
    if not args.seeds.strip():
        return [int(args.seed)]
    out = []
    for s in args.seeds.split(","):
        s = s.strip()
        if s:
            out.append(int(s))
    if not out:
        out = [int(args.seed)]
    return out


def _to_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _mean(vals: List[float]) -> float:
    xs = [x for x in vals if not math.isnan(x)]
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def _std(vals: List[float]) -> float:
    xs = [x for x in vals if not math.isnan(x)]
    n = len(xs)
    if n <= 1:
        return 0.0 if n == 1 else float("nan")
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var)


def _fmt_pm(m: float, s: float, ndigits: int = 4) -> str:
    if math.isnan(m):
        return "nan"
    if math.isnan(s):
        return f"{m:.{ndigits}f}±nan"
    return f"{m:.{ndigits}f}±{s:.{ndigits}f}"


def _write_mean_std(rows: List[Dict], out_dir: Path) -> None:
    metrics = ["test_known_acc", "test_open_auroc", "test_open_aupr", "test_open_fpr95"]

    by_case: Dict[Tuple[str, str, str], List[Dict]] = {}
    for r in rows:
        key = (str(r["scenario"]), str(r["case"]), str(r["variant"]))
        by_case.setdefault(key, []).append(r)

    by_case_rows = []
    for (scenario, case, variant), rs in sorted(by_case.items()):
        out = {"scenario": scenario, "case": case, "variant": variant, "n_runs": len(rs)}
        for m in metrics:
            vals = [_to_float(x.get(m, "nan")) for x in rs]
            out[m] = _fmt_pm(_mean(vals), _std(vals))
        by_case_rows.append(out)

    case_fields = ["scenario", "case", "variant", "n_runs", *metrics]
    with (out_dir / "summary_mean_pm_std_by_case.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=case_fields)
        w.writeheader()
        for r in by_case_rows:
            w.writerow({k: r.get(k, "") for k in case_fields})

    md_case = [
        "| Scenario | Case | Variant | N | Acc | AUROC | AUPR | FPR95 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in by_case_rows:
        md_case.append(
            f"| {r['scenario']} | {r['case']} | {r['variant']} | {r['n_runs']} | "
            f"{r['test_known_acc']} | {r['test_open_auroc']} | {r['test_open_aupr']} | {r['test_open_fpr95']} |"
        )
    (out_dir / "summary_mean_pm_std_by_case.md").write_text("\n".join(md_case), encoding="utf-8")

    by_overall: Dict[Tuple[str, str], List[Dict]] = {}
    for r in rows:
        key = (str(r["scenario"]), str(r["variant"]))
        by_overall.setdefault(key, []).append(r)

    overall_rows = []
    for (scenario, variant), rs in sorted(by_overall.items()):
        out = {"scenario": scenario, "variant": variant, "n_runs": len(rs)}
        for m in metrics:
            vals = [_to_float(x.get(m, "nan")) for x in rs]
            out[m] = _fmt_pm(_mean(vals), _std(vals))
        overall_rows.append(out)

    overall_fields = ["scenario", "variant", "n_runs", *metrics]
    with (out_dir / "summary_mean_pm_std_overall.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=overall_fields)
        w.writeheader()
        for r in overall_rows:
            w.writerow({k: r.get(k, "") for k in overall_fields})

    md_overall = [
        "| Scenario | Variant | N | Acc | AUROC | AUPR | FPR95 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in overall_rows:
        md_overall.append(
            f"| {r['scenario']} | {r['variant']} | {r['n_runs']} | "
            f"{r['test_known_acc']} | {r['test_open_auroc']} | {r['test_open_aupr']} | {r['test_open_fpr95']} |"
        )
    (out_dir / "summary_mean_pm_std_overall.md").write_text("\n".join(md_overall), encoding="utf-8")


def _write_summary(rows: List[Dict], out_dir: Path) -> None:
    fields = [
        "scenario",
        "case",
        "variant",
        "seed",
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
        "| Scenario | Case | Variant | Seed | Test Acc | Test AUROC | Test AUPR | Test FPR95 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['scenario']} | {r['case']} | {r['variant']} | {r['seed']} | {float(r['test_known_acc']):.4f} | "
            f"{float(r['test_open_auroc']):.4f} | {float(r['test_open_aupr']):.4f} | {float(r['test_open_fpr95']):.4f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")

    tex = [
        "\\begin{tabular}{llllcccc}",
        "\\hline",
        "Scenario & Case & Variant & Seed & Test Acc & Test AUROC & Test AUPR & Test FPR95 \\\\",
        "\\hline",
    ]
    for r in rows:
        tex.append(
            f"{r['scenario']} & {r['case']} & {r['variant']} & {r['seed']} & "
            f"{float(r['test_known_acc']):.4f} & {float(r['test_open_auroc']):.4f} & "
            f"{float(r['test_open_aupr']):.4f} & {float(r['test_open_fpr95']):.4f} \\\\"
        )
    tex += ["\\hline", "\\end{tabular}"]
    (out_dir / "summary.tex").write_text("\n".join(tex), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    seeds = _parse_seeds(args)

    variants = [_norm_variant(v) for v in args.variants.split(",") if v.strip()]
    allowed = {
        "FULL",
        "WO_NATURAL_COLLAPSE",
        "WO_CONFIDENCE_WEIGHTING",
        "WO_WORST_STEP",
        "WO_RECONSTRUCTION_ROUTING",
        "SOFT_ONLY_ROUTING",
        "EQCAP_NO_ENV",
    }
    bad = [v for v in variants if v not in allowed]
    if bad:
        raise ValueError(f"Unsupported variants: {bad}")

    cases = _build_cases(args)
    out_root = base_dir / "outputs_ablation" / args.exp_name
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for c in cases:
        case_dir = out_root / c["case_name"]
        case_dir.mkdir(parents=True, exist_ok=True)

        for v in variants:
            for seed in seeds:
                train_dir = case_dir / v.lower() / f"seed_{seed}" / "train"
                eval_dir = case_dir / v.lower() / f"seed_{seed}" / "eval"
                train_dir.mkdir(parents=True, exist_ok=True)
                eval_dir.mkdir(parents=True, exist_ok=True)

                common_train = [
                    "--dataset",
                    c["train_dataset"],
                    "--domains",
                    c["train_domains"],
                    "--epochs",
                    str(args.epochs),
                    "--batch_size",
                    str(args.batch_size),
                    "--m_max",
                    str(args.m_max),
                    "--seed",
                    str(seed),
                    "--known_classes",
                    args.known_classes,
                    "--unknown_classes",
                    args.unknown_classes,
                    "--ablation_variant",
                    _variant_to_train_arg(v),
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
                if args.route1_pretrain_epochs >= 0:
                    common_train += ["--route1_pretrain_epochs", str(args.route1_pretrain_epochs)]
                if args.loss_w_adv >= 0:
                    common_train += ["--loss_w_adv", str(args.loss_w_adv)]
                if args.grl_alpha >= 0:
                    common_train += ["--grl_alpha", str(args.grl_alpha)]
                if args.routing_temp > 0:
                    common_train += ["--routing_temp", str(args.routing_temp)]
                if args.ood_score_type.strip():
                    common_train += ["--ood_score_type", args.ood_score_type]
                if args.ood_energy_temp > 0:
                    common_train += ["--ood_energy_temp", str(args.ood_energy_temp)]

                _run(["python", "train.py"] + common_train, cwd=base_dir)

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

                _run(["python", "test.py"] + common_eval, cwd=base_dir)

                metrics = _read_json(eval_dir / "test_metrics.json")
                rows.append(
                    {
                        "scenario": c["scenario"],
                        "case": c["case_name"],
                        "variant": v,
                        "seed": seed,
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
                )

    _write_summary(rows, out_root)
    _write_mean_std(rows, out_root)
    print(f"Ablation benchmark finished. Output dir: {out_root}")
    print(f"- {out_root / 'summary.csv'}")
    print(f"- {out_root / 'summary.md'}")
    print(f"- {out_root / 'summary.tex'}")
    print(f"- {out_root / 'summary_mean_pm_std_by_case.csv'}")
    print(f"- {out_root / 'summary_mean_pm_std_overall.csv'}")


if __name__ == "__main__":
    main()
