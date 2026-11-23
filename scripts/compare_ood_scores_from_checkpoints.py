#!/usr/bin/env python3
"""Re-evaluate existing checkpoints with different OOD score types.

This script does NOT retrain. It only reruns test.py / test_baseline.py
on existing checkpoints under outputs_scenarios and aggregates results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


METRICS = [
    ("test_known_acc", "KnownAcc"),
    ("test_open_auroc", "AUROC"),
    ("test_open_aupr", "AUPR"),
    ("test_open_fpr95", "FPR95"),
]

METHOD_ORDER = [
    "ROUTE1",
    "ERM",
    "JTT",
    "CVAR_DRO",
    "GROUPDRO",
    "DIVERSIFY",
    "AMINET",
    "DACN",
    "AFRDN",
    "WPA_PDG",
]

B_TASKS = {
    "B_CWRU_train_0HP-2HP-3HP_test_1HP": ("B-1", "CWRU {0,2,3}HP -> CWRU 1HP"),
    "B_CWRU_train_0HP-1HP-2HP_test_3HP": ("B-2", "CWRU {0,1,2}HP -> CWRU 3HP"),
    "B_JNU_train_600-800_test_1000": ("B-3", "JNU {600,800}rpm -> JNU 1000rpm"),
    "B_JNU_train_800-1000_test_600": ("B-4", "JNU {800,1000}rpm -> JNU 600rpm"),
    "B_SELF_A_train_1200_test_1800": ("B-5", "In-house A 1200rpm -> In-house A 1800rpm"),
    "B_SELF_B_train_1800_test_1200": ("B-6", "In-house B 1800rpm -> In-house B 1200rpm"),
}

C_TASKS = {
    "C_CWRU_to_JNU": ("C-1", "CWRU (All loads) -> JNU (All speeds)"),
    "C_JNU_to_CWRU": ("C-2", "JNU (All speeds) -> CWRU (All loads)"),
    "C_SELF_A_to_SELF_B": ("C-3", "In-house A (1200,1800) -> In-house B (1200,1800)"),
    "C_SELF_B_to_SELF_A": ("C-4", "In-house B (1200,1800) -> In-house A (1200,1800)"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Compare OOD scores from existing checkpoints")
    p.add_argument("--outputs_root", type=str, default="outputs_scenarios")
    p.add_argument(
        "--methods",
        type=str,
        default="",
        help="Optional method filter, comma-separated, e.g. AFRDN,WPA_PDG",
    )
    p.add_argument(
        "--exp_prefixes",
        type=str,
        default="",
        help="Optional run-name prefix filter, comma-separated, e.g. tii319_new_c2j,tii319_new_j2c",
    )
    p.add_argument("--score_types", type=str, default="msp,energy,maxlogit")
    p.add_argument("--energy_temp", type=float, default=1.0)
    p.add_argument("--unknowns", type=str, default="1,2,3")
    p.add_argument("--seeds", type=str, default="41,42,43")
    p.add_argument("--settings", type=str, default="B,C", help="comma list from {B,C}")
    p.add_argument("--task_ids", type=str, default="", help="optional filter, e.g. B-2,C-1")
    p.add_argument("--python_bin", type=str, default="python")
    p.add_argument("--out_dir", type=str, default="outputs_scenarios/ood_score_compare")
    p.add_argument("--overwrite_eval", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    p.add_argument("--self_processed_root", type=str, default="")
    p.add_argument("--self_a_subdir", type=str, default="")
    p.add_argument("--self_b_subdir", type=str, default="")
    p.add_argument("--self_norm_mode", type=str, default="")
    return p.parse_args()


def _parse_int_csv(text: str) -> List[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def _parse_str_csv(text: str) -> List[str]:
    return [v.strip() for v in text.split(",") if v.strip()]


def _to_float(v: object) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _mean(vals: Sequence[float]) -> float:
    arr = [x for x in vals if not math.isnan(x)]
    if not arr:
        return float("nan")
    return float(sum(arr) / len(arr))


def _std(vals: Sequence[float]) -> float:
    arr = [x for x in vals if not math.isnan(x)]
    n = len(arr)
    if n == 0:
        return float("nan")
    if n == 1:
        return 0.0
    m = _mean(arr)
    return float(math.sqrt(sum((x - m) ** 2 for x in arr) / (n - 1)))


def _fmt_pm(m: float, s: float) -> str:
    if math.isnan(m):
        return "nan"
    if math.isnan(s):
        return f"{m:.4f} +/- nan"
    return f"{m:.4f} +/- {s:.4f}"


def _method_sort_key(method: str) -> Tuple[int, str]:
    m = str(method).upper()
    try:
        return (METHOD_ORDER.index(m), m)
    except ValueError:
        return (999, m)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _run(cmd: List[str], cwd: Path, dry_run: bool) -> None:
    print("RUN:", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _scan_run_dirs(
    outputs_root: Path,
    unknowns: Sequence[int],
    seeds: Sequence[int],
    settings: Sequence[str],
    exp_prefixes: Sequence[str],
) -> List[Dict[str, object]]:
    pat = re.compile(r"^(.+)_([BC])_u(\d+)_s(\d+)$")
    wanted_unknown = set(int(x) for x in unknowns)
    wanted_seed = set(int(x) for x in seeds)
    wanted_setting = set(s.upper() for s in settings)
    wanted_prefix = set(p.strip() for p in exp_prefixes if p.strip())
    out: List[Dict[str, object]] = []

    for p in sorted(outputs_root.iterdir()):
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        run_prefix = m.group(1)
        if wanted_prefix and run_prefix not in wanted_prefix:
            continue
        setting = m.group(2).upper()
        unknown = int(m.group(3))
        seed = int(m.group(4))
        if setting not in wanted_setting:
            continue
        if unknown not in wanted_unknown or seed not in wanted_seed:
            continue
        summary = p / "summary.csv"
        if not summary.exists():
            continue
        out.append({"run_dir": p, "setting": setting, "unknown": unknown, "seed": seed, "summary_csv": summary})
    return out


def _known_from_unknown(unknown: int) -> str:
    base = [0, 1, 2, 3]
    known = [x for x in base if x != int(unknown)]
    return ",".join(str(x) for x in known)


def _task_info(setting: str, case_name: str) -> Optional[Tuple[str, str]]:
    if setting == "B":
        return B_TASKS.get(case_name)
    return C_TASKS.get(case_name)


def _resolve_ckpt(run_dir: Path, case: str, method: str, train_dir_cell: str) -> Path:
    method_dir = method.lower()
    if train_dir_cell.strip():
        p = Path(train_dir_cell.strip()) / "best_model.pt"
        if p.exists():
            return p
    p2 = run_dir / case / method_dir / "train" / "best_model.pt"
    return p2


def _score_tag(score: str, energy_temp: float) -> str:
    if score == "energy":
        return f"energy_t{energy_temp:g}"
    return score


def _load_metrics(metrics_path: Path) -> Dict[str, float]:
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return {
        "test_known_acc": _to_float(d.get("test_known_acc", float("nan"))),
        "test_open_auroc": _to_float(d.get("test_open_auroc", float("nan"))),
        "test_open_aupr": _to_float(d.get("test_open_aupr", float("nan"))),
        "test_open_fpr95": _to_float(d.get("test_open_fpr95", float("nan"))),
    }


def _append_optional_self_args(cmd: List[str], args: argparse.Namespace) -> None:
    if args.self_processed_root.strip():
        cmd += ["--self_processed_root", args.self_processed_root.strip()]
    if args.self_a_subdir.strip():
        cmd += ["--self_a_subdir", args.self_a_subdir.strip()]
    if args.self_b_subdir.strip():
        cmd += ["--self_b_subdir", args.self_b_subdir.strip()]
    if args.self_norm_mode.strip():
        cmd += ["--self_norm_mode", args.self_norm_mode.strip()]


def _evaluate_all(args: argparse.Namespace, outputs_root: Path, project_root: Path) -> List[Dict[str, object]]:
    score_types = [s.lower() for s in _parse_str_csv(args.score_types)]
    method_filter = set(m.upper() for m in _parse_str_csv(args.methods))
    task_filter = set(v.upper() for v in _parse_str_csv(args.task_ids))
    run_infos = _scan_run_dirs(
        outputs_root=outputs_root,
        unknowns=_parse_int_csv(args.unknowns),
        seeds=_parse_int_csv(args.seeds),
        settings=[s.upper() for s in _parse_str_csv(args.settings)],
        exp_prefixes=_parse_str_csv(args.exp_prefixes),
    )

    records: List[Dict[str, object]] = []
    for ri in run_infos:
        run_dir: Path = ri["run_dir"]  # type: ignore[assignment]
        setting = str(ri["setting"])
        unknown = int(ri["unknown"])
        seed = int(ri["seed"])
        known_csv = _known_from_unknown(unknown)

        rows = _read_csv(Path(ri["summary_csv"]))
        for row in rows:
            case = str(row.get("case", "")).strip()
            method = str(row.get("method", "")).strip().upper()
            if method_filter and method not in method_filter:
                continue
            test_dataset = str(row.get("test_dataset", "")).strip()
            test_domains = str(row.get("test_domains", "")).strip()
            task = _task_info(setting, case)
            if task is None:
                continue
            task_id, task_desc = task
            if task_filter and task_id.upper() not in task_filter:
                continue

            ckpt = _resolve_ckpt(run_dir, case, method, str(row.get("train_dir", "")))
            if not ckpt.exists():
                print(f"[WARN] checkpoint missing: {ckpt}")
                continue

            for score in score_types:
                tag = _score_tag(score, args.energy_temp)
                eval_out = run_dir / case / method.lower() / f"eval_{tag}"
                metrics_path = eval_out / "test_metrics.json"
                read_metrics_path = metrics_path

                # Reuse existing default eval folder for MSP if available.
                if (score == "msp") and (not metrics_path.exists()) and (not args.overwrite_eval):
                    legacy_metrics = run_dir / case / method.lower() / "eval" / "test_metrics.json"
                    if legacy_metrics.exists():
                        read_metrics_path = legacy_metrics

                need_run = args.overwrite_eval or (not metrics_path.exists() and read_metrics_path == metrics_path)
                if need_run:
                    cmd: List[str]
                    if method == "ROUTE1":
                        cmd = [
                            args.python_bin,
                            "test.py",
                            "--ckpt",
                            str(ckpt),
                            "--output",
                            str(eval_out),
                            "--dataset_override",
                            test_dataset,
                            "--domains_override",
                            test_domains,
                            "--known_classes",
                            known_csv,
                            "--unknown_classes",
                            str(unknown),
                            "--ood_score_type",
                            score,
                            "--full_eval",
                        ]
                    else:
                        cmd = [
                            args.python_bin,
                            "test_baseline.py",
                            "--method",
                            method,
                            "--ckpt",
                            str(ckpt),
                            "--output",
                            str(eval_out),
                            "--dataset_override",
                            test_dataset,
                            "--domains_override",
                            test_domains,
                            "--known_classes",
                            known_csv,
                            "--unknown_classes",
                            str(unknown),
                            "--ood_score_type",
                            score,
                            "--full_eval",
                        ]
                    if score == "energy":
                        cmd += ["--ood_energy_temp", str(args.energy_temp)]
                    _append_optional_self_args(cmd, args)
                    _run(cmd, cwd=project_root, dry_run=args.dry_run)

                m = _load_metrics(read_metrics_path)
                if not m:
                    continue
                rec = {
                    "setting": setting,
                    "unknown": unknown,
                    "seed": seed,
                    "score_type": score,
                    "score_tag": tag,
                    "task_id": task_id,
                    "task_desc": task_desc,
                    "case": case,
                    "method": method,
                }
                rec.update(m)
                records.append(rec)
    return records


def _aggregate_seed(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    bucket: Dict[Tuple[str, str, int, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for r in records:
        k = (
            str(r["setting"]),
            str(r["score_type"]),
            int(r["unknown"]),
            str(r["task_id"]),
            str(r["task_desc"]),
            str(r["method"]),
        )
        bucket[k].append(r)

    out: List[Dict[str, object]] = []
    for (setting, score, unknown, task_id, task_desc, method), rows in bucket.items():
        row: Dict[str, object] = {
            "setting": setting,
            "score_type": score,
            "unknown": unknown,
            "task_id": task_id,
            "task_desc": task_desc,
            "method": method,
            "n_seeds": len(rows),
        }
        for k, nice in METRICS:
            vals = [_to_float(x.get(k, float("nan"))) for x in rows]
            m = _mean(vals)
            s = _std(vals)
            row[f"{k}_mean"] = m
            row[f"{k}_std"] = s
            row[nice] = _fmt_pm(m, s)
        out.append(row)
    out.sort(key=lambda r: (r["setting"], r["score_type"], int(r["unknown"]), r["task_id"], _method_sort_key(str(r["method"]))))
    return out


def _aggregate_unknown_rotation(seed_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    bucket: Dict[Tuple[str, str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for r in seed_rows:
        k = (
            str(r["setting"]),
            str(r["score_type"]),
            str(r["task_id"]),
            str(r["task_desc"]),
            str(r["method"]),
        )
        bucket[k].append(r)

    out: List[Dict[str, object]] = []
    for (setting, score, task_id, task_desc, method), rows in bucket.items():
        row: Dict[str, object] = {
            "setting": setting,
            "score_type": score,
            "task_id": task_id,
            "task_desc": task_desc,
            "method": method,
            "n_unknown": len(rows),
            "unknown_list": ",".join(str(int(x["unknown"])) for x in sorted(rows, key=lambda z: int(z["unknown"]))),
        }
        for k, nice in METRICS:
            vals = [_to_float(x.get(f"{k}_mean", float("nan"))) for x in rows]
            m = _mean(vals)
            s = _std(vals)
            row[f"{k}_urot_mean"] = m
            row[f"{k}_urot_std"] = s
            row[nice] = _fmt_pm(m, s)
        out.append(row)
    out.sort(key=lambda r: (r["setting"], r["score_type"], r["task_id"], _method_sort_key(str(r["method"]))))
    return out


def _aggregate_unknown_rotation_overall(seed_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    # per (setting, score, method, unknown): mean over tasks
    by_sm_u: Dict[Tuple[str, str, str, int], List[Dict[str, object]]] = defaultdict(list)
    for r in seed_rows:
        by_sm_u[(str(r["setting"]), str(r["score_type"]), str(r["method"]), int(r["unknown"]))].append(r)

    tmp: List[Dict[str, object]] = []
    for (setting, score, method, unknown), rows in by_sm_u.items():
        row = {"setting": setting, "score_type": score, "method": method, "unknown": unknown, "n_tasks": len(rows)}
        for k, _ in METRICS:
            vals = [_to_float(x.get(f"{k}_mean", float("nan"))) for x in rows]
            row[f"{k}_taskavg"] = _mean(vals)
        tmp.append(row)

    by_sm: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for r in tmp:
        by_sm[(str(r["setting"]), str(r["score_type"]), str(r["method"]))].append(r)

    out: List[Dict[str, object]] = []
    for (setting, score, method), rows in by_sm.items():
        row: Dict[str, object] = {
            "setting": setting,
            "score_type": score,
            "method": method,
            "n_unknown": len(rows),
            "unknown_list": ",".join(str(int(x["unknown"])) for x in sorted(rows, key=lambda z: int(z["unknown"]))),
        }
        for k, nice in METRICS:
            vals = [_to_float(x.get(f"{k}_taskavg", float("nan"))) for x in rows]
            m = _mean(vals)
            s = _std(vals)
            row[f"{k}_urot_mean"] = m
            row[f"{k}_urot_std"] = s
            row[nice] = _fmt_pm(m, s)
        out.append(row)
    out.sort(key=lambda r: (r["setting"], r["score_type"], _method_sort_key(str(r["method"]))))
    return out


def _write_markdown(path: Path, title: str, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(f, "")) for f in fields) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    outputs_root = Path(args.outputs_root)
    if not outputs_root.is_absolute():
        outputs_root = (project_root / outputs_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _evaluate_all(args, outputs_root=outputs_root, project_root=project_root)
    detail_fields = ["setting", "score_type", "unknown", "seed", "task_id", "task_desc", "case", "method"] + [k for k, _ in METRICS]
    _write_csv(out_dir / "ood_score_detail.csv", records, detail_fields)

    seed_rows = _aggregate_seed(records)
    seed_fields = [
        "setting",
        "score_type",
        "unknown",
        "task_id",
        "task_desc",
        "method",
        "n_seeds",
        "test_known_acc_mean",
        "test_known_acc_std",
        "test_open_auroc_mean",
        "test_open_auroc_std",
        "test_open_aupr_mean",
        "test_open_aupr_std",
        "test_open_fpr95_mean",
        "test_open_fpr95_std",
        "KnownAcc",
        "AUROC",
        "AUPR",
        "FPR95",
    ]
    _write_csv(out_dir / "ood_score_seedavg_by_unknown_task.csv", seed_rows, seed_fields)

    task_urot = _aggregate_unknown_rotation(seed_rows)
    task_urot_fields = [
        "setting",
        "score_type",
        "task_id",
        "task_desc",
        "method",
        "n_unknown",
        "unknown_list",
        "test_known_acc_urot_mean",
        "test_known_acc_urot_std",
        "test_open_auroc_urot_mean",
        "test_open_auroc_urot_std",
        "test_open_aupr_urot_mean",
        "test_open_aupr_urot_std",
        "test_open_fpr95_urot_mean",
        "test_open_fpr95_urot_std",
        "KnownAcc",
        "AUROC",
        "AUPR",
        "FPR95",
    ]
    _write_csv(out_dir / "ood_score_unknown_rotation_by_task.csv", task_urot, task_urot_fields)

    overall = _aggregate_unknown_rotation_overall(seed_rows)
    overall_fields = [
        "setting",
        "score_type",
        "method",
        "n_unknown",
        "unknown_list",
        "test_known_acc_urot_mean",
        "test_known_acc_urot_std",
        "test_open_auroc_urot_mean",
        "test_open_auroc_urot_std",
        "test_open_aupr_urot_mean",
        "test_open_aupr_urot_std",
        "test_open_fpr95_urot_mean",
        "test_open_fpr95_urot_std",
        "KnownAcc",
        "AUROC",
        "AUPR",
        "FPR95",
    ]
    _write_csv(out_dir / "ood_score_unknown_rotation_overall.csv", overall, overall_fields)

    _write_markdown(
        out_dir / "ood_score_unknown_rotation_overall.md",
        title="OOD Score Comparison (Unknown-Rotation Overall)",
        rows=overall,
        fields=["setting", "score_type", "method", "KnownAcc", "AUROC", "AUPR", "FPR95", "n_unknown"],
    )
    _write_markdown(
        out_dir / "ood_score_unknown_rotation_by_task.md",
        title="OOD Score Comparison (Unknown-Rotation By Task)",
        rows=task_urot,
        fields=["setting", "score_type", "task_id", "method", "KnownAcc", "AUROC", "AUPR", "FPR95", "n_unknown"],
    )

    print(f"[DONE] out_dir: {out_dir}")
    print(f"[DONE] detail: {out_dir / 'ood_score_detail.csv'}")
    print(f"[DONE] seed avg by unknown-task: {out_dir / 'ood_score_seedavg_by_unknown_task.csv'}")
    print(f"[DONE] unknown-rotation by task: {out_dir / 'ood_score_unknown_rotation_by_task.csv'}")
    print(f"[DONE] unknown-rotation overall: {out_dir / 'ood_score_unknown_rotation_overall.csv'}")


if __name__ == "__main__":
    main()
