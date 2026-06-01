#!/usr/bin/env python3
"""Summarize MVP results under results/logs/<dataset>/<note>/seed_*.npy."""

import argparse
import re
from pathlib import Path

import numpy as np


def discover_seeds(result_dir: Path) -> list[int]:
    seeds = []
    for p in result_dir.glob("seed_*.npy"):
        m = re.fullmatch(r"seed_(\d+)\.npy", p.name)
        if m:
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def metrics_for_seed(result_dir: Path, seed: int) -> dict:
    task_acc = np.load(result_dir / f"seed_{seed}.npy")
    eval_path = result_dir / f"seed_{seed}_eval.npy"
    a_auc = float(np.mean(np.load(eval_path))) if eval_path.exists() else float("nan")
    return {
        "A_avg": float(np.mean(task_acc)),
        "A_last": float(task_acc[-1]),
        "A_auc": a_auc,
        "task_acc": task_acc,
    }


def summarize_config(result_dir: Path, seeds: list[int] | None) -> dict | None:
    seeds = seeds or discover_seeds(result_dir)
    per_seed = []
    for seed in seeds:
        if not (result_dir / f"seed_{seed}.npy").exists():
            continue
        per_seed.append(metrics_for_seed(result_dir, seed))
    if not per_seed:
        return None

    agg = {}
    for k in ("A_avg", "A_last", "A_auc"):
        vals = np.array([m[k] for m in per_seed])
        agg[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
    agg["n_seeds"] = len(per_seed)
    return agg


def discover_configs(
    logs_root: Path,
    datasets: list[str] | None,
    note: str | None,
) -> list[tuple[str, str, Path]]:
    """Return (dataset_name, note, result_dir) for dirs containing seed_*.npy."""
    if datasets:
        ds_dirs = [logs_root / d for d in datasets]
    else:
        ds_dirs = sorted(logs_root.iterdir())

    out: list[tuple[str, str, Path]] = []
    for ds_dir in ds_dirs:
        if not ds_dir.is_dir():
            continue
        for cfg_dir in sorted(ds_dir.iterdir()):
            if not cfg_dir.is_dir():
                continue
            if note is not None and cfg_dir.name != note:
                continue
            if any(cfg_dir.glob("seed_*.npy")):
                out.append((ds_dir.name, cfg_dir.name, cfg_dir))
    return out


def print_config_detail(dataset: str, note: str, result_dir: Path, seeds: list[int] | None):
    seeds = seeds or discover_seeds(result_dir)
    print(f"\n{'=' * 72}\n{dataset} / {note}\n{'=' * 72}")
    print(f"{'seed':>6}  {'A_avg':>8}  {'A_last':>8}  {'A_auc':>8}")
    per_seed = []
    for seed in seeds:
        path = result_dir / f"seed_{seed}.npy"
        if not path.exists():
            print(f"{seed:>6}  (missing)")
            continue
        m = metrics_for_seed(result_dir, seed)
        per_seed.append(m)
        print(f"{seed:>6}  {m['A_avg']:8.4f}  {m['A_last']:8.4f}  {m['A_auc']:8.4f}")
    if not per_seed:
        print("(no results)")
        return
    agg = summarize_config(result_dir, seeds)
    print(f"\nmean over {agg['n_seeds']} seeds (± std):")
    for k in ("A_avg", "A_last", "A_auc"):
        a = agg[k]
        print(f"  {k}: {a['mean']:.4f} ± {a['std']:.4f}")


def _metric_fmt(agg: dict, key: str) -> str:
    a = agg[key]
    return f"{a['mean']:.4f}±{a['std']:.4f}"


def print_table(rows: list[tuple[str, str, dict]], note_width: int | None = None):
    metric_w = 14
    ds_w = max(len("dataset"), max((len(d) for d, _, _ in rows), default=0), 12)
    natural_note_w = max(len("note"), max((len(n) for _, n, _ in rows), default=0))
    if note_width is None:
        note_w = natural_note_w
    else:
        note_w = max(note_width, len("note"))

    header = (
        f"{'dataset':<{ds_w}}  {'note':<{note_w}}  {'n':>3}  "
        f"{'A_avg':>{metric_w}}  {'A_last':>{metric_w}}  {'A_auc':>{metric_w}}"
    )
    print()
    print(header)
    print("-" * len(header))
    for dataset, note, agg in rows:
        note_cell = note if len(note) <= note_w else note[: max(note_w - 1, 0)] + "…"
        print(
            f"{dataset:<{ds_w}}  {note_cell:<{note_w}}  {agg['n_seeds']:>3}  "
            f"{_metric_fmt(agg, 'A_avg'):>{metric_w}}  "
            f"{_metric_fmt(agg, 'A_last'):>{metric_w}}  "
            f"{_metric_fmt(agg, 'A_auc'):>{metric_w}}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=None,
        help="dataset subdir(s) under logs/ (default: all)",
    )
    parser.add_argument("--note", type=str, default=None, help="filter to one config name")
    parser.add_argument("--log_path", type=str, default="results")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--note_width",
        type=int,
        default=None,
        help="cap note column width (default: fit longest note)",
    )
    args = parser.parse_args()

    logs_root = Path(args.log_path) / "logs"
    if not logs_root.is_dir():
        raise FileNotFoundError(f"not found: {logs_root}")

    configs = discover_configs(logs_root, args.dataset, args.note)
    if not configs:
        scope = args.dataset or ["*"]
        raise SystemExit(f"no configs with seed_*.npy under {logs_root} (datasets={scope}, note={args.note})")

    rows: list[tuple[str, str, dict]] = []
    for dataset, note, result_dir in configs:
        agg = summarize_config(result_dir, args.seeds)
        if agg is None:
            print(f"[skip] {dataset}/{note}: no seed_*.npy")
            continue
        rows.append((dataset, note, agg))
        if args.verbose or args.note:
            print_config_detail(dataset, note, result_dir, args.seeds)

    if not rows:
        raise SystemExit("no results loaded")

    if len(rows) > 1 or not args.verbose:
        print_table(rows, note_width=args.note_width)
    elif len(rows) == 1 and args.verbose:
        pass  # detail only


if __name__ == "__main__":
    main()
