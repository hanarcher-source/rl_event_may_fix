#!/usr/bin/env python3
# coding: utf-8
"""
Mean daily **rank IC** (Spearman ρ) on a **tail slice** of the cross-section.

Per trading day:
  1. Drop rows with missing Factor or T0_T1_RETURN.
  2. Sort by Factor (ascending).
  3. Take the bottom `n` and top `n` rows (union = up to 2n names).
  4. Spearman correlation between Factor and T0_T1_RETURN on that subset.
  5. Average Spearman across days (days with len < 2n or degenerate variance are skipped).

This matches the "top N + bottom N" tail rank IC documented in DIARY_XS_V2_ABLATION.md.

The **reported three-way ablation** (MLP vs frozen Mistral vs rolling Mistral) uses **N=100** only
(`--n 100`). Other `--n` values are for sensitivity analysis; keep N fixed when comparing to those headline results.

Usage:
  python3 metrics_tail_rank_ic.py /path/to/final_results.csv
  python3 metrics_tail_rank_ic.py /path/to/final_results.csv --n 100   # ablation setting
  python3 metrics_tail_rank_ic.py /path/to/final_results.csv --n 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def mean_daily_tail_rank_ic(
    df: pd.DataFrame,
    n: int,
    factor_col: str = "Factor",
    ret_col: str = "T0_T1_RETURN",
    date_col: str = "Date",
) -> float:
    daily: list[float] = []
    for _, g in df.groupby(date_col):
        g = g[[factor_col, ret_col]].dropna()
        if len(g) < 2 * n:
            continue
        g = g.sort_values(factor_col, ascending=True, kind="mergesort")
        tails = pd.concat([g.head(n), g.tail(n)], axis=0, ignore_index=True)
        if tails[factor_col].nunique() < 2 or tails[ret_col].nunique() < 2:
            continue
        rho, _ = spearmanr(tails[factor_col], tails[ret_col])
        if np.isfinite(rho):
            daily.append(float(rho))
    return float(np.mean(daily)) if daily else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Mean daily Spearman rank IC on bottom-n + top-n by Factor per day.")
    ap.add_argument("final_results_csv", type=Path, help="Path to final_results.csv")
    ap.add_argument("--n", type=int, default=100, help="Tail size each side (default 100 → union 200 names/day).")
    ap.add_argument("--factor-col", type=str, default="Factor")
    ap.add_argument("--ret-col", type=str, default="T0_T1_RETURN")
    args = ap.parse_args()

    df = pd.read_csv(args.final_results_csv)
    df["Date"] = pd.to_datetime(df["Date"])

    ic = mean_daily_tail_rank_ic(
        df,
        n=args.n,
        factor_col=args.factor_col,
        ret_col=args.ret_col,
    )
    n_days = df["Date"].nunique()
    print(f"file={args.final_results_csv}")
    print(f"rows={len(df)} distinct_dates={n_days}")
    print(f"mean_daily_tail_rank_ic_spearman  n={args.n}  (bottom-{args.n} + top-{args.n} by Factor): {ic:.6f}")


if __name__ == "__main__":
    main()
