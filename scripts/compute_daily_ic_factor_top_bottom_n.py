#!/usr/bin/env python3
"""Mean daily Pearson IC (Factor vs T0_T1_RETURN), each day only on top-N + bottom-N stocks by Factor."""
from __future__ import annotations

import argparse
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def mean_daily_ic_top_bottom_n(
    df: pd.DataFrame,
    top_n: int,
    bottom_n: int,
    factor_col: str = "Factor",
    return_col: str = "T0_T1_RETURN",
) -> tuple[float, int]:
    daily_ics: List[float] = []
    n_used_days = 0
    for _, g0 in df.groupby("Date"):
        g = g0[[factor_col, return_col]].dropna()
        n = int(len(g))
        if n < max(20, top_n + bottom_n + 1):
            continue
        g = g.sort_values(by=factor_col, ascending=True, kind="mergesort")
        lo = g.head(bottom_n)
        hi = g.tail(top_n)
        tails = pd.concat([lo, hi], axis=0, ignore_index=True)
        if tails[factor_col].nunique() > 1 and tails[return_col].nunique() > 1:
            ic, _ = pearsonr(tails[factor_col], tails[return_col])
            if np.isfinite(ic):
                daily_ics.append(float(ic))
                n_used_days += 1
    return (float(np.mean(daily_ics)) if daily_ics else float("nan"), n_used_days)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_results_csv", type=str, required=True)
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--bottom_n", type=int, default=50)
    args = ap.parse_args()

    df = pd.read_csv(args.final_results_csv)
    if "Date" in df.columns and not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"])

    ic, n_days = mean_daily_ic_top_bottom_n(df, args.top_n, args.bottom_n)
    print(f"mean_daily_ic_top{args.top_n}_bot{args.bottom_n}={ic:.6f}  days_used={n_days}  rows={len(df)}")


if __name__ == "__main__":
    main()
