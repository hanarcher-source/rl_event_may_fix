#!/usr/bin/env python3
# coding: utf-8
"""
MLP ablation aligned with LLM train1m-frozen + fast OOS:

- Train **only** the first month (same train/val as roll 1 of the rolling MLP).
- **Frozen** weights for **n_rolls** OOS test months (test = start_train_month + k + 2).

**OOS inference (default `combo_cache`)**: features use month sin/cos from calendar month, so uniqueness is
**(test_month, combo_key)**. We union those pairs across all OOS months, run the MLP in batches once, then
map factors to every row. (`per_month` = call `predict_factors_mlp` per month.)

Outputs match `roll_xs_bucket_factor_mlp.py`: final_results.csv, factor_matrix.csv, roll_meta.json, final/mlp.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import roll_xs_bucket_factor_mlp as mlp


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("roll_xs_mlp_train1m_frozen_oos12")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _placeholder_date_for_period(period: pd.Period) -> str:
    return f"{period.year}-{period.month:02d}-15"


@torch.no_grad()
def predict_oos_mlp_combo_cache(
    model: torch.nn.Module,
    test_frames: List[pd.DataFrame],
    meta_pre: List[dict],
    event_cols: List[str],
    device: torch.device,
    infer_batch: int,
    logger: logging.Logger,
) -> List[pd.DataFrame]:
    """One forward per unique (month placeholder, combo_key); map factors to all rows."""
    pairs_order: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for df_test, m in zip(test_frames, meta_pre):
        period = pd.Period(m["test_month"], freq="M")
        ds_ph = _placeholder_date_for_period(period)
        for ck in df_test["combo_key"].astype(str).unique():
            key = (ds_ph, ck)
            if key not in seen:
                seen.add(key)
                pairs_order.append(key)

    eidx = mlp.event_idx_map(event_cols)
    d = len(event_cols) + 2
    X = np.zeros((len(pairs_order), d), dtype=np.float32)
    for i, (ds_ph, ck) in enumerate(pairs_order):
        X[i] = mlp.row_to_feat(ds_ph, ck, eidx, d)

    logger.info(
        "[frozen OOS combo_cache] unique (month_placeholder, combo_key) across %d test months: %d",
        len(test_frames),
        len(pairs_order),
    )
    model.eval()
    factors: List[float] = []
    for s in tqdm(range(0, len(X), infer_batch), desc="mlp_combo_infer"):
        xb = torch.from_numpy(X[s : s + infer_batch]).to(device)
        factors.extend(mlp.logits_to_factor(model(xb)).cpu().numpy().tolist())
    if len(factors) != len(pairs_order):
        raise RuntimeError(f"infer len mismatch: factors={len(factors)} pairs={len(pairs_order)}")
    fac_map = {pairs_order[i]: float(factors[i]) for i in range(len(pairs_order))}

    out_frames: List[pd.DataFrame] = []
    for df_test, m in zip(test_frames, meta_pre):
        period = pd.Period(m["test_month"], freq="M")
        ds_ph = _placeholder_date_for_period(period)
        pred_df = df_test.copy()
        pred_df["Factor"] = [fac_map[(ds_ph, str(c))] for c in pred_df["combo_key"].astype(str)]
        pred_df = pred_df[["Date", "Stock", "Factor", "T0_T1_RETURN"]].copy()
        pred_df["roll"] = m["roll"]
        pred_df["train_month"] = m["train_month"]
        pred_df["test_month"] = m["test_month"]
        out_frames.append(pred_df)
    return out_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_train_month", type=str, default=mlp.START_TRAIN_MONTH)
    ap.add_argument("--n_rolls", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=mlp.HIDDEN)
    ap.add_argument("--dropout", type=float, default=mlp.DROPOUT)
    ap.add_argument("--lr", type=float, default=mlp.LR)
    ap.add_argument("--max_steps_per_month", type=int, default=mlp.MAX_STEPS_PER_MONTH)
    ap.add_argument("--resume_ckpt", type=str, default=None)
    ap.add_argument("--prepend_results_csv", type=str, default=None)
    ap.add_argument(
        "--oos_infer",
        choices=("combo_cache", "per_month"),
        default="combo_cache",
        help="combo_cache: one MLP pass per unique (test_month, combo_key); per_month: legacy per-month dedupe.",
    )
    ap.add_argument(
        "--infer_batch",
        type=int,
        default=4096,
        help="Batch size for MLP forwards (combo_cache and per_month predict).",
    )
    args = ap.parse_args()

    run_id = "mlp_xs_train1m_frozen_oos12_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(mlp.BASE_DIR, "results", run_id)
    log_dir = os.path.join(mlp.BASE_DIR, "logs", run_id)
    final_ckpt_dir = os.path.join(mlp.BASE_DIR, "mappings", run_id, "final")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(final_ckpt_dir, exist_ok=True)

    logger = setup_logger(os.path.join(log_dir, "run.log"))
    rng = np.random.default_rng(mlp.SEED)
    torch.manual_seed(mlp.SEED)

    final_csv = os.path.join(out_dir, "final_results.csv")
    factor_csv = os.path.join(out_dir, "factor_matrix.csv")
    meta_json = os.path.join(out_dir, "roll_meta.json")
    ckpt_path = os.path.join(final_ckpt_dir, "mlp.pt")

    logger.info(
        "pipeline_variant=mlp_train1month_frozen_oos12 oos_infer=%s run_id=%s start_train_month=%s n_rolls=%d",
        args.oos_infer,
        run_id,
        args.start_train_month,
        args.n_rolls,
    )

    data = mlp.load_data_xs(logger)
    event_cols = list(data.event_cols)
    df_all = mlp.attach_combo_keys(data.df_events, event_cols)
    ret_str = data.ret_clean.copy()
    ret_str.index = ret_str.index.strftime("%Y-%m-%d")

    in_dim = len(event_cols) + 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s in_dim=%d", device, in_dim)

    model = mlp.TrivialMLP(in_dim, args.hidden, mlp.N_CLASS, args.dropout).to(device)
    if args.resume_ckpt:
        blob = torch.load(args.resume_ckpt, map_location="cpu")
        model.load_state_dict(blob["state_dict"])
        logger.info("Loaded MLP weights from %s", args.resume_ckpt)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=mlp.WEIGHT_DECAY)

    start_p = pd.Period(args.start_train_month, freq="M")
    all_results: List[pd.DataFrame] = []
    meta_rows: List[dict] = []

    k0 = 0
    tr_p0 = start_p + k0
    va_p0 = start_p + k0 + 1
    te_p0 = start_p + k0 + 2
    df_train = df_all.loc[mlp.month_mask(df_all, tr_p0)].copy()
    df_val = df_all.loc[mlp.month_mask(df_all, va_p0)].copy()
    df_test0 = df_all.loc[mlp.month_mask(df_all, te_p0)].copy()

    if df_train.empty or df_val.empty or df_test0.empty:
        logger.error("Missing data for roll 1 (train=%s val=%s test=%s)", tr_p0, va_p0, te_p0)
        return

    lab_tr = mlp.build_combo_day_labels(df_train, ret_str, mlp.MIN_UNIV, mlp.MIN_COMBO_STOCKS)
    lab_va = mlp.build_combo_day_labels(df_val, ret_str, mlp.MIN_UNIV, mlp.MIN_COMBO_STOCKS)
    if len(lab_tr) < 100:
        logger.error("Too few train labels: %d", len(lab_tr))
        return
    if len(lab_tr) > mlp.MAX_TRAIN_SAMPLES:
        lab_tr = lab_tr.sample(
            n=mlp.MAX_TRAIN_SAMPLES, random_state=int(rng.integers(0, 2**31 - 1))
        ).reset_index(drop=True)

    X_tr = mlp.lab_to_X(lab_tr, event_cols)
    X_va = mlp.lab_to_X(lab_va, event_cols)

    logger.info(
        "[train once] train=%s val=%s | first OOS=%s | lab_tr=%d lab_va=%d | first test_rows=%d",
        tr_p0,
        va_p0,
        te_p0,
        len(lab_tr),
        len(lab_va),
        len(df_test0),
    )
    mlp.train_one_month_mlp(
        model,
        opt,
        X_tr,
        lab_tr,
        X_va,
        lab_va,
        device,
        logger,
        1,
        args.max_steps_per_month,
    )

    test_frames: List[pd.DataFrame] = []
    meta_pre: List[dict] = []
    for k in range(args.n_rolls):
        tr_p = start_p + k
        va_p = start_p + k + 1
        te_p = start_p + k + 2
        df_test = df_all.loc[mlp.month_mask(df_all, te_p)].copy()
        if df_test.empty:
            logger.error("Missing test month %s (roll %d)", te_p, k + 1)
            break
        test_frames.append(df_test)
        meta_pre.append(
            {
                "roll": k + 1,
                "train_month": str(tr_p),
                "val_month": str(va_p),
                "test_month": str(te_p),
                "n_test_rows": int(len(df_test)),
            }
        )

    if not test_frames:
        logger.error("No test data.")
        return
    if len(test_frames) != args.n_rolls:
        logger.warning(
            "Using %d/%d OOS months (stopped early on missing data).",
            len(test_frames),
            args.n_rolls,
        )

    if args.oos_infer == "combo_cache":
        pred_list = predict_oos_mlp_combo_cache(
            model, test_frames, meta_pre, event_cols, device, args.infer_batch, logger
        )
        for pred_df, m in zip(pred_list, meta_pre):
            all_results.append(pred_df)
            meta_rows.append({**m, "infer_mode": "combo_cache"})
    else:
        for k, df_test in enumerate(test_frames):
            m = meta_pre[k]
            te_p = pd.Period(m["test_month"], freq="M")
            logger.info(
                "[frozen infer] roll %d/%d test=%s test_rows=%d",
                k + 1,
                len(test_frames),
                te_p,
                len(df_test),
            )
            pred_df = mlp.predict_factors_mlp(model, df_test, event_cols, device, infer_batch=args.infer_batch)
            pred_df = pred_df[["Date", "Stock", "Factor", "T0_T1_RETURN"]].copy()
            pred_df["roll"] = m["roll"]
            pred_df["train_month"] = m["train_month"]
            pred_df["test_month"] = m["test_month"]
            all_results.append(pred_df)
            meta_rows.append({**m, "infer_mode": "per_month"})

    final_results = pd.concat(all_results, ignore_index=True)
    if args.prepend_results_csv:
        prev = pd.read_csv(args.prepend_results_csv)
        if "Date" in prev.columns and not np.issubdtype(prev["Date"].dtype, np.datetime64):
            prev["Date"] = pd.to_datetime(prev["Date"])
        final_results = pd.concat([prev, final_results], ignore_index=True)

    final_results.to_csv(final_csv, index=False)
    logger.info("Saved final_results: %s", final_csv)

    test_ic = mlp.compute_mean_daily_ic_factor(final_results, factor_col="Factor")
    logger.info("[final] Test mean daily IC (Factor vs T0_T1_RETURN): %.6f", test_ic)
    print(f"[final] Test mean daily IC (Factor vs T0_T1_RETURN): {test_ic:.6f}")

    test_ic_tb = mlp.compute_mean_daily_ic_factor_top_bottom(
        final_results, factor_col="Factor", tail_pct=0.05
    )
    logger.info("[final] Test mean daily IC TOP/BOT 5%%: %.6f", test_ic_tb)
    print(f"[final] Test mean daily IC TOP/BOT 5%: {test_ic_tb:.6f}")

    pivoted = final_results.pivot_table(index="Date", columns="Stock", values="Factor", aggfunc="first")
    pivoted.index = pd.to_datetime(pivoted.index).strftime("%Y%m%d")
    pivoted.to_csv(factor_csv)
    logger.info("Saved factor matrix: %s", factor_csv)

    n_pairs = None
    if args.oos_infer == "combo_cache":
        seen_ct = set()
        for df_test, m in zip(test_frames, meta_pre):
            period = pd.Period(m["test_month"], freq="M")
            ds_ph = _placeholder_date_for_period(period)
            for ck in df_test["combo_key"].astype(str).unique():
                seen_ct.add((ds_ph, ck))
        n_pairs = len(seen_ct)

    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pipeline_variant": "mlp_train1month_frozen_oos12",
                "model": "trivial_mlp",
                "oos_infer": args.oos_infer,
                "infer_batch": args.infer_batch,
                "run_id": run_id,
                "in_dim": in_dim,
                "hidden": args.hidden,
                "dropout": args.dropout,
                "lr": args.lr,
                "max_steps_per_month": args.max_steps_per_month,
                "event_cols_count": len(event_cols),
                "test_ic": test_ic,
                "test_ic_top_bottom_5pct": test_ic_tb,
                "n_unique_month_combo_infer": n_pairs,
                "rolls": meta_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "event_cols": event_cols,
            "in_dim": in_dim,
            "hidden": args.hidden,
            "n_class": mlp.N_CLASS,
            "dropout": args.dropout,
        },
        ckpt_path,
    )
    logger.info("Saved checkpoint: %s", ckpt_path)
    print("done run_id=", run_id)


if __name__ == "__main__":
    main()
