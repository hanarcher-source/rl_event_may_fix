#!/usr/bin/env python3
# coding: utf-8
"""
Trivial MLP baseline: same rolling 1m train / 1m val / 1m test as the LLM pipeline.

- Supervision: same combo-day quintile labels (build_combo_day_labels).
- Features: multi-hot over event_cols for combo_key + 2-dim month cycle from DATE_STR
  (matches info roughly present in the text prompt).
- Head: 5 logits → cross-entropy; Factor = softmax expected bucket scaled to [0,100]
  (identical to roll_xs_bucket_factor_pipeline.logits_to_factor).
- Inference: dedupe (DATE_STR, combo_key), map Factor back to each stock row.

Outputs match LLM runs: final_results.csv, factor_matrix.csv, roll_meta.json, final/mlp.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from combo_xs_bucket_sft_lib import (
    attach_combo_keys,
    build_combo_day_labels,
    load_data_xs,
    pred_class_spread,
)

BASE_DIR = "/finance_ML/zhanghaohan/rl_remodeled/may_fix"

N_CLASS = 5
MIN_UNIV = 200
MIN_COMBO_STOCKS = 1
MAX_TRAIN_SAMPLES = 16384
BATCH_SIZE = 512
MAX_STEPS_PER_MONTH = 400
LR = 3e-3
WEIGHT_DECAY = 1e-4
HIDDEN = 256
DROPOUT = 0.1
SEED = 20260417

N_ROLLS = 12
START_TRAIN_MONTH = "2023-01"


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("roll_xs_mlp")
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


def compute_mean_daily_ic_factor(df_pred: pd.DataFrame, factor_col: str = "Factor") -> float:
    daily_ics: List[float] = []
    for _, g in df_pred.groupby("Date"):
        if g[factor_col].nunique() > 1 and g["T0_T1_RETURN"].nunique() > 1:
            ic, _ = pearsonr(g[factor_col], g["T0_T1_RETURN"])
            if np.isfinite(ic):
                daily_ics.append(float(ic))
    return float(np.mean(daily_ics)) if daily_ics else 0.0


def compute_mean_daily_ic_factor_top_bottom(
    df_pred: pd.DataFrame,
    factor_col: str = "Factor",
    tail_pct: float = 0.05,
) -> float:
    if tail_pct <= 0 or tail_pct >= 0.5:
        raise ValueError("tail_pct must be in (0, 0.5).")
    daily_ics: List[float] = []
    for _, g0 in df_pred.groupby("Date"):
        g = g0[[factor_col, "T0_T1_RETURN"]].dropna()
        n = int(len(g))
        if n < 20:
            continue
        k = max(1, int(np.floor(n * tail_pct)))
        g = g.sort_values(by=factor_col, ascending=True, kind="mergesort")
        tails = pd.concat([g.head(k), g.tail(k)], axis=0, ignore_index=True)
        if tails[factor_col].nunique() > 1 and tails["T0_T1_RETURN"].nunique() > 1:
            ic, _ = pearsonr(tails[factor_col], tails["T0_T1_RETURN"])
            if np.isfinite(ic):
                daily_ics.append(float(ic))
    return float(np.mean(daily_ics)) if daily_ics else 0.0


def event_idx_map(event_cols: List[str]) -> Dict[str, int]:
    return {c: i for i, c in enumerate(event_cols)}


def row_to_feat(date_str: str, combo_key: str, eidx: Dict[str, int], dim: int) -> np.ndarray:
    x = np.zeros(dim, dtype=np.float32)
    if combo_key:
        for e in str(combo_key).split("|"):
            j = eidx.get(e)
            if j is not None:
                x[j] = 1.0
    ts = pd.Timestamp(date_str)
    m = int(ts.month)
    x[dim - 2] = np.sin(2.0 * np.pi * (m - 1) / 12.0)
    x[dim - 1] = np.cos(2.0 * np.pi * (m - 1) / 12.0)
    return x


def lab_to_X(lab: pd.DataFrame, event_cols: List[str]) -> np.ndarray:
    eidx = event_idx_map(event_cols)
    d = len(event_cols) + 2
    n = len(lab)
    X = np.zeros((n, d), dtype=np.float32)
    ds = lab["DATE_STR"].astype(str).values
    ks = lab["combo_key"].astype(str).values
    for i in range(n):
        X[i] = row_to_feat(ds[i], ks[i], eidx, d)
    return X


class LabTensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]


class TrivialMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_class: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


def logits_to_factor(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits.float(), dim=-1)
    k = torch.arange(N_CLASS, device=logits.device, dtype=torch.float32)
    ev = (p * k).sum(dim=-1)
    return ev * (100.0 / float(N_CLASS - 1))


@torch.no_grad()
def accuracy_and_spread_mlp(
    model: nn.Module, loader: DataLoader, lab: pd.DataFrame, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    preds: List[int] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=-1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
        preds.extend(pred.cpu().numpy().tolist())
    acc = float(correct / max(1, total))
    spr = pred_class_spread(lab, np.array(preds, dtype=np.int64))
    model.train()
    return acc, spr


def month_mask(df: pd.DataFrame, period: pd.Period) -> pd.Series:
    return df["Date"].dt.to_period("M") == period


def train_one_month_mlp(
    model: nn.Module,
    opt: AdamW,
    X_tr: np.ndarray,
    lab_tr: pd.DataFrame,
    X_va: np.ndarray,
    lab_va: pd.DataFrame,
    device: torch.device,
    logger: logging.Logger,
    roll_id: int,
    max_steps: int,
) -> Tuple[float, float]:
    y_tr = lab_tr["bucket"].to_numpy(dtype=np.int64)
    ds_tr = LabTensorDataset(X_tr, y_tr)
    train_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)

    val_loader = None
    if len(lab_va) > 0:
        y_va = lab_va["bucket"].to_numpy(dtype=np.int64)
        ds_va = LabTensorDataset(X_va, y_va)
        val_loader = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model.train()
    step = 0
    it = iter(train_loader)
    pbar = tqdm(range(max_steps), desc=f"mlp roll{roll_id} steps")
    for _ in pbar:
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(train_loader)
            xb, yb = next(it)
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        pbar.set_postfix(loss=f"{float(loss.item()):.4f}")

    val_acc, val_spr = (0.0, float("nan"))
    if val_loader is not None and len(lab_va) > 0:
        val_acc, val_spr = accuracy_and_spread_mlp(model, val_loader, lab_va, device)
    logger.info("[roll %d] VAL acc=%.4f pred_spread(4-0)=%s", roll_id, val_acc, val_spr)
    return val_acc, val_spr


@torch.no_grad()
def predict_factors_mlp(
    model: nn.Module,
    df_test: pd.DataFrame,
    event_cols: List[str],
    device: torch.device,
    infer_batch: int = 4096,
) -> pd.DataFrame:
    df = df_test.copy()
    df["DATE_STR"] = df["Date"].dt.strftime("%Y-%m-%d")
    uniq = df[["DATE_STR", "combo_key"]].drop_duplicates().reset_index(drop=True)
    eidx = event_idx_map(event_cols)
    d = len(event_cols) + 2
    X = np.zeros((len(uniq), d), dtype=np.float32)
    for i in range(len(uniq)):
        X[i] = row_to_feat(str(uniq.iloc[i]["DATE_STR"]), str(uniq.iloc[i]["combo_key"]), eidx, d)

    model.eval()
    factors: List[float] = []
    for s in range(0, len(X), infer_batch):
        xb = torch.from_numpy(X[s : s + infer_batch]).to(device)
        factors.extend(logits_to_factor(model(xb)).cpu().numpy().tolist())
    if len(factors) != len(uniq):
        raise RuntimeError(f"infer len mismatch: factors={len(factors)} uniq={len(uniq)}")
    key_to_fac: Dict[Tuple[str, str], float] = {}
    for i in range(len(uniq)):
        key_to_fac[(str(uniq.iloc[i]["DATE_STR"]), str(uniq.iloc[i]["combo_key"]))] = float(factors[i])
    df["Factor"] = [key_to_fac[(str(r["DATE_STR"]), str(r["combo_key"]))] for _, r in df.iterrows()]
    model.train()
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_train_month", type=str, default=START_TRAIN_MONTH)
    ap.add_argument("--n_rolls", type=int, default=N_ROLLS)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--dropout", type=float, default=DROPOUT)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--max_steps_per_month", type=int, default=MAX_STEPS_PER_MONTH)
    ap.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Path to prior mlp.pt from this script to continue rolling.",
    )
    ap.add_argument(
        "--prepend_results_csv",
        type=str,
        default=None,
        help="Prepend an existing final_results.csv (e.g. stitch 6+6 months).",
    )
    args = ap.parse_args()

    run_id = "mlp_xs_roll" + str(args.n_rolls) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(BASE_DIR, "results", run_id)
    log_dir = os.path.join(BASE_DIR, "logs", run_id)
    final_ckpt_dir = os.path.join(BASE_DIR, "mappings", run_id, "final")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(final_ckpt_dir, exist_ok=True)

    logger = setup_logger(os.path.join(log_dir, "run.log"))
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    final_csv = os.path.join(out_dir, "final_results.csv")
    factor_csv = os.path.join(out_dir, "factor_matrix.csv")
    meta_json = os.path.join(out_dir, "roll_meta.json")
    ckpt_path = os.path.join(final_ckpt_dir, "mlp.pt")

    logger.info("run_id=%s start_train_month=%s n_rolls=%d", run_id, args.start_train_month, args.n_rolls)

    data = load_data_xs(logger)
    event_cols = list(data.event_cols)
    df_all = attach_combo_keys(data.df_events, event_cols)
    ret_str = data.ret_clean.copy()
    ret_str.index = ret_str.index.strftime("%Y-%m-%d")

    in_dim = len(event_cols) + 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s in_dim=%d (events+2 month)", device, in_dim)

    model = TrivialMLP(in_dim, args.hidden, N_CLASS, args.dropout).to(device)
    if args.resume_ckpt:
        blob = torch.load(args.resume_ckpt, map_location="cpu")
        model.load_state_dict(blob["state_dict"])
        logger.info("Loaded MLP weights from %s", args.resume_ckpt)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    start_p = pd.Period(args.start_train_month, freq="M")
    all_results: List[pd.DataFrame] = []
    meta_rows: List[dict] = []

    for k in range(args.n_rolls):
        tr_p = start_p + k
        va_p = start_p + k + 1
        te_p = start_p + k + 2

        df_train = df_all.loc[month_mask(df_all, tr_p)].copy()
        df_val = df_all.loc[month_mask(df_all, va_p)].copy()
        df_test = df_all.loc[month_mask(df_all, te_p)].copy()

        if df_train.empty or df_val.empty or df_test.empty:
            logger.error("Missing data for roll %d (train=%s val=%s test=%s)", k + 1, tr_p, va_p, te_p)
            break

        lab_tr = build_combo_day_labels(df_train, ret_str, MIN_UNIV, MIN_COMBO_STOCKS)
        lab_va = build_combo_day_labels(df_val, ret_str, MIN_UNIV, MIN_COMBO_STOCKS)
        if len(lab_tr) < 100:
            logger.error("Too few train labels at roll %d: %d", k + 1, len(lab_tr))
            break
        if len(lab_tr) > MAX_TRAIN_SAMPLES:
            lab_tr = lab_tr.sample(
                n=MAX_TRAIN_SAMPLES, random_state=int(rng.integers(0, 2**31 - 1))
            ).reset_index(drop=True)

        X_tr = lab_to_X(lab_tr, event_cols)
        X_va = lab_to_X(lab_va, event_cols)

        logger.info(
            "[roll %d/%d] train=%s val=%s test=%s | lab_tr=%d lab_va=%d test_rows=%d",
            k + 1,
            args.n_rolls,
            tr_p,
            va_p,
            te_p,
            len(lab_tr),
            len(lab_va),
            len(df_test),
        )

        train_one_month_mlp(
            model,
            opt,
            X_tr,
            lab_tr,
            X_va,
            lab_va,
            device,
            logger,
            k + 1,
            args.max_steps_per_month,
        )

        pred_df = predict_factors_mlp(model, df_test, event_cols, device)
        pred_df = pred_df[["Date", "Stock", "Factor", "T0_T1_RETURN"]].copy()
        pred_df["roll"] = k + 1
        pred_df["train_month"] = str(tr_p)
        pred_df["test_month"] = str(te_p)
        all_results.append(pred_df)
        meta_rows.append(
            {
                "roll": k + 1,
                "train_month": str(tr_p),
                "val_month": str(va_p),
                "test_month": str(te_p),
                "n_test_rows": int(len(pred_df)),
            }
        )

    if not all_results:
        logger.error("No results produced.")
        return

    final_results = pd.concat(all_results, ignore_index=True)
    if args.prepend_results_csv:
        prev = pd.read_csv(args.prepend_results_csv)
        if "Date" in prev.columns and not np.issubdtype(prev["Date"].dtype, np.datetime64):
            prev["Date"] = pd.to_datetime(prev["Date"])
        final_results = pd.concat([prev, final_results], ignore_index=True)

    final_results.to_csv(final_csv, index=False)
    logger.info("Saved final_results: %s", final_csv)

    test_ic = compute_mean_daily_ic_factor(final_results, factor_col="Factor")
    logger.info("[final] Test mean daily IC (Factor vs T0_T1_RETURN): %.6f", test_ic)
    print(f"[final] Test mean daily IC (Factor vs T0_T1_RETURN): {test_ic:.6f}")

    test_ic_tb = compute_mean_daily_ic_factor_top_bottom(final_results, factor_col="Factor", tail_pct=0.05)
    logger.info("[final] Test mean daily IC TOP/BOT 5%%: %.6f", test_ic_tb)
    print(f"[final] Test mean daily IC TOP/BOT 5%: {test_ic_tb:.6f}")

    pivoted = final_results.pivot_table(index="Date", columns="Stock", values="Factor", aggfunc="first")
    pivoted.index = pd.to_datetime(pivoted.index).strftime("%Y%m%d")
    pivoted.to_csv(factor_csv)
    logger.info("Saved factor matrix: %s", factor_csv)
    print("Saved final results:", final_csv)
    print("Saved factor matrix:", factor_csv)

    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "trivial_mlp",
                "in_dim": in_dim,
                "hidden": args.hidden,
                "dropout": args.dropout,
                "lr": args.lr,
                "max_steps_per_month": args.max_steps_per_month,
                "event_cols_count": len(event_cols),
                "test_ic": test_ic,
                "test_ic_top_bottom_5pct": test_ic_tb,
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
            "n_class": N_CLASS,
            "dropout": args.dropout,
        },
        ckpt_path,
    )
    logger.info("Saved checkpoint: %s", ckpt_path)


if __name__ == "__main__":
    main()
