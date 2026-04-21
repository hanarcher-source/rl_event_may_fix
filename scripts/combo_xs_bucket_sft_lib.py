#!/usr/bin/env python3
# coding: utf-8
"""Shared data + label helpers for cross-sectional combo-day bucket SFT."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class LoadedDataXs:
    df_events: pd.DataFrame
    event_cols: List[str]
    ret_clean: pd.DataFrame


def load_data_xs(logger: logging.Logger) -> LoadedDataXs:
    """Event rows + T0_T1_RETURN + full return matrix (masked universe)."""
    all_k_line_df = pd.read_csv("/finance_ML/zhanghaohan/RL_tune/stored_data/relevent_k_df.csv")
    all_k_line_df["DYID"] = all_k_line_df["sec_id"].str.replace("_", ".")

    unique_dates = all_k_line_df["date"].unique()
    unique_dates_df = pd.DataFrame({"date": unique_dates})
    unique_dates_df["date_datetime"] = pd.to_datetime(unique_dates_df["date"])
    unique_dates_df = unique_dates_df.sort_values(by="date_datetime").reset_index(drop=True)
    unique_dates_df["date_index"] = unique_dates_df.index
    all_k_line_df = all_k_line_df.merge(unique_dates_df, on=["date"])

    open_931 = all_k_line_df[all_k_line_df["time"] == "09:31"][["date", "date_index", "DYID", "adj_vwap"]]
    open_931_T0 = open_931[["date", "date_index", "DYID", "adj_vwap"]].copy()
    open_931_T1 = open_931[["date_index", "DYID", "adj_vwap"]].copy()
    open_931_T0["next_date_index"] = open_931_T0["date_index"] + 1

    joined = open_931_T0.merge(
        open_931_T1.rename(columns={"date_index": "next_date_index", "adj_vwap": "T1_adj_vwap"}),
        on=["next_date_index", "DYID"],
        how="inner",
    )
    joined["T0_T1_RETURN"] = (joined["T1_adj_vwap"] - joined["adj_vwap"]) / joined["adj_vwap"]
    ret_mat = joined.pivot(index="date", columns="DYID", values="T0_T1_RETURN")

    full_index = all_k_line_df["date"].unique()
    full_columns = all_k_line_df["DYID"].unique()
    ret_mat = ret_mat.reindex(index=full_index, columns=full_columns)

    df_st_mask = pd.read_parquet(
        "/finance_ML/zhanghaohan/RL_tune/stored_data/univ_a_2_8773b1b43d51.parquet",
        engine="pyarrow",
    )
    df_st_mask["T0_DAY"] = df_st_mask.index
    df_st_mask.columns = [col.replace("_", ".") if col != "T0_DAY" else col for col in df_st_mask.columns]
    df_st_mask_long = df_st_mask.melt(id_vars="T0_DAY", var_name="stock", value_name="st_mask_value")

    up_down_mask = pd.read_csv("/finance_ML/zhanghaohan/RL_tune/stored_data/limit_up_down_open_mask_5b87e2b5e4a2.csv")
    up_down_mask.rename(columns={"Unnamed: 0": "T0_DAY"}, inplace=True)
    up_down_mask.columns = [col.replace("_", ".") if col != "T0_DAY" else col for col in up_down_mask.columns]
    up_down_mask_long = up_down_mask.melt(id_vars="T0_DAY", var_name="stock", value_name="up_down_mask_value")
    up_down_mask_long["T0_DAY"] = up_down_mask_long["T0_DAY"].astype(str)

    combined = pd.merge(df_st_mask_long, up_down_mask_long, on=["T0_DAY", "stock"], how="inner")
    combined["two_mask"] = ((combined["st_mask_value"] == 1) & (combined["up_down_mask_value"] == 1)).astype(int)
    combined.rename(columns={"stock": "DYID"}, inplace=True)
    combined["date"] = pd.to_datetime(combined["T0_DAY"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    mask_mat = combined.pivot(index="date", columns="DYID", values="two_mask").reindex(index=full_index, columns=full_columns)

    ret_clean = ret_mat.where((ret_mat.notna()) & (mask_mat == 1), other=np.nan)
    ret_clean.index = pd.to_datetime(ret_clean.index)
    ret_clean = ret_clean.sort_index()

    binary_sequences = pd.read_csv(
        "/finance_ML/zhanghaohan/RL_tune/stored_data/binary_sequence_wo_nan_updated_event.csv",
        index_col=0,
    )
    event_columns_start = binary_sequences.columns.get_loc("IPO")
    binary_sequences = binary_sequences.loc[
        ~(binary_sequences.iloc[:, event_columns_start:] == 0).all(axis=1)
    ].copy()

    binary_sequences["Date"] = pd.to_datetime(binary_sequences["Date"], format="%Y%m%d")
    binary_sequences["DATE_STR"] = binary_sequences["Date"].dt.strftime("%Y-%m-%d")

    ret_clean_str_index = ret_clean.copy()
    ret_clean_str_index.index = ret_clean_str_index.index.strftime("%Y-%m-%d")
    long_ret = ret_clean_str_index.reset_index(names="DATE_STR").melt(
        id_vars="DATE_STR", var_name="DYID", value_name="T0_T1_RETURN"
    )
    binary_sequences = binary_sequences.merge(
        long_ret,
        left_on=["DATE_STR", "Stock"],
        right_on=["DATE_STR", "DYID"],
        how="left",
    )
    binary_sequences.drop(columns=["DYID"], inplace=True, errors="ignore")
    binary_sequences.dropna(subset=["T0_T1_RETURN"], inplace=True)
    binary_sequences["Date"] = pd.to_datetime(binary_sequences["DATE_STR"])

    stop_col = "DATE_STR" if "DATE_STR" in binary_sequences.columns else "T0_T1_RETURN"
    event_cols = list(binary_sequences.columns[event_columns_start : binary_sequences.columns.get_loc(stop_col)])

    logger.info("Rows after dropna(T0_T1_RETURN): %d | event_cols=%d", len(binary_sequences), len(event_cols))
    return LoadedDataXs(df_events=binary_sequences, event_cols=event_cols, ret_clean=ret_clean)


def row_to_combo_key(row_events: np.ndarray, event_cols: List[str]) -> str:
    idx = np.flatnonzero(row_events > 0)
    if idx.size == 0:
        return ""
    return "|".join([event_cols[i] for i in idx])


def universe_quintile_bucket(mean_ret: float, univ: np.ndarray) -> Optional[int]:
    x = univ[np.isfinite(univ)]
    if x.size < 5:
        return None
    try:
        _, bin_edges = pd.qcut(x, q=5, retbins=True, duplicates="drop")
    except Exception:
        return None
    if bin_edges is None or len(bin_edges) < 2:
        return None
    n_bins = len(bin_edges) - 1
    if n_bins < 2:
        return None
    try:
        cat = pd.cut(
            pd.Series([float(mean_ret)], dtype="float64"),
            bins=bin_edges,
            labels=False,
            include_lowest=True,
        )
        b = cat.iloc[0]
        if pd.isna(b):
            return None
        bi = int(b)
        if bi < 0 or bi >= n_bins:
            return int(np.clip(bi, 0, n_bins - 1))
        return bi
    except Exception:
        return None


def attach_combo_keys(df: pd.DataFrame, event_cols: List[str]) -> pd.DataFrame:
    X = df[event_cols].fillna(0).astype(np.int8).values
    keys = [row_to_combo_key(X[i], event_cols) for i in tqdm(range(X.shape[0]), desc="combo keys")]
    out = df.copy()
    out["combo_key"] = keys
    return out.loc[out["combo_key"] != ""].copy()


def build_combo_day_labels(
    df: pd.DataFrame,
    ret_str: pd.DataFrame,
    min_univ: int,
    min_combo_stocks: int,
) -> pd.DataFrame:
    """
    df: columns include Date, combo_key, T0_T1_RETURN
    ret_str: index YYYY-MM-DD, columns stocks, values T0_T1 universe returns
    """
    records: List[dict] = []
    for d, g_day in df.groupby("Date", sort=True):
        d_str = pd.Timestamp(d).strftime("%Y-%m-%d")
        if d_str not in ret_str.index:
            continue
        univ = ret_str.loc[d_str].to_numpy(dtype=np.float64)
        if int(np.isfinite(univ).sum()) < min_univ:
            continue
        agg = g_day.groupby("combo_key", sort=False)["T0_T1_RETURN"].agg(["mean", "count"]).reset_index()
        agg.rename(columns={"mean": "combo_day_mean_ret", "count": "n_stocks"}, inplace=True)
        if min_combo_stocks > 1:
            agg = agg[agg["n_stocks"] >= min_combo_stocks].copy()
        for _, row in agg.iterrows():
            b = universe_quintile_bucket(float(row["combo_day_mean_ret"]), univ)
            if b is None:
                continue
            records.append(
                {
                    "DATE_STR": d_str,
                    "combo_key": str(row["combo_key"]),
                    "bucket": int(b),
                    "n_stocks": int(row["n_stocks"]),
                    "combo_day_mean_ret": float(row["combo_day_mean_ret"]),
                }
            )
    return pd.DataFrame.from_records(records)


def pred_class_spread(lab: pd.DataFrame, pred_class: np.ndarray) -> float:
    """E[ret | pred=4] - E[ret | pred=0] when both classes exist."""
    if lab.empty or len(pred_class) != len(lab):
        return float("nan")
    tmp = lab[["combo_day_mean_ret"]].copy()
    tmp["pred"] = pred_class.astype(np.int64)
    m = tmp.groupby("pred", sort=True)["combo_day_mean_ret"].mean()
    if 0 not in m.index or 4 not in m.index:
        return float("nan")
    return float(m.loc[4] - m.loc[0])
