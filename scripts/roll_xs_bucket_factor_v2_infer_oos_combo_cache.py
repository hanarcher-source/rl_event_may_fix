#!/usr/bin/env python3
# coding: utf-8
"""
Fast OOS inference for **v2 prompt** (combo-only text): load a saved LoRA + bucket head, then

1) Union all `combo_key` seen across the `n_rolls` test months (same calendar as rolling pipeline).
2) Run the LM **once per unique combo** (prompt does not depend on date).
3) Map `combo_key -> Factor` onto every row of each month’s `df_test` and write the same CSVs as training scripts.

Use when the original job spent too long on repeated infer forwards.

Requires a completed checkpoint dir (e.g. .../mappings/<run_id>/final) with adapter + bucket_head.pt + tokenizer.

Example:
  python3 -u roll_xs_bucket_factor_v2_infer_oos_combo_cache.py \\
    --resume_ckpt_dir /finance_ML/zhanghaohan/rl_remodeled/may_fix/mappings/qwen_v2_train1m_frozen_oos12_20260418_121948/final \\
    --backend qwen --start_train_month 2023-01 --n_rolls 12 --infer_batch_size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

import roll_xs_bucket_factor_pipeline_v2 as v2


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("roll_xs_factor_v2_infer_combo_cache")
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


def load_wrapped_from_ckpt(
    args: argparse.Namespace,
    bc: v2.BackendConfig,
    logger: logging.Logger,
) -> tuple:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    use_auto = args.multi_gpu_auto or (
        args.backend in ("qwen32", "mistral3_24b") and torch.cuda.is_available() and torch.cuda.device_count() > 1
    )
    max_memory = None
    if use_auto and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device_map = "auto"
        max_memory = {i: f"{args.max_mem_gib}GiB" for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "200GiB"
        logger.info("multi_gpu_auto: device_map=auto")
    elif torch.cuda.is_available():
        device_map = {"": 0}
    else:
        device_map = "auto"

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    tokenizer = v2.load_tokenizer_for_backend(bc, resume_ckpt_dir=args.resume_ckpt_dir, logger=logger)

    lm_kw = dict(
        cache_dir=bc.cache_dir,
        local_files_only=True,
        device_map=device_map,
        torch_dtype=dtype,
    )
    if max_memory is not None:
        lm_kw["max_memory"] = max_memory
    if bc.trust_remote_code:
        lm_kw["trust_remote_code"] = True

    if bc.use_image_text_to_text:
        from transformers import AutoModelForImageTextToText

        base_lm = AutoModelForImageTextToText.from_pretrained(bc.model_id, **lm_kw)
    elif bc.use_mistral3:
        from transformers import Mistral3ForConditionalGeneration

        base_lm = Mistral3ForConditionalGeneration.from_pretrained(bc.model_id, **lm_kw)
    else:
        base_lm = v2.AutoModelForCausalLM.from_pretrained(bc.model_id, **lm_kw)
    base_lm.config.use_cache = False

    lm = v2.PeftModel.from_pretrained(base_lm, args.resume_ckpt_dir, is_trainable=False)
    wrapped = v2.LMWithBucketHead(lm, v2.config_hidden_size(lm.config), v2.N_CLASS)
    input_device = base_lm.get_input_embeddings().weight.device
    device = input_device
    wrapped.head = wrapped.head.to(device)

    head_path = os.path.join(args.resume_ckpt_dir, "bucket_head.pt")
    if not os.path.exists(head_path):
        raise FileNotFoundError(f"Missing {head_path}")
    wrapped.head.load_state_dict(torch.load(head_path, map_location="cpu"))
    if not use_auto:
        wrapped.head = wrapped.head.to(device)
    logger.info("Loaded checkpoint from %s", args.resume_ckpt_dir)

    wrapped.eval()
    for p in wrapped.parameters():
        p.requires_grad_(False)

    pad_id = tokenizer.pad_token_id or 0
    return wrapped, tokenizer, pad_id, device


def infer_combo_to_factor(
    wrapped: v2.LMWithBucketHead,
    uniq_combos: List[str],
    tokenizer,
    backend: str,
    pad_id: int,
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, float]:
    """Single forward pass per unique combo_key (v2 prompt is date-free)."""
    placeholder = "1970-01-01"
    synth = pd.DataFrame(
        {
            "Date": pd.to_datetime([placeholder] * len(uniq_combos)),
            "Stock": ["_"] * len(uniq_combos),
            "combo_key": uniq_combos,
            "T0_T1_RETURN": np.nan,
        }
    )
    logger.info("Infer unique combos: %d (batch_size=%d)", len(uniq_combos), v2.BATCH_SIZE)
    out = v2.predict_factors_for_test(wrapped, synth, tokenizer, backend, pad_id, device)
    return {str(r["combo_key"]): float(r["Factor"]) for _, r in out.iterrows()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume_ckpt_dir", type=str, required=True, help=".../mappings/<run_id>/final")
    ap.add_argument("--backend", choices=list(v2.BACKENDS.keys()), default="qwen")
    ap.add_argument("--start_train_month", type=str, default=v2.START_TRAIN_MONTH)
    ap.add_argument("--n_rolls", type=int, default=12)
    ap.add_argument("--infer_batch_size", type=int, default=32, help="Micro-batch for LM forward")
    ap.add_argument("--multi_gpu_auto", action="store_true")
    ap.add_argument("--max_mem_gib", type=int, default=70)
    ap.add_argument("--prepend_results_csv", type=str, default=None)
    ap.add_argument(
        "--out_run_id",
        type=str,
        default=None,
        help="Optional explicit run_id folder name under results/logs (default: auto timestamp).",
    )
    args = ap.parse_args()

    ckpt = os.path.abspath(args.resume_ckpt_dir)
    if not os.path.isdir(ckpt):
        raise NotADirectoryError(ckpt)

    v2.BATCH_SIZE = args.infer_batch_size

    bc = v2.BACKENDS[args.backend]
    run_id = args.out_run_id or (
        args.backend
        + "_v2_infer_combo_cache_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir = os.path.join(v2.BASE_DIR, "results", run_id)
    log_dir = os.path.join(v2.BASE_DIR, "logs", run_id)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger = setup_logger(os.path.join(log_dir, "run.log"))
    logger.info("pipeline_variant=v2_infer_oos_combo_cache resume_ckpt_dir=%s", ckpt)
    logger.info("backend=%s run_id=%s n_rolls=%d infer_batch_size=%d", args.backend, run_id, args.n_rolls, v2.BATCH_SIZE)

    data = v2.load_data_xs(logger)
    df_all = v2.attach_combo_keys(data.df_events, data.event_cols)
    ret_str = data.ret_clean.copy()
    ret_str.index = ret_str.index.strftime("%Y-%m-%d")

    start_p = pd.Period(args.start_train_month, freq="M")
    test_frames: List[pd.DataFrame] = []
    meta_pre: List[dict] = []

    for k in range(args.n_rolls):
        tr_p = start_p + k
        va_p = start_p + k + 1
        te_p = start_p + k + 2
        df_test = df_all.loc[v2.month_mask(df_all, te_p)].copy()
        if df_test.empty:
            logger.error("Missing test month %s", te_p)
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

    all_keys: List[str] = []
    for df in test_frames:
        all_keys.extend(df["combo_key"].astype(str).tolist())
    uniq = sorted(set(all_keys))
    logger.info("Union combo_key across OOS months: %d unique", len(uniq))

    wrapped, tokenizer, pad_id, device = load_wrapped_from_ckpt(args, bc, logger)
    combo_to_f = infer_combo_to_factor(wrapped, uniq, tokenizer, args.backend, pad_id, device, logger)

    all_results: List[pd.DataFrame] = []
    meta_rows: List[dict] = []

    for k, df_test in enumerate(test_frames):
        m = meta_pre[k]
        pred_df = df_test.copy()
        pred_df["Factor"] = [combo_to_f[str(c)] for c in pred_df["combo_key"].astype(str)]
        pred_df = pred_df[["Date", "Stock", "Factor", "T0_T1_RETURN"]].copy()
        pred_df["roll"] = k + 1
        pred_df["train_month"] = m["train_month"]
        pred_df["test_month"] = m["test_month"]
        all_results.append(pred_df)
        meta_rows.append({**m, "infer_mode": "combo_cache"})

    final_csv = os.path.join(out_dir, "final_results.csv")
    factor_csv = os.path.join(out_dir, "factor_matrix.csv")
    meta_json = os.path.join(out_dir, "roll_meta.json")

    final_results = pd.concat(all_results, ignore_index=True)
    if args.prepend_results_csv:
        prev = pd.read_csv(args.prepend_results_csv)
        if "Date" in prev.columns and not np.issubdtype(prev["Date"].dtype, np.datetime64):
            prev["Date"] = pd.to_datetime(prev["Date"])
        final_results = pd.concat([prev, final_results], ignore_index=True)

    final_results.to_csv(final_csv, index=False)
    logger.info("Saved final_results: %s", final_csv)

    test_ic = v2.compute_mean_daily_ic_factor(final_results, factor_col="Factor")
    logger.info("[final] Test mean daily IC: %.6f", test_ic)
    print(f"[final] Test mean daily IC: {test_ic:.6f}")

    test_ic_tb = v2.compute_mean_daily_ic_factor_top_bottom(final_results, factor_col="Factor", tail_pct=0.05)
    logger.info("[final] Test mean daily IC TOP/BOT 5%%: %.6f", test_ic_tb)
    print(f"[final] Test mean daily IC TOP/BOT 5%: {test_ic_tb:.6f}")

    pivoted = final_results.pivot_table(index="Date", columns="Stock", values="Factor", aggfunc="first")
    pivoted.index = pd.to_datetime(pivoted.index).strftime("%Y%m%d")
    pivoted.to_csv(factor_csv)
    logger.info("Saved factor matrix: %s", factor_csv)

    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pipeline_variant": "v2_infer_oos_combo_cache",
                "prompt_module": "prompt_xs_bucket_v2",
                "backend": args.backend,
                "run_id": run_id,
                "resume_ckpt_dir": ckpt,
                "test_ic": test_ic,
                "test_ic_top_bottom_5pct": test_ic_tb,
                "n_unique_combos_infer": len(uniq),
                "rolls": meta_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Done. meta=%s", meta_json)


if __name__ == "__main__":
    main()
