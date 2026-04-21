#!/usr/bin/env python3
# coding: utf-8
"""
V2 prompt + **train only the first calendar month** (same train/val months as roll 1 of the rolling pipeline),
then **freeze** weights and run **inference only** for N OOS test months (test month = start_train_month + k + 2).

**Default OOS inference** uses the **combo cache** path: one LM forward per unique `combo_key` across all OOS
months, then map factors onto each row (same semantics as `roll_xs_bucket_factor_v2_infer_oos_combo_cache.py`).
Use `--oos_infer per_month` for the slower month-by-month full forward pass.

After training, logs a **validation** breakdown: predicted-class counts, true-bucket counts, confusion matrix.

Does **not** modify roll_xs_bucket_factor_pipeline_v2.py — this is a separate entrypoint.

Typical: Qwen2.5-7B-Instruct
  python3 -u roll_xs_bucket_factor_v2_train1m_frozen_oos12.py --backend qwen --start_train_month 2023-01 --n_rolls 12
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
from torch.utils.data import DataLoader
from tqdm import tqdm

import roll_xs_bucket_factor_pipeline_v2 as v2
from roll_xs_bucket_factor_v2_infer_oos_combo_cache import infer_combo_to_factor


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("roll_xs_factor_v2_train1m_frozen_oos12")
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


@torch.no_grad()
def log_val_prediction_breakdown(
    wrapped: v2.LMWithBucketHead,
    lab_va: pd.DataFrame,
    tokenizer,
    backend: str,
    pad_id: int,
    device: torch.device,
    logger: logging.Logger,
) -> None:
    """After training, log val set: pred class histogram, true bucket histogram, confusion matrix."""
    wrapped.eval()
    ds = v2.ComboBucketTrainDataset(lab_va, tokenizer, v2.MAX_LENGTH, backend)
    loader = DataLoader(
        ds,
        batch_size=v2.BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: v2.collate_batch(b, pad_id),
        num_workers=0,
    )
    preds_all: List[int] = []
    true_all: List[int] = []
    for input_ids, attention_mask, labels in tqdm(loader, desc="val_breakdown"):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        logits = wrapped(input_ids, attention_mask)
        labels = labels.to(logits.device)
        pred = logits.argmax(dim=-1)
        preds_all.extend(pred.cpu().numpy().tolist())
        true_all.extend(labels.cpu().numpy().tolist())
    pred_s = pd.Series(preds_all, name="pred")
    true_s = pd.Series(true_all, name="true")
    pred_counts = pred_s.value_counts().sort_index()
    true_counts = true_s.value_counts().sort_index()
    logger.info("VAL predicted class (argmax) counts: %s", pred_counts.to_dict())
    logger.info("VAL true bucket counts: %s", true_counts.to_dict())
    cm = pd.crosstab(true_s, pred_s, rownames=["true_bucket"], colnames=["pred_class"])
    logger.info("VAL confusion [true_bucket x pred_class]:\n%s", cm)
    print("VAL predicted class counts:", pred_counts.to_dict())
    print("VAL true bucket counts:", true_counts.to_dict())
    print(cm)
    wrapped.train()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=list(v2.BACKENDS.keys()), default="qwen")
    ap.add_argument("--start_train_month", type=str, default=v2.START_TRAIN_MONTH)
    ap.add_argument(
        "--n_rolls",
        type=int,
        default=12,
        help="Number of OOS test months (default 12). Train only uses k=0.",
    )
    ap.add_argument("--multi_gpu_auto", action="store_true")
    ap.add_argument("--max_mem_gib", type=int, default=70)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--resume_ckpt_dir", type=str, default=None)
    ap.add_argument("--prepend_results_csv", type=str, default=None)
    ap.add_argument(
        "--oos_infer",
        choices=("combo_cache", "per_month"),
        default="combo_cache",
        help="OOS inference: combo_cache (default) = one forward per unique combo_key; per_month = legacy.",
    )
    ap.add_argument(
        "--infer_batch_size",
        type=int,
        default=32,
        help="Batch size for combo_cache OOS forwards only (ignored for per_month).",
    )
    args = ap.parse_args()

    if args.batch_size is not None:
        v2.BATCH_SIZE = args.batch_size
    if args.grad_accum is not None:
        v2.GRAD_ACCUM = args.grad_accum
    if args.backend in ("qwen32", "mistral3_24b"):
        if args.batch_size is None:
            v2.BATCH_SIZE = 1
        if args.grad_accum is None:
            v2.GRAD_ACCUM = 16

    bc = v2.BACKENDS[args.backend]
    run_id = (
        args.backend
        + "_v2_train1m_frozen_oos"
        + str(args.n_rolls)
        + "_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir = os.path.join(v2.BASE_DIR, "results", run_id)
    log_dir = os.path.join(v2.BASE_DIR, "logs", run_id)
    final_ckpt_dir = os.path.join(v2.BASE_DIR, "mappings", run_id, "final")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(final_ckpt_dir, exist_ok=True)

    logger = setup_logger(os.path.join(log_dir, "run.log"))
    rng = np.random.default_rng(v2.SEED)
    torch.manual_seed(v2.SEED)

    final_csv = os.path.join(out_dir, "final_results.csv")
    factor_csv = os.path.join(out_dir, "factor_matrix.csv")
    meta_json = os.path.join(out_dir, "roll_meta.json")

    logger.info(
        "pipeline_variant=v2_train1month_frozen_oos12 prompt_module=prompt_xs_bucket_v2 "
        "oos_infer=%s (single train month, then frozen inference)",
        args.oos_infer,
    )
    logger.info("backend=%s run_id=%s", args.backend, run_id)
    logger.info("start_train_month=%s n_rolls(OOS)=%d", args.start_train_month, args.n_rolls)
    logger.info("BATCH_SIZE=%d GRAD_ACCUM=%d", v2.BATCH_SIZE, v2.GRAD_ACCUM)

    data = v2.load_data_xs(logger)
    df_all = v2.attach_combo_keys(data.df_events, data.event_cols)
    ret_str = data.ret_clean.copy()
    ret_str.index = ret_str.index.strftime("%Y-%m-%d")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    use_auto = args.multi_gpu_auto or (
        args.backend in ("qwen32", "mistral3_24b") and torch.cuda.is_available() and torch.cuda.device_count() > 1
    )
    max_memory = None
    if use_auto and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device_map = "auto"
        max_memory = {i: f"{args.max_mem_gib}GiB" for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "200GiB"
        logger.info("multi_gpu_auto: device_map=auto max_memory per GPU ~%dGiB", args.max_mem_gib)
    elif torch.cuda.is_available():
        device_map = {"": 0}
    else:
        device_map = "auto"

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    logger.info(
        "hub_offline: HF_HUB_OFFLINE=%s TRANSFORMERS_OFFLINE=%s",
        os.environ.get("HF_HUB_OFFLINE"),
        os.environ.get("TRANSFORMERS_OFFLINE"),
    )

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
        logger.info("Loaded %s via AutoModelForImageTextToText", bc.model_id)
    elif bc.use_mistral3:
        from transformers import Mistral3ForConditionalGeneration

        base_lm = Mistral3ForConditionalGeneration.from_pretrained(bc.model_id, **lm_kw)
        logger.info("Loaded %s via Mistral3ForConditionalGeneration", bc.model_id)
    else:
        base_lm = v2.AutoModelForCausalLM.from_pretrained(bc.model_id, **lm_kw)
    base_lm.config.use_cache = False
    try:
        base_lm.gradient_checkpointing_enable()
    except Exception as e:
        logger.info("gradient_checkpointing_enable: %r", e)

    lora_cfg = v2.LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=v2.detect_lora_targets(base_lm),
    )
    if args.resume_ckpt_dir:
        lm = v2.PeftModel.from_pretrained(base_lm, args.resume_ckpt_dir, is_trainable=True)
    else:
        lm = v2.get_peft_model(base_lm, lora_cfg)
    wrapped = v2.LMWithBucketHead(lm, v2.config_hidden_size(lm.config), v2.N_CLASS)
    input_device = base_lm.get_input_embeddings().weight.device
    device = input_device
    wrapped.head = wrapped.head.to(device)

    if args.resume_ckpt_dir:
        head_path = os.path.join(args.resume_ckpt_dir, "bucket_head.pt")
        if os.path.exists(head_path):
            wrapped.head.load_state_dict(torch.load(head_path, map_location="cpu"))
            if not use_auto:
                wrapped.head = wrapped.head.to(device)
            logger.info("Loaded bucket head: %s", head_path)

    opt = v2.AdamW(
        [
            {"params": [p for p in wrapped.lm.parameters() if p.requires_grad], "lr": v2.LR_LORA},
            {"params": wrapped.head.parameters(), "lr": v2.LR_HEAD},
        ]
    )

    pad_id = tokenizer.pad_token_id or 0
    start_p = pd.Period(args.start_train_month, freq="M")
    all_results: List[pd.DataFrame] = []
    meta_rows: List[dict] = []

    if args.oos_infer == "combo_cache":
        k0 = 0
        tr_p0 = start_p + k0
        va_p0 = start_p + k0 + 1
        te_p0 = start_p + k0 + 2
        df_train = df_all.loc[v2.month_mask(df_all, tr_p0)].copy()
        df_val = df_all.loc[v2.month_mask(df_all, va_p0)].copy()
        df_test0 = df_all.loc[v2.month_mask(df_all, te_p0)].copy()
        if df_train.empty or df_val.empty or df_test0.empty:
            logger.error(
                "Missing data for roll %d (train=%s val=%s test=%s)", k0 + 1, tr_p0, va_p0, te_p0
            )
        else:
            lab_tr = v2.build_combo_day_labels(df_train, ret_str, v2.MIN_UNIV, v2.MIN_COMBO_STOCKS)
            lab_va = v2.build_combo_day_labels(df_val, ret_str, v2.MIN_UNIV, v2.MIN_COMBO_STOCKS)
            if len(lab_tr) < 100:
                logger.error("Too few train labels at roll %d: %d", k0 + 1, len(lab_tr))
            else:
                if len(lab_tr) > v2.MAX_TRAIN_SAMPLES:
                    lab_tr = lab_tr.sample(
                        n=v2.MAX_TRAIN_SAMPLES, random_state=int(rng.integers(0, 2**31 - 1))
                    ).reset_index(drop=True)
                logger.info(
                    "[train once] train=%s val=%s | first OOS test month=%s | lab_tr=%d lab_va=%d | "
                    "this roll test_rows=%d",
                    tr_p0,
                    va_p0,
                    te_p0,
                    len(lab_tr),
                    len(lab_va),
                    len(df_test0),
                )
                v2.train_one_month(
                    wrapped, opt, lab_tr, lab_va, tokenizer, args.backend, pad_id, device, logger, k0 + 1
                )
                logger.info("VAL class breakdown after single-month training (val month %s):", va_p0)
                log_val_prediction_breakdown(wrapped, lab_va, tokenizer, args.backend, pad_id, device, logger)

                test_frames: List[pd.DataFrame] = []
                meta_pre: List[dict] = []
                for k in range(args.n_rolls):
                    tr_p = start_p + k
                    va_p = start_p + k + 1
                    te_p = start_p + k + 2
                    df_test = df_all.loc[v2.month_mask(df_all, te_p)].copy()
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

                if test_frames:
                    if len(test_frames) != args.n_rolls:
                        logger.warning(
                            "combo_cache: using %d/%d OOS test months (stopped early on missing data).",
                            len(test_frames),
                            args.n_rolls,
                        )
                    all_keys: List[str] = []
                    for df in test_frames:
                        all_keys.extend(df["combo_key"].astype(str).tolist())
                    uniq = sorted(set(all_keys))
                    logger.info(
                        "[frozen OOS combo_cache] union combo_key across %d test months: %d unique",
                        len(test_frames),
                        len(uniq),
                    )
                    _bs = v2.BATCH_SIZE
                    v2.BATCH_SIZE = args.infer_batch_size
                    try:
                        wrapped.eval()
                        combo_to_f = infer_combo_to_factor(
                            wrapped, uniq, tokenizer, args.backend, pad_id, device, logger
                        )
                    finally:
                        v2.BATCH_SIZE = _bs

                    for k, df_test in enumerate(test_frames):
                        m = meta_pre[k]
                        pred_df = df_test.copy()
                        pred_df["Factor"] = [combo_to_f[str(c)] for c in pred_df["combo_key"].astype(str)]
                        pred_df = pred_df[["Date", "Stock", "Factor", "T0_T1_RETURN"]].copy()
                        pred_df["roll"] = k + 1
                        pred_df["train_month"] = m["train_month"]
                        pred_df["test_month"] = m["test_month"]
                        all_results.append(pred_df)
                        meta_rows.append(
                            {
                                "roll": m["roll"],
                                "train_month": m["train_month"],
                                "val_month": m["val_month"],
                                "test_month": m["test_month"],
                                "n_test_rows": m["n_test_rows"],
                                "trained_this_roll": k == 0,
                                "infer_mode": "combo_cache",
                            }
                        )
    else:
        for k in range(args.n_rolls):
            tr_p = start_p + k
            va_p = start_p + k + 1
            te_p = start_p + k + 2

            df_train = df_all.loc[v2.month_mask(df_all, tr_p)].copy()
            df_val = df_all.loc[v2.month_mask(df_all, va_p)].copy()
            df_test = df_all.loc[v2.month_mask(df_all, te_p)].copy()

            if k == 0:
                if df_train.empty or df_val.empty or df_test.empty:
                    logger.error(
                        "Missing data for roll %d (train=%s val=%s test=%s)", k + 1, tr_p, va_p, te_p
                    )
                    break
                lab_tr = v2.build_combo_day_labels(df_train, ret_str, v2.MIN_UNIV, v2.MIN_COMBO_STOCKS)
                lab_va = v2.build_combo_day_labels(df_val, ret_str, v2.MIN_UNIV, v2.MIN_COMBO_STOCKS)
                if len(lab_tr) < 100:
                    logger.error("Too few train labels at roll %d: %d", k + 1, len(lab_tr))
                    break
                if len(lab_tr) > v2.MAX_TRAIN_SAMPLES:
                    lab_tr = lab_tr.sample(
                        n=v2.MAX_TRAIN_SAMPLES, random_state=int(rng.integers(0, 2**31 - 1))
                    ).reset_index(drop=True)
                logger.info(
                    "[train once] train=%s val=%s | first OOS test month=%s | lab_tr=%d lab_va=%d | "
                    "this roll test_rows=%d",
                    tr_p,
                    va_p,
                    te_p,
                    len(lab_tr),
                    len(lab_va),
                    len(df_test),
                )
                v2.train_one_month(
                    wrapped, opt, lab_tr, lab_va, tokenizer, args.backend, pad_id, device, logger, k + 1
                )
                logger.info("VAL class breakdown after single-month training (val month %s):", va_p)
                log_val_prediction_breakdown(wrapped, lab_va, tokenizer, args.backend, pad_id, device, logger)
            else:
                if df_test.empty:
                    logger.error("Missing test data for frozen roll %d (test=%s)", k + 1, te_p)
                    break
                logger.info(
                    "[frozen infer] roll %d/%d test=%s (no train) test_rows=%d",
                    k + 1,
                    args.n_rolls,
                    te_p,
                    len(df_test),
                )

            pred_df = v2.predict_factors_for_test(wrapped, df_test, tokenizer, args.backend, pad_id, device)
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
                    "trained_this_roll": k == 0,
                    "infer_mode": "per_month",
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

    test_ic = v2.compute_mean_daily_ic_factor(final_results, factor_col="Factor")
    logger.info("[final] Test mean daily IC (Factor vs T0_T1_RETURN): %.6f", test_ic)
    print(f"[final] Test mean daily IC (Factor vs T0_T1_RETURN): {test_ic:.6f}")

    test_ic_tb = v2.compute_mean_daily_ic_factor_top_bottom(final_results, factor_col="Factor", tail_pct=0.05)
    logger.info("[final] Test mean daily IC TOP/BOT 5%%: %.6f", test_ic_tb)
    print(f"[final] Test mean daily IC TOP/BOT 5%: {test_ic_tb:.6f}")

    pivoted = final_results.pivot_table(index="Date", columns="Stock", values="Factor", aggfunc="first")
    pivoted.index = pd.to_datetime(pivoted.index).strftime("%Y%m%d")
    pivoted.to_csv(factor_csv)
    logger.info("Saved factor matrix: %s", factor_csv)

    combo_meta: Dict[str, object] = {}
    if args.oos_infer == "combo_cache":
        # n_unique_combos_infer was logged; recompute from results for meta file
        ukeys = set()
        for _k in range(args.n_rolls):
            te_p = start_p + _k + 2
            dft = df_all.loc[v2.month_mask(df_all, te_p)]
            if not dft.empty:
                ukeys.update(dft["combo_key"].astype(str).tolist())
        combo_meta["n_unique_combos_infer"] = len(ukeys)

    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pipeline_variant": "v2_train1month_frozen_oos12",
                "prompt_module": "prompt_xs_bucket_v2",
                "oos_infer": args.oos_infer,
                "infer_batch_size": args.infer_batch_size if args.oos_infer == "combo_cache" else None,
                "backend": args.backend,
                "run_id": run_id,
                "test_ic": test_ic,
                "test_ic_top_bottom_5pct": test_ic_tb,
                "rolls": meta_rows,
                **combo_meta,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    wrapped.lm.save_pretrained(final_ckpt_dir)
    torch.save(wrapped.head.state_dict(), os.path.join(final_ckpt_dir, "bucket_head.pt"))
    tokenizer.save_pretrained(final_ckpt_dir)
    logger.info("Saved final checkpoint: %s", final_ckpt_dir)


if __name__ == "__main__":
    main()
