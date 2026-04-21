#!/usr/bin/env python3
# coding: utf-8
"""
Qwen2.5-32B rolling bucket training — **same as** `roll_xs_bucket_factor_qwen32b.py`, but uses
`roll_xs_bucket_factor_pipeline_nodate` (no calendar line in the user prompt).

Does **not** modify `roll_xs_bucket_factor_pipeline_nodate.py` or `roll_xs_bucket_factor_qwen32b.py`.

`run_id` / results folder names include `_nodate_`, e.g. `qwen32_xs_roll12_nodate_YYYYMMDD_HHMMSS`.

Example:
  python3 -u roll_xs_bucket_factor_qwen32b_nodate_roll.py --start_train_month 2023-01 --n_rolls 12 --max_mem_gib 70
"""

from __future__ import annotations

import sys

# Re-use all symbols from the nodate pipeline module (single source of truth for training logic).
import roll_xs_bucket_factor_pipeline_nodate as _nodate_core
from roll_xs_bucket_factor_pipeline_nodate import *  # noqa: F401,F403


def main() -> None:
    global BATCH_SIZE, GRAD_ACCUM

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=list(BACKENDS.keys()), default="mistral")
    ap.add_argument("--start_train_month", type=str, default=START_TRAIN_MONTH)
    ap.add_argument("--n_rolls", type=int, default=N_ROLLS)
    ap.add_argument(
        "--multi_gpu_auto",
        action="store_true",
        help="Use HuggingFace device_map=auto (+ max_memory) across all visible GPUs. "
        "Default on for backend qwen32 when more than one GPU is visible.",
    )
    ap.add_argument(
        "--max_mem_gib",
        type=int,
        default=70,
        help="Per-GPU memory cap passed to max_memory when --multi_gpu_auto (GiB).",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override micro-batch size (default: 4, or 1 for qwen32).",
    )
    ap.add_argument(
        "--grad_accum",
        type=int,
        default=None,
        help="Override gradient accumulation steps (default: 4, or 16 for qwen32).",
    )
    ap.add_argument(
        "--resume_ckpt_dir",
        type=str,
        default=None,
        help="Path to a prior saved checkpoint dir (LoRA adapter + bucket_head.pt [+ tokenizer]) to continue rolling.",
    )
    ap.add_argument(
        "--prepend_results_csv",
        type=str,
        default=None,
        help="If set, read an existing final_results.csv and prepend it before saving (for stitching 6+6 into 12 months).",
    )
    args = ap.parse_args()

    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    if args.grad_accum is not None:
        GRAD_ACCUM = args.grad_accum
    if args.backend == "qwen32":
        if args.batch_size is None:
            BATCH_SIZE = 1
        if args.grad_accum is None:
            GRAD_ACCUM = 16

    # train_one_month / DataLoader read BATCH_SIZE from the pipeline module globals, not this file.
    _nodate_core.BATCH_SIZE = BATCH_SIZE
    _nodate_core.GRAD_ACCUM = GRAD_ACCUM

    bc = BACKENDS[args.backend]
    run_id = args.backend + "_xs_roll" + str(args.n_rolls) + "_nodate_" + datetime.now().strftime("%Y%m%d_%H%M%S")
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

    logger.info("backend=%s run_id=%s (nodate rolling entry)", args.backend, run_id)
    logger.info("start_train_month=%s n_rolls=%d", args.start_train_month, args.n_rolls)
    logger.info("BATCH_SIZE=%d GRAD_ACCUM=%d", BATCH_SIZE, GRAD_ACCUM)

    data = load_data_xs(logger)
    df_all = attach_combo_keys(data.df_events, data.event_cols)
    ret_str = data.ret_clean.copy()
    ret_str.index = ret_str.index.strftime("%Y-%m-%d")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    use_auto = args.multi_gpu_auto or (
        args.backend == "qwen32" and torch.cuda.is_available() and torch.cuda.device_count() > 1
    )
    max_memory: Optional[Dict[Union[int, str], str]] = None
    if use_auto and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device_map = "auto"
        max_memory = {i: f"{args.max_mem_gib}GiB" for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "200GiB"
        logger.info("multi_gpu_auto: device_map=auto max_memory per GPU ~%dGiB", args.max_mem_gib)
    elif torch.cuda.is_available():
        device_map = {"": 0}
    else:
        device_map = "auto"

    tok_src = args.resume_ckpt_dir if args.resume_ckpt_dir else bc.model_id
    tok_kw = dict(cache_dir=bc.cache_dir, local_files_only=True)
    if bc.trust_remote_code:
        tok_kw["trust_remote_code"] = True
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True, **tok_kw)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=False, **tok_kw)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError as e:
            raise ImportError(
                "Backend requires transformers with AutoModelForImageTextToText "
                "(Qwen3.5). Upgrade transformers to a recent release or install from main."
            ) from e
        base_lm = AutoModelForImageTextToText.from_pretrained(bc.model_id, **lm_kw)
        logger.info("Loaded %s via AutoModelForImageTextToText", bc.model_id)
    else:
        base_lm = AutoModelForCausalLM.from_pretrained(bc.model_id, **lm_kw)
    base_lm.config.use_cache = False
    try:
        base_lm.gradient_checkpointing_enable()
    except Exception as e:
        logger.info("gradient_checkpointing_enable: %r", e)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=detect_lora_targets(base_lm),
    )
    if args.resume_ckpt_dir:
        lm = PeftModel.from_pretrained(base_lm, args.resume_ckpt_dir, is_trainable=True)
    else:
        lm = get_peft_model(base_lm, lora_cfg)
    wrapped = LMWithBucketHead(lm, lm.config.hidden_size, N_CLASS)
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
        else:
            logger.warning("resume_ckpt_dir set but bucket_head.pt not found at %s", head_path)

    opt = AdamW(
        [
            {"params": [p for p in wrapped.lm.parameters() if p.requires_grad], "lr": LR_LORA},
            {"params": wrapped.head.parameters(), "lr": LR_HEAD},
        ]
    )

    pad_id = tokenizer.pad_token_id or 0
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

        train_one_month(wrapped, opt, lab_tr, lab_va, tokenizer, args.backend, pad_id, device, logger, k + 1)

        pred_df = predict_factors_for_test(wrapped, df_test, tokenizer, args.backend, pad_id, device)
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
                "run_id": run_id,
                "prompt_module": "roll_xs_bucket_factor_pipeline_nodate",
                "test_ic": test_ic,
                "test_ic_top_bottom_5pct": test_ic_tb,
                "rolls": meta_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    wrapped.lm.save_pretrained(final_ckpt_dir)
    torch.save(wrapped.head.state_dict(), os.path.join(final_ckpt_dir, "bucket_head.pt"))
    tokenizer.save_pretrained(final_ckpt_dir)
    logger.info("Saved final checkpoint: %s", final_ckpt_dir)


def _ensure_argv(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    _ensure_argv("--backend", "qwen32")
    main()
