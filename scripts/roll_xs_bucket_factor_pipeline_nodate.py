#!/usr/bin/env python3
# coding: utf-8
"""
Rolling 1m train / 1m val / 1m test → stitch 6 test months of stock-level factors.

- First train month: 2023-01 (configurable).
- Each roll: train on month M, validate on M+1, predict factors on M+2 (all calendar months).
- Repeat 6 times; concatenate test predictions (Date, Stock, Factor, T0_T1_RETURN).
- Factor = 100 * E_p[bucket] / 4 with E under softmax(logits) → confident class 0 → ~0, class 4 → ~100.
- Saves only final LoRA + head + stitched CSVs + pivoted factor matrix + mean daily Pearson IC.

Usage:
  python3 roll_xs_bucket_factor_pipeline.py --backend mistral
  python3 roll_xs_bucket_factor_pipeline.py --backend qwen
  python3 roll_xs_bucket_factor_pipeline.py --backend llama
  python3 roll_xs_bucket_factor_pipeline.py --backend qwen35_9b_base
  python3 roll_xs_bucket_factor_pipeline.py --backend qwen3_8b

Qwen3.5 checkpoints (including -Base) are registered as AutoModelForImageTextToText; this
script uses text-only batches. Requires a recent transformers build with that class.

Qwen3-8B is AutoModelForCausalLM (Qwen3ForCausalLM); use the same llm_model_cache as other Qwen IDs.

Reproducibility: a frozen copy of this trainer lives in archive/roll_xs_bucket_factor_pipeline_frozen_v1.py.
For new prompts, heads, or losses, add a new module (e.g. roll_xs_bucket_factor_pipeline_v2.py) instead of
rewriting this file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

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
MAX_LENGTH = 1024
MAX_TRAIN_SAMPLES = 16384
BATCH_SIZE = 4
GRAD_ACCUM = 4
TRAIN_EPOCHS = 1
LR_HEAD = 5e-4
LR_LORA = 2e-4
MAX_STEPS_PER_MONTH = 400
SEED = 20260417

N_ROLLS = 6
START_TRAIN_MONTH = "2023-01"


@dataclass
class BackendConfig:
    name: str
    model_id: str
    cache_dir: str
    trust_remote_code: bool = False
    # Qwen3.5 family: HF uses AutoModelForImageTextToText (text-only forward still supported).
    use_image_text_to_text: bool = False


BACKENDS: Dict[str, BackendConfig] = {
    "mistral": BackendConfig(
        name="mistral",
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        cache_dir="/finance_ML/zhanghaohan/rank_based/llm_model_cache/mistral_instruct7B",
    ),
    "qwen": BackendConfig(
        name="qwen",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        cache_dir="/finance_ML/zhanghaohan/rank_based/llm_model_cache",
    ),
    "qwen32": BackendConfig(
        name="qwen32",
        model_id="Qwen/Qwen2.5-32B-Instruct",
        cache_dir="/finance_ML/zhanghaohan/rank_based/llm_model_cache",
    ),
    "qwen3_8b": BackendConfig(
        name="qwen3_8b",
        model_id="Qwen/Qwen3-8B",
        cache_dir="/finance_ML/zhanghaohan/rank_based/llm_model_cache",
        trust_remote_code=True,
    ),
    "llama": BackendConfig(
        name="llama",
        model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        cache_dir="/finance_ML/zhanghaohan/rank_based/llm_model_cache",
        trust_remote_code=True,
    ),
    "qwen35_9b_base": BackendConfig(
        name="qwen35_9b_base",
        model_id="Qwen/Qwen3.5-9B-Base",
        cache_dir="/finance_ML/zhanghaohan/rank_based/llm_model_cache",
        trust_remote_code=True,
        use_image_text_to_text=True,
    ),
}


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("roll_xs_factor")
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
    """
    Mean daily Pearson IC computed only on the union of top/bottom tail_pct by factor, per day.
    This matches tail-focused trading rules (e.g. top 5% / bottom 5%).
    """
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


def build_user_prompt(date_str: str, combo_key: str) -> str:
    events = combo_key.split("|") if combo_key else []
    event_str = "，".join(events)
    return (
    
        f"当日事件组合（多事件同时成立）:\n{event_str}\n\n"
        f"监督目标: 该组合当日覆盖股票的平均隔夜(open-to-open)收益，相对于当日全市场收益率横截面五分位的档位标签 "
        f"y ∈ {{0,1,2,3,4}}（0=最低档，4=最高档）。"
    ).strip()


def wrap_prompt_tokenizer(tokenizer: AutoTokenizer, backend: str, user: str) -> str:
    if backend == "mistral":
        return f"[INST] {user} [/INST]"
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        try:
            messages = [{"role": "user", "content": user}]
            kwargs: Dict[str, object] = dict(tokenize=False, add_generation_prompt=True)
            if backend in ("qwen35_9b_base", "qwen3_8b"):
                kwargs["chat_template_kwargs"] = {"enable_thinking": False}
            try:
                return tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                kwargs.pop("chat_template_kwargs", None)
                return tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            return user
    return user


def detect_lora_targets(model: torch.nn.Module) -> List[str]:
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj"]
    name_set = set()
    for n, _ in model.named_modules():
        last = n.split(".")[-1]
        if last in candidates:
            name_set.add(last)
    return sorted(list(name_set)) if name_set else candidates


def collate_batch(batch, pad_id: int):
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    max_len = max(b[0].numel() for b in batch)
    input_ids = []
    attn = []
    for ids, mask, _ in batch:
        pad_len = max_len - ids.numel()
        if pad_len > 0:
            ids = torch.cat([torch.full((pad_len,), pad_id, dtype=ids.dtype), ids])
            mask = torch.cat([torch.zeros(pad_len, dtype=mask.dtype), mask])
        input_ids.append(ids)
        attn.append(mask)
    return torch.stack(input_ids, dim=0), torch.stack(attn, dim=0), labels


class ComboBucketTrainDataset(Dataset):
    def __init__(self, lab: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int, backend: str):
        self.rows = lab.reset_index(drop=True)
        self.tok = tokenizer
        self.max_length = max_length
        self.backend = backend

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        r = self.rows.iloc[i]
        user = build_user_prompt(str(r["DATE_STR"]), str(r["combo_key"]))
        text = wrap_prompt_tokenizer(self.tok, self.backend, user)
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), int(r["bucket"])


class ComboInferDataset(Dataset):
    """Rows with DATE_STR, combo_key only (deduped)."""

    def __init__(self, rows: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int, backend: str):
        self.rows = rows.reset_index(drop=True)
        self.tok = tokenizer
        self.max_length = max_length
        self.backend = backend

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.rows.iloc[i]
        user = build_user_prompt(str(r["DATE_STR"]), str(r["combo_key"]))
        text = wrap_prompt_tokenizer(self.tok, self.backend, user)
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)


def collate_infer(batch, pad_id: int):
    max_len = max(b[0].numel() for b in batch)
    input_ids = []
    attn = []
    for ids, mask in batch:
        pad_len = max_len - ids.numel()
        if pad_len > 0:
            ids = torch.cat([torch.full((pad_len,), pad_id, dtype=ids.dtype), ids])
            mask = torch.cat([torch.zeros(pad_len, dtype=mask.dtype), mask])
        input_ids.append(ids)
        attn.append(mask)
    return torch.stack(input_ids, dim=0), torch.stack(attn, dim=0)


class LMWithBucketHead(torch.nn.Module):
    def __init__(self, lm: torch.nn.Module, hidden: int, n_class: int):
        super().__init__()
        self.lm = lm
        self.head = torch.nn.Linear(hidden, n_class)
        torch.nn.init.normal_(self.head.weight, std=0.02)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        try:
            out = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        except TypeError:
            out = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                pixel_values=None,
            )
        h = out.hidden_states[-1]
        # Multi-GPU sharding: last hidden may live on a different device than embeddings.
        if self.head.weight.device != h.device:
            self.head = self.head.to(h.device)
        idx = attention_mask.long().sum(dim=1) - 1
        idx = idx.clamp(min=0)
        b_idx = torch.arange(h.size(0), device=h.device, dtype=torch.long)
        pooled = h[b_idx, idx]
        return self.head(pooled.float())


def logits_to_factor(logits: torch.Tensor) -> torch.Tensor:
    """Softmax expected bucket in [0,4] scaled to [0,100]. Confident 0→0, confident 4→100."""
    p = F.softmax(logits.float(), dim=-1)
    k = torch.arange(N_CLASS, device=logits.device, dtype=torch.float32)
    ev = (p * k).sum(dim=-1)
    return ev * (100.0 / float(N_CLASS - 1))


@torch.no_grad()
def accuracy_and_spread(
    wrapped: LMWithBucketHead, loader: DataLoader, lab: pd.DataFrame, device: torch.device
) -> Tuple[float, float]:
    wrapped.eval()
    correct = 0
    total = 0
    preds: List[int] = []
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        logits = wrapped(input_ids, attention_mask)
        labels = labels.to(logits.device)
        pred = logits.argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
        preds.extend(pred.cpu().numpy().tolist())
    acc = float(correct / max(1, total))
    spr = pred_class_spread(lab, np.array(preds, dtype=np.int64))
    wrapped.train()
    return acc, spr


def month_mask(df: pd.DataFrame, period: pd.Period) -> pd.Series:
    return df["Date"].dt.to_period("M") == period


def train_one_month(
    wrapped: LMWithBucketHead,
    opt: AdamW,
    lab_tr: pd.DataFrame,
    lab_va: pd.DataFrame,
    tokenizer: AutoTokenizer,
    backend: str,
    pad_id: int,
    device: torch.device,
    logger: logging.Logger,
    roll_id: int,
) -> Tuple[float, float]:
    ds_tr = ComboBucketTrainDataset(lab_tr, tokenizer, MAX_LENGTH, backend)
    train_loader = DataLoader(
        ds_tr,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_id),
        num_workers=0,
    )
    val_loader = None
    if len(lab_va) > 0:
        ds_va = ComboBucketTrainDataset(lab_va, tokenizer, MAX_LENGTH, backend)
        val_loader = DataLoader(
            ds_va,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=lambda b: collate_batch(b, pad_id),
            num_workers=0,
        )

    wrapped.train()
    step = 0
    for ep in range(TRAIN_EPOCHS):
        opt.zero_grad(set_to_none=True)
        micro = 0
        loss_acc = 0.0
        pbar = tqdm(train_loader, desc=f"roll{roll_id} ep{ep + 1}")
        for input_ids, attention_mask, labels in pbar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = wrapped(input_ids, attention_mask)
            labels = labels.to(logits.device)
            loss = F.cross_entropy(logits, labels) / GRAD_ACCUM
            loss.backward()
            loss_acc += float(loss.item()) * GRAD_ACCUM
            micro += 1
            if micro % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                pbar.set_postfix(loss=f"{loss_acc / GRAD_ACCUM:.4f}")
                loss_acc = 0.0
            if step >= MAX_STEPS_PER_MONTH:
                break
        if micro % GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

    val_acc, val_spr = (0.0, float("nan"))
    if val_loader is not None and len(lab_va) > 0:
        val_acc, val_spr = accuracy_and_spread(wrapped, val_loader, lab_va, device)
    logger.info("[roll %d] VAL acc=%.4f pred_spread(4-0)=%s", roll_id, val_acc, val_spr)
    return val_acc, val_spr


@torch.no_grad()
def predict_factors_for_test(
    wrapped: LMWithBucketHead,
    df_test: pd.DataFrame,
    tokenizer: AutoTokenizer,
    backend: str,
    pad_id: int,
    device: torch.device,
) -> pd.DataFrame:
    """Add Factor column via (DATE_STR, combo_key) deduped forwards."""
    df = df_test.copy()
    df["DATE_STR"] = df["Date"].dt.strftime("%Y-%m-%d")
    uniq = df[["DATE_STR", "combo_key"]].drop_duplicates()
    ds = ComboInferDataset(uniq, tokenizer, MAX_LENGTH, backend)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_infer(b, pad_id),
        num_workers=0,
    )
    wrapped.eval()
    factors: List[float] = []
    for input_ids, attention_mask in tqdm(loader, desc="infer"):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        logits = wrapped(input_ids, attention_mask)
        factors.extend(logits_to_factor(logits).detach().float().cpu().numpy().tolist())
    if len(factors) != len(uniq):
        raise RuntimeError(f"infer len mismatch: factors={len(factors)} uniq={len(uniq)}")
    key_to_fac: Dict[Tuple[str, str], float] = {}
    for i in range(len(uniq)):
        key_to_fac[(str(uniq.iloc[i]["DATE_STR"]), str(uniq.iloc[i]["combo_key"]))] = float(factors[i])
    df["Factor"] = [key_to_fac[(str(r["DATE_STR"]), str(r["combo_key"]))] for _, r in df.iterrows()]
    wrapped.train()
    return df


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

    bc = BACKENDS[args.backend]
    run_id = args.backend + "_xs_roll" + str(args.n_rolls) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
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

    logger.info("backend=%s run_id=%s", args.backend, run_id)
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
        # Load adapter weights into base model for continued training.
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
        # Ensure Date is consistent for grouping/pivot later.
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
            {"test_ic": test_ic, "test_ic_top_bottom_5pct": test_ic_tb, "rolls": meta_rows},
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
