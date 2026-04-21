#!/usr/bin/env python3
# coding: utf-8
"""
Rolling bucket factor pipeline **v2**: same training/inference code as v1, but user prompt from
`prompt_xs_bucket_v2.build_user_prompt_v2` (no calendar date; analyst template).

Run IDs and logs are suffixed so outputs are not confused with v1, e.g. mistral_v2_xs_roll12_<timestamp>.

Usage:
  python3 roll_xs_bucket_factor_pipeline_v2.py --backend mistral
  python3 roll_xs_bucket_factor_pipeline_v2.py --backend qwen
  python3 roll_xs_bucket_factor_pipeline_v2.py --backend llama
  python3 roll_xs_bucket_factor_pipeline_v2.py --backend qwen3_8b
  python3 roll_xs_bucket_factor_pipeline_v2.py --backend mistral3_24b

v1 frozen trainer: roll_xs_bucket_factor_pipeline.py + archive/roll_xs_bucket_factor_pipeline_frozen_v1.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
from prompt_xs_bucket_v2 import build_user_prompt_v2

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
    # Mistral Small 3.x: HF uses Mistral3ForConditionalGeneration; tokenization via `mistral_common`.
    use_mistral3: bool = False


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
    "mistral3_24b": BackendConfig(
        name="mistral3_24b",
        model_id="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        cache_dir="/finance_ML/zhanghaohan/rank_based/llm_model_cache",
        trust_remote_code=True,
        use_mistral3=True,
    ),
}


class MistralCommonTokenizerFacade:
    """Minimal HF-like tokenizer interface backed by `mistral_common.MistralTokenizer`."""

    def __init__(self, repo_id: str, max_length: int):
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        self.repo_id = repo_id
        self.max_length = int(max_length)
        self._tok = MistralTokenizer.from_hf_hub(repo_id, local_files_only=True)
        # Mistral uses a byte-level style vocab; pick a safe pad id if needed downstream.
        self.pad_token = None
        self.pad_token_id = 0
        self.padding_side = "left"
        self.chat_template = None

    def __call__(
        self,
        text: str,
        truncation: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        from mistral_common.protocol.instruct.messages import UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest

        _ = padding  # collate handles padding; single-example encoding is un-padded
        if return_tensors != "pt":
            raise ValueError("Only return_tensors='pt' is supported")

        ml = int(max_length or self.max_length)
        req = ChatCompletionRequest(messages=[UserMessage(content=text)])
        out = self._tok.encode_chat_completion(req, max_model_input_len=ml if truncation else None)
        ids = list(out.tokens)
        if truncation and len(ids) > ml:
            ids = ids[:ml]
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def save_pretrained(self, save_directory: str) -> None:
        """Save enough artifacts for offline `MistralTokenizer.from_hf_hub(repo_id, local_files_only=True)`.

        We copy Tekken + params files from the locally cached HF snapshot into the checkpoint directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        snap_dir = self._snapshot_dir()
        for fname in ("tekken.json", "params.json", "SYSTEM_PROMPT.txt"):
            src = os.path.join(snap_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(save_directory, fname))
        meta = {"repo_id": self.repo_id, "tokenizer": "mistral_common.MistralTokenizer", "source_snapshot": snap_dir}
        with open(os.path.join(save_directory, "mistral_tokenizer_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _snapshot_dir(self) -> str:
        # Mirrors `mistral_common.tokens.tokenizers.utils.list_local_hf_repo_files` path logic.
        from huggingface_hub.constants import DEFAULT_REVISION, HF_HUB_CACHE, REPO_ID_SEPARATOR

        repo_cache = os.path.join(HF_HUB_CACHE, REPO_ID_SEPARATOR.join(["models", *self.repo_id.split("/")]))
        rev_file = os.path.join(repo_cache, "refs", DEFAULT_REVISION)
        revision = None
        if os.path.isfile(rev_file):
            with open(rev_file, "r", encoding="utf-8") as fh:
                revision = fh.read().strip()
        if not revision:
            raise FileNotFoundError(f"Cannot resolve HF revision for {self.repo_id} under {repo_cache}")
        snap = os.path.join(repo_cache, "snapshots", revision)
        if not os.path.isdir(snap):
            raise FileNotFoundError(f"Missing snapshot dir: {snap}")
        return snap


def load_tokenizer_for_backend(
    bc: BackendConfig,
    *,
    resume_ckpt_dir: Optional[str],
    logger: logging.Logger,
) -> Any:
    """Tokenizer loader shared by v2 training/inference entrypoints."""
    if bc.use_mistral3:
        os.environ.setdefault("HF_HUB_CACHE", bc.cache_dir)
        if resume_ckpt_dir and os.path.isdir(resume_ckpt_dir):
            tekken = os.path.join(resume_ckpt_dir, "tekken.json")
            if os.path.isfile(tekken):
                from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

                meta_path = os.path.join(resume_ckpt_dir, "mistral_tokenizer_meta.json")
                repo_id = bc.model_id
                if os.path.isfile(meta_path):
                    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
                    repo_id = str(meta.get("repo_id") or bc.model_id)
                tok = MistralTokenizer.from_file(tekken)
                logger.info("tokenizer=mistral_common.MistralTokenizer.from_file ckpt_dir=%s", resume_ckpt_dir)
                return _mistral_tokenizer_to_facade(tok, repo_id=repo_id, max_length=MAX_LENGTH)

        tokenizer = MistralCommonTokenizerFacade(bc.model_id, MAX_LENGTH)
        logger.info(
            "tokenizer=mistral_common.MistralTokenizer repo_id=%s HF_HUB_CACHE=%s",
            bc.model_id,
            os.environ.get("HF_HUB_CACHE"),
        )
        return tokenizer

    tok_src = resume_ckpt_dir if resume_ckpt_dir else bc.model_id
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
    return tokenizer


def _mistral_tokenizer_to_facade(tok: Any, *, repo_id: str, max_length: int) -> Any:
    """Wrap an instantiated `MistralTokenizer` with the same __call__ interface as `MistralCommonTokenizerFacade`."""

    class _Wrapped:
        def __init__(self, inner: Any):
            self._tok = inner
            self.repo_id = repo_id
            self.max_length = int(max_length)
            self.pad_token = None
            self.pad_token_id = 0
            self.padding_side = "left"
            self.chat_template = None

        def __call__(
            self,
            text: str,
            truncation: bool = True,
            max_length: Optional[int] = None,
            padding: bool = False,
            return_tensors: str = "pt",
        ) -> Dict[str, torch.Tensor]:
            from mistral_common.protocol.instruct.messages import UserMessage
            from mistral_common.protocol.instruct.request import ChatCompletionRequest

            _ = padding
            if return_tensors != "pt":
                raise ValueError("Only return_tensors='pt' is supported")
            ml = int(max_length or self.max_length)
            req = ChatCompletionRequest(messages=[UserMessage(content=text)])
            out = self._tok.encode_chat_completion(req, max_model_input_len=ml if truncation else None)
            ids = list(out.tokens)
            if truncation and len(ids) > ml:
                ids = ids[:ml]
            input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        def save_pretrained(self, save_directory: str) -> None:
            # Already persisted as tekken.json/params.json in checkpoint dir.
            _ = save_directory

    return _Wrapped(tok)


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("roll_xs_factor_v2")
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


def wrap_prompt_tokenizer(tokenizer: Any, backend: str, user: str) -> str:
    if backend == "mistral":
        return f"[INST] {user} [/INST]"
    if backend == "mistral3_24b":
        # `MistralCommonTokenizerFacade` wraps with instruct formatting internally.
        return user
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
    def __init__(self, lab: pd.DataFrame, tokenizer: Any, max_length: int, backend: str):
        self.rows = lab.reset_index(drop=True)
        self.tok = tokenizer
        self.max_length = max_length
        self.backend = backend

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        r = self.rows.iloc[i]
        user = build_user_prompt_v2(str(r["combo_key"]))
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

    def __init__(self, rows: pd.DataFrame, tokenizer: Any, max_length: int, backend: str):
        self.rows = rows.reset_index(drop=True)
        self.tok = tokenizer
        self.max_length = max_length
        self.backend = backend

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.rows.iloc[i]
        user = build_user_prompt_v2(str(r["combo_key"]))
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


def config_hidden_size(config: Any) -> int:
    """Resolve backbone hidden size. Mistral3 uses nested `text_config` (no top-level hidden_size)."""
    hs = getattr(config, "hidden_size", None)
    if hs is not None:
        return int(hs)
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None:
        hs = getattr(text_cfg, "hidden_size", None)
        if hs is not None:
            return int(hs)
    raise AttributeError(f"Cannot resolve hidden_size from config type {type(config)!r}")


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
    wrapped: LMWithBucketHead,
    loader: DataLoader,
    lab: pd.DataFrame,
    device: torch.device,
    tqdm_desc: Optional[str] = None,
) -> Tuple[float, float]:
    wrapped.eval()
    correct = 0
    total = 0
    preds: List[int] = []
    it = tqdm(loader, desc=tqdm_desc) if tqdm_desc else loader
    for input_ids, attention_mask, labels in it:
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
                pct = 100.0 * (micro) / max(1, len(train_loader))
                logger.info(
                    "[roll %d] Train stop: MAX_STEPS_PER_MONTH=%d (%d micro-batches, %.1f%% of epoch "
                    "loader; bar stops early by design).",
                    roll_id,
                    MAX_STEPS_PER_MONTH,
                    micro,
                    pct,
                )
                break
        if micro % GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

    val_acc, val_spr = (0.0, float("nan"))
    if val_loader is not None and len(lab_va) > 0:
        logger.info("[roll %d] Validation: %d batches (frozen eval, no grad).", roll_id, len(val_loader))
        val_acc, val_spr = accuracy_and_spread(
            wrapped, val_loader, lab_va, device, tqdm_desc=f"roll{roll_id} val"
        )
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
        help="Override micro-batch size (default: 4, or 1 for qwen32 / mistral3_24b).",
    )
    ap.add_argument(
        "--grad_accum",
        type=int,
        default=None,
        help="Override gradient accumulation steps (default: 4, or 16 for qwen32 / mistral3_24b).",
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
    if args.backend in ("qwen32", "mistral3_24b"):
        if args.batch_size is None:
            BATCH_SIZE = 1
        if args.grad_accum is None:
            GRAD_ACCUM = 16

    bc = BACKENDS[args.backend]
    run_id = (
        args.backend
        + "_v2_xs_roll"
        + str(args.n_rolls)
        + "_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
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

    logger.info("pipeline_version=v2 prompt_module=prompt_xs_bucket_v2")
    logger.info("backend=%s run_id=%s", args.backend, run_id)
    logger.info("start_train_month=%s n_rolls=%d", args.start_train_month, args.n_rolls)
    logger.info("BATCH_SIZE=%d GRAD_ACCUM=%d", BATCH_SIZE, GRAD_ACCUM)

    data = load_data_xs(logger)
    df_all = attach_combo_keys(data.df_events, data.event_cols)
    ret_str = data.ret_clean.copy()
    ret_str.index = ret_str.index.strftime("%Y-%m-%d")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    use_auto = args.multi_gpu_auto or (
        args.backend in ("qwen32", "mistral3_24b") and torch.cuda.is_available() and torch.cuda.device_count() > 1
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

    # Compute nodes often have no outbound network. Transformers 5.x may call the Hub when
    # initializing some slow tokenizers (e.g. model_info for regex patching). Use cache only.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    logger.info(
        "hub_offline: HF_HUB_OFFLINE=%s TRANSFORMERS_OFFLINE=%s",
        os.environ.get("HF_HUB_OFFLINE"),
        os.environ.get("TRANSFORMERS_OFFLINE"),
    )

    tokenizer = load_tokenizer_for_backend(bc, resume_ckpt_dir=args.resume_ckpt_dir, logger=logger)

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
    elif bc.use_mistral3:
        from transformers import Mistral3ForConditionalGeneration

        base_lm = Mistral3ForConditionalGeneration.from_pretrained(bc.model_id, **lm_kw)
        logger.info("Loaded %s via Mistral3ForConditionalGeneration", bc.model_id)
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
    wrapped = LMWithBucketHead(lm, config_hidden_size(lm.config), N_CLASS)
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
            {
                "pipeline_version": "v2",
                "prompt_module": "prompt_xs_bucket_v2",
                "backend": args.backend,
                "run_id": run_id,
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


if __name__ == "__main__":
    main()
