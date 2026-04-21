#!/usr/bin/env python3
# coding: utf-8
"""
Entry point for Qwen2.5-32B-Instruct rolling bucket pipeline.

Same code as roll_xs_bucket_factor_pipeline.py with defaults:
  --backend qwen32
  (multi-GPU sharding is auto-enabled when >1 GPU is visible; see --multi_gpu_auto / --max_mem_gib)

Typical Slurm: request 4 GPUs; this script does not use `accelerate launch` — it relies on
transformers `device_map="auto"` + `max_memory` (same idea as multi-GPU HF inference sharding).

Example:
  python3 -u roll_xs_bucket_factor_qwen32b.py --start_train_month 2023-01 --n_rolls 12
"""

from __future__ import annotations

import sys

from roll_xs_bucket_factor_pipeline import main


def _ensure_argv(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    _ensure_argv("--backend", "qwen32")
    main()
