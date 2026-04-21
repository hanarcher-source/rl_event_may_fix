#!/usr/bin/env python3
# coding: utf-8
"""
Qwen2.5-32B-Instruct: **V1 no-date prompt** pipeline, train **one** month only, frozen OOS (default 12)
with **combo_cache** (union unique combo_key across OOS). Same factor head as v1 (5-bucket softmax mean).

Same multi-GPU idea as `roll_xs_bucket_factor_qwen32b.py`: pass `--multi_gpu_auto` when >1 GPU.

Example:
  python3 -u roll_xs_bucket_factor_qwen32b_nodate_train1m_frozen_oos12.py \\
    --multi_gpu_auto --max_mem_gib 70 --start_train_month 2023-01 --n_rolls 12
"""

from __future__ import annotations

import sys

from roll_xs_bucket_factor_nodate_train1m_frozen_oos12 import main


def _ensure_argv(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    _ensure_argv("--backend", "qwen32")
    main()
