#!/usr/bin/env python3
# coding: utf-8
"""
Explicit alias for the Qwen2.5-32B **V2 prompt** rolling run.

V2 for Qwen32 means `roll_xs_bucket_factor_pipeline_v2.py` with:
  --backend qwen32

This preserves the old no-date Qwen32 runner as V1 and provides a separate entrypoint for the v2
prompt module (`prompt_xs_bucket_v2`).

Defaults follow the shared v2 pipeline:
  - Qwen32 auto-defaults to BATCH_SIZE=1, GRAD_ACCUM=16 unless overridden.
  - Multi-GPU auto sharding is enabled when >1 GPU is visible.
"""

from __future__ import annotations

import sys

from roll_xs_bucket_factor_pipeline_v2 import main


def _ensure_argv(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    _ensure_argv("--backend", "qwen32")
    main()
