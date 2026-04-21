#!/usr/bin/env python3
# coding: utf-8
"""
Explicit alias for the Qwen2.5-32B **V1 prompt** rolling run.

V1 for Qwen32 means the existing **no-date** prompt path only:
`roll_xs_bucket_factor_qwen32b_nodate_roll.py`

This file preserves the old trainer unchanged and just gives it a stable name so future runs can say
"use qwen32 v1 prompt".
"""

from __future__ import annotations

from roll_xs_bucket_factor_qwen32b_nodate_roll import main


if __name__ == "__main__":
    main()
