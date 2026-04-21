# coding: utf-8
"""
V2 user prompt for combo-day quintile bucket supervision.

Aligned with combo_xs_bucket_sft_lib.build_combo_day_labels + universe_quintile_bucket:
  - Same calendar day, same tradable stock pool (universe).
  - combo_key: events that co-occur on a stock-day (names joined by '|'); AND across names.
  - Target: mean open-to-open return over all stocks that satisfy the combo that day.
  - Label: that mean is placed into the same-day universe cross-section split by equal-count
    quintiles (pandas qcut q=5; edges from all stocks' returns that day). y=0 lowest, y=4 highest.

No calendar date string in the prompt (by design for v2). Template kept short to reduce noise for the
pooled hidden-state classifier. When you add roll_xs_bucket_factor_pipeline_v2.py, import build_user_prompt_v2
from here.

Label is still equal-count quintiles of the same-day universe (qcut); the text below states ordinal y only,
which matches training without over-specifying pandas details to the LM.
"""

from __future__ import annotations


def build_user_prompt_v2(combo_key: str) -> str:
    events = combo_key.split("|") if combo_key else []
    event_str = "，".join(events)
    return (
        "你是一名股票量化分析师。下列条目刻画同一交易日、同一可交易股票池中，于个股层面上同时成立的"
        "事件标签组合。\n"
        "说明：这里仅是事件名称/标签，而非长篇新闻正文。\n"
        "请据此判断：在该交易日开盘前，该可交易股票池中所有同时满足该事件标签组合的股票，其隔日收益率"
        "（本交易日开盘至下一交易日开盘，open-to-open）的截面算术平均值，"
        "在同市场股票池截面的五档有序分位中处于哪一档？\n"
        "监督标签 y ∈ {0,1,2,3,4}：y=0 表示该事件标签组合内股票隔日收益截面均值相对同市场股票池截面处于最低档，"
        "y=4 表示处于最高档；中间档位依次递增。\n"
        "事件标签组合（同时成立）：\n"
        f"{event_str}"
    ).strip()
