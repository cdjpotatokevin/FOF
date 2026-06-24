#!/usr/bin/env python3
"""第⑥层稳健性检查：成本与约束敏感性。"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from .data import get_provider
from .data.pit import PITDataStore
from .engine.robustness import run_robustness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FOF稳健性压力测试（参数与交易成本）")
    parser.add_argument("--source", default="mock", choices=["mock", "akshare", "ifind"])
    parser.add_argument("--val-source", default="", choices=["", "mock", "ifind", "ifind_http"])
    parser.add_argument("--codes", default="", help="候选代码，逗号分隔；mock可留空")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="")
    parser.add_argument("--pit-root", default="", help="历史PIT仓；未提供时结果含静态池偏差")
    parser.add_argument("--out", default="", help="输出CSV路径")
    args = parser.parse_args(argv)

    provider = get_provider(args.source)
    val_provider = get_provider(args.val_source) if args.val_source else None
    codes = [code.strip() for code in args.codes.split(",") if code.strip()]
    if not codes:
        if args.source != "mock":
            parser.error("真实数据压力测试必须指定候选代码")
        codes = list(provider.truth.keys())  # type: ignore[attr-defined]

    result = run_robustness(
        provider, codes, args.start, args.end, val_provider=val_provider,
        pit_store=PITDataStore(args.pit_root) if args.pit_root else None,
    )
    display = result.copy()
    for column in ["ann_excess", "annual_turnover", "total_transaction_cost"]:
        if column in display:
            display[column] = display[column].map(lambda value: f"{value:+.2%}" if pd.notna(value) else "")
    if "info_ratio" in display:
        display["info_ratio"] = display["info_ratio"].map(lambda value: f"{value:+.2f}" if pd.notna(value) else "")
    print("\n=== 稳健性压力测试 ===")
    print(display.to_string(index=False))
    successful = result.dropna(subset=["ann_excess"])
    if not successful.empty:
        robust = bool((successful["ann_excess"] > 0).all() and (successful["info_ratio"] > 0).all())
        print("\n结论：" + ("所有已完成情景均为正超额/正IR。" if robust else "至少一个情景为负超额或负IR，不能宣称策略稳健。"))
    if args.out:
        result.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"明细已写入：{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
