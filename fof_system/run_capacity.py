#!/usr/bin/env python3
"""ETF容量压力检查：从PIT行情读取成交额并计算可承载的组合规模。"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from .data.pit import PITDataStore
from .engine.capacity import etf_capacity_report


def _weights(value: str) -> pd.Series:
    pairs = [piece.strip() for piece in value.split(",") if piece.strip()]
    data = {}
    for pair in pairs:
        try:
            code, weight = pair.split("=", 1)
            data[code.strip()] = float(weight)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("weights 格式应为 159915=0.25,510300=0.10") from exc
    if not data:
        raise argparse.ArgumentTypeError("weights 不能为空")
    return pd.Series(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FOF ETF流动性与容量压力检查")
    parser.add_argument("--pit-root", required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--weights", required=True, type=_weights)
    parser.add_argument("--participation-rate", type=float, default=0.10)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--portfolio-aum-yi", type=float, default=None)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    store = PITDataStore(args.pit_root)
    frames = [store.read_market_asof(code, args.asof) for code in args.weights.index]
    market = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    report = etf_capacity_report(
        market, args.weights, participation_rate=args.participation_rate,
        lookback=args.lookback, portfolio_aum_yi=args.portfolio_aum_yi,
    )
    if report.empty:
        parser.error("没有可用ETF成交额；请先使用 run_data ingest-etf 写入PIT行情")
    display = report.copy()
    for column in ["weight", "participation_rate", "order_participation"]:
        if column in display:
            display[column] = display[column].map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
    print("\n=== ETF容量压力 ===")
    print(display.to_string())
    if "max_portfolio_yi" in report:
        print(f"\n按当前ETF权重，参与率上限下的组合规模上限：{report['max_portfolio_yi'].min():.2f} 亿元")
    if args.out:
        report.to_csv(args.out, encoding="utf-8-sig")
        print(f"明细已写入：{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
