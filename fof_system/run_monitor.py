#!/usr/bin/env python3
"""命令行入口：调仓前风险快照与基金风格漂移预警。

本入口监控的是按当前模型构建的目标组合。它把风格目标、事前特质 TE、集中度和每只
基金相对自身历史中枢的滚动 RBSA 漂移放在同一份检查单中；预警用于触发人工复核，
不构成自动卖出指令。
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from . import config
from .data import get_provider
from .data.pit import PITDataStore
from .engine import risk_model
from .pipeline import build_benchmark
from .portfolio import build_portfolio, resolve_target_growth


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FOF 第⑥层：调仓前风险与风格漂移监控")
    parser.add_argument("--source", default="mock", choices=["mock", "akshare", "ifind"])
    parser.add_argument("--val-source", default="", choices=["", "mock", "ifind", "ifind_http"])
    parser.add_argument("--codes", default="", help="候选基金代码，逗号分隔")
    parser.add_argument("--target-growth", type=float, default=None)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="")
    parser.add_argument("--pit-root", default="", help="PIT数据仓；用于识别实际ETF与PIT主数据")
    parser.add_argument("--universe-asof", default="", help="PIT元数据时点，默认评价截止日")
    parser.add_argument("--drift-window", type=int, default=config.ROLLING_WINDOW)
    parser.add_argument("--drift-threshold", type=float, default=0.15)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    provider = get_provider(args.source)
    val_provider = get_provider(args.val_source) if args.val_source else None
    codes = [code.strip() for code in args.codes.split(",") if code.strip()]
    if not codes:
        if args.source != "mock":
            print("请用 --codes 指定候选基金。")
            return 2
        codes = list(provider.truth.keys())  # type: ignore[attr-defined]

    target_growth = args.target_growth
    if target_growth is None:
        target_growth = resolve_target_growth(provider, val_provider, args.start, args.end)
    asset_metadata = {}
    if args.pit_root:
        asof = args.universe_asof or args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
        universe = PITDataStore(args.pit_root).read_universe_asof(asof, asset_types=("fund", "etf"))
        asset_metadata = {str(row["code"]): row.to_dict() for _, row in universe.iterrows()}
    result, detail = build_portfolio(provider, codes, target_growth, args.start, args.end,
                                     asset_metadata=asset_metadata)
    diagnostics = result.diagnostics

    print("\n=== 第⑥层 · 调仓前风险检查 ===")
    print(f"  目标/实际成长暴露：{target_growth:.1%} / {result.realized_growth_exposure:.1%}")
    print(f"  相对70/30基准成长偏离：{diagnostics['active_growth_vs_benchmark']:+.1%}")
    print(f"  期望选基alpha / 特质TE / IR：{diagnostics['ex_ante_alpha']:+.2%}"
          f" / {diagnostics['ex_ante_idio_te']:.2%} / {diagnostics['ex_ante_ir']:.2f}")
    print(f"  ETF合计 / HHI / 有效持仓：{diagnostics['etf_weight']:.1%}"
          f" / {diagnostics['hhi']:.3f} / {diagnostics['effective_holdings']:.1f}")
    for alert in diagnostics["alerts"]:
        print(f"  ⚠ {alert}")
    if not diagnostics["style_feasible"]:
        print("  ⚠ 风格目标不可达："
              f"可达成长区间 {diagnostics['growth_reachable_min']:.1%}"
              f"~{diagnostics['growth_reachable_max']:.1%}")

    factor_df, _ = build_benchmark(provider, args.start, args.end)
    fund_codes = detail.loc[detail["type"] == "主动基金", "code"].tolist()
    return_map = {
        code: provider.to_returns(provider.get_fund_nav(code, args.start, args.end), config.RETURN_FREQ)
        for code in fund_codes
    }
    drift = risk_model.style_drift_report(
        pd.DataFrame(return_map), factor_df, args.drift_window,
        threshold=args.drift_threshold,
        rf_per_period=config.RISK_FREE_ANNUAL / config.PERIODS_PER_YEAR[config.RETURN_FREQ],
    )
    print("\n  基金风格漂移（相对自身滚动历史中枢）：")
    if drift.empty:
        print("    样本不足，未形成滚动RBSA载荷。")
    else:
        show = drift.copy()
        for column in ["latest_growth_load", "historical_growth_median", "growth_drift"]:
            show[column] = show[column].map(lambda x: f"{x:+.1%}")
        print(show.to_string())
    print("\n  注：预警代表需要复核经理风格、持仓和组织变化；不是机械调仓信号。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
