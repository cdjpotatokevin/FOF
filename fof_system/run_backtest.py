#!/usr/bin/env python3
"""命令行入口：第⑥层整链 walk-forward 回测。

每月末仅以当时可得的净值和风格信号重建组合，下一持有期结算净收益；首次建仓和
换仓成本均计入。它不是对单一静态组合做事后归因，而是检验 ②选基、③风格目标和
④优化器联动后能否在真实决策顺序下创造超额。

示例：
  python -m fof_system.run_backtest --source mock
  python -m fof_system.run_backtest --source akshare --val-source ifind \\
      --codes 005827,163406,110011,161005,260108 --start 2016-01-01
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from . import config
from .data import get_provider
from .data.pit import PITDataStore
from .engine.backtest import ChainBacktester


def _print_pct(label: str, value: float) -> None:
    print(f"  {label:<18}: {value:+.2%}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FOF 第⑥层：整链滚动回测")
    parser.add_argument("--source", default="mock", choices=["mock", "akshare", "ifind"])
    parser.add_argument("--val-source", default="", choices=["", "mock", "ifind", "ifind_http"])
    parser.add_argument("--codes", default="", help="候选基金代码，逗号分隔")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="")
    parser.add_argument("--warmup", type=int, default=None, help="首次调仓前的周频历史数")
    parser.add_argument("--max-weight", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--te-budget", type=float, default=None)
    parser.add_argument("--fund-cost", type=float, default=None,
                        help="主动基金单边交易成本，例如 0.0015")
    parser.add_argument("--etf-cost", type=float, default=None,
                        help="ETF 单边交易成本，例如 0.0005")
    parser.add_argument("--out-dir", default="", help="可选：写出净值、收益和调仓权重CSV")
    parser.add_argument("--pit-root", default="", help="可选：PIT主数据仓；按每个调仓日筛选候选池")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    pd.set_option("display.width", 180)

    if args.max_weight is not None:
        config.OPTIMIZER.max_weight_fund = args.max_weight
    if args.gamma is not None:
        config.OPTIMIZER.risk_aversion = args.gamma
    if args.te_budget is not None:
        config.OPTIMIZER.te_budget_annual = args.te_budget
    if args.fund_cost is not None:
        config.BACKTEST.fund_one_way_cost = args.fund_cost
    if args.etf_cost is not None:
        config.BACKTEST.etf_one_way_cost = args.etf_cost

    provider = get_provider(args.source)
    val_provider = get_provider(args.val_source) if args.val_source else None
    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    if not codes:
        if args.source != "mock":
            print("请用 --codes 指定候选基金；实盘回测不得把今天的全市场基金池倒灌进历史。")
            return 2
        codes = list(provider.truth.keys())  # type: ignore[attr-defined]

    result = ChainBacktester(
        provider, codes, args.start, args.end, val_provider=val_provider,
        pit_store=PITDataStore(args.pit_root) if args.pit_root else None,
    ).run(warmup_obs=args.warmup)
    m = result.metrics
    print(f"\n=== 第⑥层 · 整链滚动回测（{result.rebalances} 次调仓）===")
    print(f"  基准：国证成长 {config.BENCHMARK_WEIGHTS['growth']:.0%}"
          f" + 国证价值 {config.BENCHMARK_WEIGHTS['value']:.0%}")
    _print_pct("策略年化（净）", m["ann_return_strat"])
    _print_pct("策略年化（毛）", m["ann_return_gross"])
    _print_pct("基准年化", m["ann_return_bench"])
    _print_pct("年化超额（净）", m["ann_excess"])
    _print_pct("年化超额（毛）", m["ann_excess_gross"])
    _print_pct("跟踪误差", m["tracking_error"])
    print(f"  {'信息比率IR':<18}: {m['info_ratio']:.2f}")
    _print_pct("主动最大回撤", m["active_max_dd"])
    _print_pct("月度胜率", m["monthly_hit"])
    _print_pct("年化单边换手", m["annual_turnover"])
    _print_pct("累计交易成本", m["total_transaction_cost"])
    print(f"  平均成长目标暴露     : {result.target_growth_history.mean():.1%}")
    print("\n  注：基金存续、申赎限制和产品级费率必须在生产数据中逐只处理；"
          "本回测的默认成本是假设，不能替代交易台复核。")
    if not args.pit_root:
        print("  ⚠ 未提供 --pit-root：本次使用静态候选池，不可视为已消除幸存者偏差。")
    else:
        print(f"  PIT候选池             : {args.pit_root}（每个调仓日按可得状态筛选）")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        result.nav.to_csv(out_dir / "nav.csv", encoding="utf-8-sig")
        result.weekly.to_csv(out_dir / "weekly_returns.csv", encoding="utf-8-sig")
        result.weights_history.to_csv(out_dir / "weights_history.csv", encoding="utf-8-sig")
        result.target_growth_history.rename("target_growth").to_csv(
            out_dir / "target_growth_history.csv", encoding="utf-8-sig",
        )
        result.cost_history.rename("transaction_cost").to_csv(
            out_dir / "cost_history.csv", encoding="utf-8-sig",
        )
        print(f"\n  明细已写入：{out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
