#!/usr/bin/env python3
"""命令行入口：风格择时（成长 vs 价值）当期建议 + 历史回测。

示例：
  # 离线自检（合成数据）
  python -m fof_system.run_style --source mock

  # akshare 真实国证成长/价值，跑动量+波动信号回测（估值信号需 iFinD）
  python -m fof_system.run_style --source akshare --start 2015-01-01

  # iFinD（含估值价差信号；需先在 skill 的 mcp_config.json 配好密钥）
  python -m fof_system.run_style --source ifind --start 2018-01-01
"""
from __future__ import annotations
import argparse
import logging
import sys
import pandas as pd

from .data import get_provider
from .engine.style_timing import StyleTimer
from .engine.style_backtest import run_style_backtest
from . import config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="FOF 风格择时（第③层）")
    p.add_argument("--source", default="mock", choices=["mock", "akshare", "ifind"],
                   help="价格数据源")
    p.add_argument("--val-source", default="", choices=["", "ifind", "ifind_http", "mock"],
                   help="估值数据源（启用估值价差信号）；如 --source akshare --val-source ifind")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--max-tilt", type=float, default=None, help="覆盖最大主动偏离(默认0.15)")
    p.add_argument("--no-backtest", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    if args.max_tilt is not None:
        config.STYLE_TIMING.max_tilt = args.max_tilt

    provider = get_provider(args.source)
    val_provider = get_provider(args.val_source) if args.val_source else None
    timer = StyleTimer(provider, valuation_provider=val_provider)

    # 当期建议
    view = timer.current_view(args.start, args.end)
    print("\n=== 风格择时 · 当期建议（基准 成长{:.0%}/价值{:.0%}）===".format(
        config.BENCHMARK_WEIGHTS["growth"], config.BENCHMARK_WEIGHTS["value"],
    ))
    print(f"  截至 {view.asof.date()}")
    print(f"  合成观点 composite = {view.composite:+.3f}  "
          f"（>0偏成长 / <0偏价值）")
    print(f"  目标权重：国证成长 {view.w_growth:.1%}  |  国证价值 {view.w_value:.1%}")
    print(f"  相对基准成长净偏离：{view.active_growth:+.1%}")
    if view.contributions:
        print("  各信号当期观点：")
        for k, val in view.contributions.items():
            tilt = "偏成长" if val > 0 else "偏价值"
            print(f"    - {k:<11}: {val:+.3f} ({tilt})")
    if "valuation" not in view.contributions:
        print("  ⚠ 估值价差信号未启用（数据源无指数PE/PB；配好 iFinD 密钥后自动启用）")

    # 回测
    if not args.no_backtest:
        g = provider.get_index_close(config.STYLE_FACTORS["growth"], args.start, args.end)
        v = provider.get_index_close(config.STYLE_FACTORS["value"], args.start, args.end)
        wdf = timer.target_weight_series(args.start, args.end)
        res, _ = run_style_backtest(g, v, wdf, rebalance=config.STYLE_TIMING.rebalance)
        print("\n=== 风格择时 · 回测（tilt vs 考核基准）===")
        d = res.as_dict()
        labels = {
            "ann_return_tilt": "择时组合年化", "ann_return_bench": "基准年化",
            "ann_excess": "年化超额", "tracking_error": "跟踪误差",
            "info_ratio": "信息比率IR", "hit_rate": "调仓胜率",
            "avg_active_growth": "平均成长偏离", "turnover_annual": "年化换手",
            "active_max_drawdown": "主动最大回撤", "n_periods": "调仓期数",
        }
        for k, lab in labels.items():
            val = d[k]
            if k in ("n_periods",):
                print(f"  {lab:<10}: {int(val)}")
            elif k in ("info_ratio",):
                print(f"  {lab:<10}: {val:.2f}")
            else:
                print(f"  {lab:<10}: {val:+.2%}")
        print("\n  解读：年化超额>0 且 IR 越高，说明风格择时稳定贡献超额；"
              "\n        换手反映交易成本压力；主动回撤反映择时做错时的痛感。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
