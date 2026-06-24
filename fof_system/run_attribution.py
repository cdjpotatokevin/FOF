#!/usr/bin/env python3
"""命令行入口：第⑤层组合归因（超额来自选基 vs 风格择时）。

先按 ②③④ 建出组合，再在历史窗口上把组合相对考核基准的超额，
恒等地拆成"风格择时(③)"与"选股(②)"两部分，并给逐基金选股贡献。

示例：
  # 离线自检（合成数据，整条 ②③④⑤）
  python -m fof_system.run_attribution --source mock

  # 真实：候选基金 + 第③层风格目标 + 归因
  python -m fof_system.run_attribution --source akshare --val-source ifind \
      --codes 005827,163406,110011,161005,260108 --start 2019-01-01
"""
from __future__ import annotations
import argparse
import logging
import sys
import pandas as pd

from .data import get_provider
from .data.pit import PITDataStore
from .portfolio import build_portfolio, resolve_target_growth
from .pipeline import build_benchmark
from .engine.attribution import style_selection_attribution, synth_etf_returns
from . import config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="FOF 组合归因（第⑤层）")
    p.add_argument("--source", default="mock", choices=["mock", "akshare", "ifind"])
    p.add_argument("--val-source", default="", choices=["", "ifind", "ifind_http", "mock"])
    p.add_argument("--codes", default="", help="候选基金代码，逗号分隔")
    p.add_argument("--target-growth", type=float, default=None,
                   help="目标成长暴露 0~1；不给则由第③层自动决定")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--pit-root", default="", help="PIT数据仓；用于识别实际ETF与PIT主数据")
    p.add_argument("--universe-asof", default="", help="PIT元数据时点，默认评价截止日")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    pd.set_option("display.width", 200)

    provider = get_provider(args.source)
    val_provider = get_provider(args.val_source) if args.val_source else None

    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    if not codes:
        if args.source == "mock":
            codes = list(provider.truth.keys())  # type: ignore[attr-defined]
        else:
            print("请用 --codes 指定候选基金。")
            return 2

    # 目标成长暴露 → 建组合
    if args.target_growth is not None:
        target_g = args.target_growth
    else:
        target_g = resolve_target_growth(provider, val_provider, args.start, args.end)
    asset_metadata = {}
    if args.pit_root:
        asof = args.universe_asof or args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
        universe = PITDataStore(args.pit_root).read_universe_asof(asof, asset_types=("fund", "etf"))
        asset_metadata = {str(row["code"]): row.to_dict() for _, row in universe.iterrows()}
    res, detail = build_portfolio(provider, codes, target_g, args.start, args.end,
                                  asset_metadata=asset_metadata)

    # 归因窗口的风格因子收益（weekly，与系统一致）
    factor_df, _ = build_benchmark(provider, args.start, args.end)

    # 组装持有资产的收益矩阵
    weights = detail.set_index("code")["weight"]
    growth_load = detail.set_index("code")["growth_load"]
    value_load = 1.0 - growth_load                      # RBSA 两因子和为1
    market_codes = detail.loc[detail["type"].isin(["主动基金", "ETF"]), "code"].tolist()
    etf_codes = detail.loc[detail["type"] == "ETF补全", "code"].tolist()

    ret_cols = {}
    for c in market_codes:
        nav = provider.get_fund_nav(c, args.start, args.end)
        ret_cols[c] = provider.to_returns(nav, config.RETURN_FREQ)
    returns_df = pd.DataFrame(ret_cols)
    if etf_codes:
        etf_ret = synth_etf_returns(factor_df, {e: float(growth_load[e]) for e in etf_codes})
        returns_df = pd.concat([returns_df, etf_ret], axis=1)

    ppy = config.PERIODS_PER_YEAR[config.RETURN_FREQ]
    attr = style_selection_attribution(
        weights=weights, returns=returns_df,
        growth_load=growth_load, value_load=value_load,
        style_returns=factor_df, bench_growth_weight=config.BENCHMARK_WEIGHTS["growth"],
        ppy=ppy,
    )

    # ---- 输出 ----
    print(f"\n=== 第⑤层 · 组合归因（{attr.diagnostics['n_periods']} 期 ≈ {attr.years:.1f} 年）===")
    print(f"  组合累计收益 {attr.cum_port:+.2%} | 基准累计 {attr.cum_bench:+.2%} | "
          f"累计超额 {attr.cum_excess:+.2%}")
    print(f"  组合平均成长暴露 {attr.avg_growth_exposure:.1%}"
          f"（基准 {config.BENCHMARK_WEIGHTS['growth']:.0%}）")

    print("\n  超额来源拆解（年化）：")
    print(f"    ├─ 风格择时(③)  {attr.ann_style_timing:+.2%}")
    print(f"    ├─ 选股(②)      {attr.ann_selection:+.2%}")
    print(f"    └─ 合计          {attr.ann_excess:+.2%}"
          f"   (几何链接残差 {attr.linking_residual/attr.years:+.2%}/年)")

    bigger = "选股" if attr.total_selection > attr.total_style_timing else "风格择时"
    print(f"\n  结论：本组合的超额主要由 **{bigger}** 贡献。")

    print("\n  逐基金选股贡献（窗口累计，正=跑赢自身风格）：")
    fs = attr.fund_selection
    name_map = detail.set_index("code")["name"].to_dict()
    for code, val in fs.items():
        tag = "" if code in name_map and detail.set_index("code").loc[code, "type"] == "主动基金" else "(ETF)"
        print(f"    {code:<14}{name_map.get(code,''):<12}{val:+.2%} {tag}")

    print("\n  说明：每期超额 = 风格择时 + 选股（由 RBSA 载荷恒等拆分，无残差）；"
          "\n        跨期的几何链接残差来自复利与载荷时变，通常很小。"
          "\n        行业层归因需 iFinD 持仓穿透，作为后续扩展。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
