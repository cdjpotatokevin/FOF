#!/usr/bin/env python3
"""命令行入口：跑基金评价打分。

示例：
  # 用合成数据离线验证整条链路（无需联网）
  python -m fof_system.run_score --source mock

  # 用 akshare 评价指定基金（逗号分隔）
  python -m fof_system.run_score --source akshare --codes 005827,110011,163406 \
      --start 2021-01-01

  # PIT严格合规全池评分；成功获取的净值会缓存，可中断后重跑
  python -m fof_system.run_score --source akshare --pit-root /data/fof_pit \
      --universe-asof 2026-06-23 --end 2026-06-23 --cache-dir /data/fof_pit/raw/akshare
"""
from __future__ import annotations
import argparse
import logging
import sys
import pandas as pd

from .data import get_provider
from .data.pit import PITDataStore
from .pipeline import score_universe
from .engine.scoring import explain_top


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="FOF 基金评价打分引擎")
    p.add_argument("--source", default="mock",
                   choices=["mock", "akshare", "tushare", "ifind"],
                   help="数据源")
    p.add_argument("--codes", default="", help="基金代码，逗号分隔；留空则全市场初筛")
    p.add_argument("--start", default="2019-01-01", help="评价起始日 YYYY-MM-DD")
    p.add_argument("--end", default="", help="评价截止日，默认今天")
    p.add_argument("--limit", type=int, default=None, help="全市场模式下最多评价多少只")
    p.add_argument("--top", type=int, default=15, help="展示前 N 名")
    p.add_argument("--out", default="", help="把完整打分表存为 CSV 的路径")
    p.add_argument("--pit-root", default="", help="PIT数据仓目录；全市场模式下按 --universe-asof 取池")
    p.add_argument("--universe-asof", default="", help="PIT基金池时点，默认评价截止日")
    p.add_argument("--cache-dir", default="", help="akshare净值/指数本地缓存目录；全池运行建议设置")
    p.add_argument("--token", default="", help="tushare token（如用 tushare）")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    kwargs = {}
    if args.source == "tushare" and args.token:
        kwargs["token"] = args.token
    if args.source == "akshare":
        cache_dir = args.cache_dir or (str(PITDataStore(args.pit_root).root / "raw" / "akshare") if args.pit_root else "")
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
    provider = get_provider(args.source, **kwargs)

    codes = [c.strip() for c in args.codes.split(",") if c.strip()] or None
    scored = score_universe(
        provider, codes=codes, start=args.start, end=args.end, limit=args.limit,
        pit_store=PITDataStore(args.pit_root) if args.pit_root else None,
        universe_asof=args.universe_asof or None,
    )

    print("\n=== 基金评价打分 · 前 {} 名（综合分越高越能贡献相对基准的超额）===\n".format(args.top))
    top = explain_top(scored, n=args.top)
    print(top.to_string(index=False))

    print("\n指标释义："
          "\n  style_alpha_ann  风格调整后年化alpha（剥离成长/价值后的选股超额，核心）"
          "\n  info_ratio       信息比率（alpha/主动波动，越高越稳）"
          "\n  style_vs_bench   相对考核基准的成长净偏离（>0偏成长，<0偏价值）"
          "\n  excess_vs_bench  相对考核基准的年化超额；te_vs_bench 为对应跟踪误差")

    if args.out:
        scored.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n完整打分表已写入：{args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
