#!/usr/bin/env python3
"""命令行入口：第⑥层整链 walk-forward 回测。

与 ``run_portfolio --all-eligible`` 对齐：月末调仓走 ``rebalance_at`` 共享路径。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data import get_provider
from .data.pit import PITDataStore
from .engine.backtest import ChainBacktester
from .rebalance import rebalance_at


def _default_pit_root() -> str:
    candidate = Path(__file__).resolve().parent / "fof_pit_data"
    return str(candidate) if candidate.exists() else ""


def _print_pct(label: str, value: float) -> None:
    print(f"  {label:<18}: {value:+.2%}")


def _compare_portfolio(golden: Path, weights: pd.Series, tol: float = 1e-3) -> tuple[bool, str]:
    ref = pd.read_csv(golden, dtype={"code": "string"})
    ref["code"] = ref["code"].astype(str)
    ref_w = ref.set_index("code")["weight"].astype(float)
    merged = pd.concat([ref_w.rename("golden"), weights.rename("rebalance")], axis=1).fillna(0.0)
    diff = (merged["golden"] - merged["rebalance"]).abs()
    max_diff = float(diff.max()) if not diff.empty else 0.0
    ok = max_diff <= tol
    detail = f"最大权重偏差 {max_diff:.4%}（容差 {tol:.2%}）"
    if not ok:
        worst = diff.sort_values(ascending=False).head(5)
        detail += "\n  偏差最大：" + ", ".join(f"{code} Δ{delta:.2%}" for code, delta in worst.items())
    return ok, detail


def _run_rebalance_check(args: argparse.Namespace) -> int:
    pit_root = args.pit_root or _default_pit_root()
    if not pit_root:
        print("rebalance-check 需要 --pit-root 或项目内 fof_pit_data 目录。")
        return 2
    provider_kwargs = {}
    if args.source == "akshare":
        cache_dir = args.cache_dir or str(Path(pit_root) / "raw" / "akshare")
        provider_kwargs["cache_dir"] = cache_dir
    provider = get_provider(args.source, **provider_kwargs)
    val_provider = get_provider(args.val_source) if args.val_source else None
    asof = args.rebalance_asof or args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    pit_store = PITDataStore(pit_root)

    if args.max_weight is not None:
        config.OPTIMIZER.max_weight_fund = args.max_weight

    result = rebalance_at(
        provider,
        asof=asof,
        eval_start=args.eval_start,
        pit_store=pit_store,
        val_provider=val_provider,
        opt_cfg=config.OPTIMIZER,
        strict_eligibility=args.strict_eligibility,
        target_growth=args.target_growth,
        portfolio_aum_yi=args.portfolio_aum_yi,
        max_order_to_fund_aum=args.max_order_to_fund_aum,
        etf_participation_rate=args.etf_participation_rate,
        etf_adv_lookback=args.etf_adv_lookback,
        enforce_active_fund_capacity=args.enforce_active_capacity,
        enforce_etf_capacity=args.enforce_etf_capacity,
        score_cache_dir=args.score_cache_dir or None,
        capacity_asof=args.capacity_asof or asof,
        scores_in=args.scores_in or None,
        skip_rolling=not args.no_fast_metrics,
        preselect_pool=args.score_pool if args.score_pool is not None else config.BACKTEST.backtest_preselect_pool,
        workers=args.workers if args.workers is not None else config.BACKTEST.backtest_workers,
        provider_kwargs=provider_kwargs,
    )
    if not result.success or result.weights is None:
        print(f"调仓失败：{result.skip_reason}")
        return 1

    print(f"\n=== rebalance-check @ {asof} ===")
    print(f"  评分成功：{result.scored_count} 只 | 优化候选：{result.candidate_count} 只")
    print(f"  目标成长暴露：{result.target_growth:.2%}")
    print(f"  持仓数：{int((result.weights > 1e-6).sum())}")
    show = result.detail.copy()
    show["weight"] = (show["weight"] * 100).round(2).astype(str) + "%"
    print(show.to_string(index=False))

    if args.golden_portfolio:
        ok, detail = _compare_portfolio(Path(args.golden_portfolio), result.weights, tol=args.weight_tol)
        print(f"\n  与 golden 对比：{'通过' if ok else '未通过'} — {detail}")
        return 0 if ok else 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FOF 第⑥层：整链滚动回测（与生产对齐）")
    parser.add_argument("--source", default="mock", choices=["mock", "akshare", "ifind"])
    parser.add_argument("--val-source", default="", choices=["", "mock", "ifind", "ifind_http"])
    parser.add_argument("--codes", default="", help="mock/小样本模式候选代码；全池模式可省略")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--eval-start", default="2019-01-01", help="评分/RBSA 窗口起点")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--max-weight", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--te-budget", type=float, default=None)
    parser.add_argument("--fund-cost", type=float, default=0.0015)
    parser.add_argument("--etf-cost", type=float, default=0.0003)
    parser.add_argument("--pit-root", default="", help="PIT 数据仓；默认使用项目内 fof_pit_data")
    parser.add_argument("--score-cache-dir", default="", help="按 asof 缓存全池评分 CSV")
    parser.add_argument("--cache-dir", default="", help="akshare 净值缓存目录")
    parser.add_argument("--portfolio-aum-yi", type=float, default=config.PORTFOLIO_AUM_YI)
    parser.add_argument("--max-order-to-fund-aum", type=float, default=config.OPTIMIZER.max_order_to_fund_aum)
    parser.add_argument("--etf-participation-rate", type=float, default=0.10)
    parser.add_argument("--etf-adv-lookback", type=int, default=20)
    parser.add_argument("--capacity-asof", default="")
    parser.add_argument("--enforce-active-capacity", action="store_true",
                        help="启用主动基金订单/规模容量上限（默认关闭，可另跑对照）")
    parser.add_argument("--enforce-etf-capacity", action="store_true",
                        help="启用 ETF 成交额容量上限（默认关闭）")
    parser.add_argument("--strict-eligibility", action="store_true",
                        help="PIT 严格合规（AUM/成立日/开放申购）；默认关闭")
    parser.add_argument("--legacy-codes", action="store_true",
                        help="关闭全池评分，仅对 --codes 列表调仓（mock 自检）")
    parser.add_argument("--rebalance-freq", default="QE", choices=["QE", "ME"],
                        help="调仓频率：QE=季度末（默认，对齐PIT）；ME=月末")
    parser.add_argument("--prefetch-all-pit", action="store_true",
                        help="预取全区间PIT并集净值（极慢，默认关闭）")
    parser.add_argument("--no-fast-metrics", action="store_true",
                        help="回测也计算滚动 alpha_consistency（极慢，与生产完全一致）")
    parser.add_argument("--score-pool", type=int, default=None,
                        help="全池轻量RBSA后保留top-N进入综合分（默认400；0=不截断）")
    parser.add_argument("--workers", type=int, default=None, help="评分并行进程数（默认4）")
    parser.add_argument("--scores-in", default="", help="复用已落盘评分 CSV（rebalance-check / 加速）")
    parser.add_argument("--rebalance-check", action="store_true",
                        help="单期调仓验收：对比 --golden-portfolio")
    parser.add_argument("--target-growth", type=float, default=None,
                        help="覆盖风格目标成长暴露（验收时与生产 summary 对齐）")
    parser.add_argument("--rebalance-asof", default="2026-06-23")
    parser.add_argument("--golden-portfolio", default="",
                        help="验收用生产组合 CSV（含 code,weight 列）")
    parser.add_argument("--weight-tol", type=float, default=0.02,
                        help="rebalance-check 权重容差，默认 2pct")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    pd.set_option("display.width", 180)

    if args.rebalance_check:
        return _run_rebalance_check(args)

    if args.max_weight is not None:
        config.OPTIMIZER.max_weight_fund = args.max_weight
    if args.gamma is not None:
        config.OPTIMIZER.risk_aversion = args.gamma
    if args.te_budget is not None:
        config.OPTIMIZER.te_budget_annual = args.te_budget

    bt_cfg = config.BACKTEST
    bt_cfg.fund_one_way_cost = args.fund_cost
    bt_cfg.etf_one_way_cost = args.etf_cost
    bt_cfg.portfolio_aum_yi = args.portfolio_aum_yi
    bt_cfg.max_order_to_fund_aum = args.max_order_to_fund_aum
    bt_cfg.etf_participation_rate = args.etf_participation_rate
    bt_cfg.etf_adv_lookback = args.etf_adv_lookback
    bt_cfg.enforce_active_fund_capacity = args.enforce_active_capacity
    bt_cfg.enforce_etf_capacity = args.enforce_etf_capacity
    bt_cfg.strict_eligibility = args.strict_eligibility
    bt_cfg.eval_start = args.eval_start
    bt_cfg.score_cache_dir = args.score_cache_dir
    bt_cfg.full_universe_scoring = not args.legacy_codes
    bt_cfg.rebalance_freq = args.rebalance_freq
    bt_cfg.prefetch_all_pit_codes = args.prefetch_all_pit
    bt_cfg.backtest_skip_rolling = not args.no_fast_metrics
    if args.score_pool is not None:
        bt_cfg.backtest_preselect_pool = args.score_pool
    if args.workers is not None:
        bt_cfg.backtest_workers = args.workers

    provider_kwargs = {}
    pit_root = args.pit_root or _default_pit_root()
    if args.source == "akshare":
        cache_dir = args.cache_dir or (str(Path(pit_root) / "raw" / "akshare") if pit_root else "")
        if cache_dir:
            provider_kwargs["cache_dir"] = cache_dir
    provider = get_provider(args.source, **provider_kwargs)
    val_provider = get_provider(args.val_source) if args.val_source else None

    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    pit_store = PITDataStore(pit_root) if pit_root else None
    if not codes:
        if args.source == "mock":
            codes = list(provider.truth.keys())  # type: ignore[attr-defined]
            bt_cfg.full_universe_scoring = False
        elif pit_store is None:
            print("全池回测需要 --pit-root 或项目内 fof_pit_data。")
            return 2

    result = ChainBacktester(
        provider, codes, args.start, args.end, val_provider=val_provider,
        backtest_cfg=bt_cfg, pit_store=pit_store,
    ).run(warmup_obs=args.warmup)
    m = result.metrics
    print(f"\n=== 第⑥层 · 整链滚动回测（{result.rebalances} 次调仓）===")
    print(f"  基准：国证成长 {config.BENCHMARK_WEIGHTS['growth']:.0%}"
          f" + 国证价值 {config.BENCHMARK_WEIGHTS['value']:.0%}")
    for label, key in [
        ("策略年化（净）", "ann_return_strat"),
        ("策略年化（毛）", "ann_return_gross"),
        ("基准年化", "ann_return_bench"),
        ("年化超额（净）", "ann_excess"),
        ("年化超额（毛）", "ann_excess_gross"),
        ("跟踪误差", "tracking_error"),
    ]:
        _print_pct(label, m[key])
    print(f"  {'信息比率IR':<18}: {m['info_ratio']:.2f}")
    _print_pct("主动最大回撤", m["active_max_dd"])
    _print_pct("月度胜率", m["monthly_hit"])
    _print_pct("年化单边换手", m["annual_turnover"])
    _print_pct("累计交易成本", m["total_transaction_cost"])
    print(f"  跳过调仓次数         : {m.get('skipped_rebalances', 0)}")
    if not result.target_growth_history.empty:
        print(f"  平均成长目标暴露     : {result.target_growth_history.mean():.1%}")
    for note in result.notes:
        print(f"  · {note}")

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
        if not result.rebalance_log.empty:
            result.rebalance_log.to_csv(out_dir / "rebalance_log.csv", index=False, encoding="utf-8-sig")
        print(f"\n  明细已写入：{out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
