#!/usr/bin/env python3
"""命令行入口：第④层组合构建（选基 alpha + 风格目标 → 基金/ETF 权重）。

示例：
  # 离线自检（合成数据，整条 ②③④ 链路）
  python -m fof_system.run_portfolio --source mock

  # akshare 选基 + 指定风格目标（成长暴露 55%）
  python -m fof_system.run_portfolio --source akshare \
      --codes 005827,163406,110011,161005,260108 --target-growth 0.55

  # 风格目标由第③层自动给（akshare 价格 + iFinD 估值）
  python -m fof_system.run_portfolio --source akshare --val-source ifind \
      --codes 005827,163406,110011,161005,260108
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import pandas as pd

from .data import get_provider
from .data.pit import PITDataStore
from .engine.capacity import active_fund_capacity_weight_caps, etf_capacity_weight_caps
from .engine.universe import filter_universe
from .pipeline import score_universe
from .portfolio import (
    build_portfolio, resolve_target_growth, select_backup_candidates,
    select_full_universe_candidates,
)
from . import config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="FOF 组合优化器（第④层）")
    p.add_argument("--source", default="mock", choices=["mock", "akshare", "ifind"])
    p.add_argument("--val-source", default="", choices=["", "ifind", "ifind_http", "mock"])
    p.add_argument("--codes", default="", help="候选基金代码，逗号分隔")
    p.add_argument("--all-eligible", action="store_true",
                   help="对PIT严格合规全池评分后选候选；不按规模预先截断，耗时较长")
    p.add_argument("--score-out", default="",
                   help="--all-eligible时保存完整评分表CSV，建议指定以便复核")
    p.add_argument("--scores-in", default="",
                   help="复用已落盘的全池评分CSV选候选，不重新拉取净值和评分")
    p.add_argument("--portfolio-out", default="", help="保存最终目标权重及alpha/R²明细CSV")
    p.add_argument("--summary-out", default="", help="保存组合摘要JSON")
    p.add_argument("--backup-count", type=int, default=5, help="未持有主动基金备选数量，默认5")
    p.add_argument("--backup-out", default="", help="保存备选基金CSV；需配合--scores-in或--all-eligible")
    p.add_argument("--target-growth", type=float, default=None,
                   help="目标成长暴露 0~1；不给则由第③层自动决定")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="")
    p.add_argument("--max-weight", type=float, default=None, help="单只基金权重上限")
    p.add_argument("--gamma", type=float, default=None, help="风险厌恶系数")
    p.add_argument("--te-budget", type=float, default=None, help="年化特质TE上限")
    p.add_argument("--pit-root", default="", help="PIT数据仓；用于识别实际ETF与PIT主数据")
    p.add_argument("--universe-asof", default="", help="PIT元数据时点，默认评价截止日")
    p.add_argument("--cache-dir", default="", help="akshare净值/指数本地缓存目录；全池运行默认写入PIT raw目录")
    p.add_argument("--portfolio-aum-yi", type=float, default=config.PORTFOLIO_AUM_YI,
                   help="FOF规模（亿元）；默认14亿元。与--pit-root联用时施加主动基金和ETF容量上限")
    p.add_argument("--max-order-to-fund-aum", type=float, default=config.OPTIMIZER.max_order_to_fund_aum,
                   help="主动基金单笔申购/基金规模上限，默认20%%")
    p.add_argument("--capacity-asof", default="",
                   help="ETF成交额在该PIT时点可见；默认与universe-asof相同")
    p.add_argument("--etf-participation-rate", type=float, default=0.10,
                   help="单日ETF最大成交额参与率，默认10%%")
    p.add_argument("--etf-adv-lookback", type=int, default=20,
                   help="ETF容量计算的平均成交额回看交易日数，默认20")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    if args.max_weight is not None:
        config.OPTIMIZER.max_weight_fund = args.max_weight
    if args.gamma is not None:
        config.OPTIMIZER.risk_aversion = args.gamma
    if args.te_budget is not None:
        config.OPTIMIZER.te_budget_annual = args.te_budget

    provider_kwargs = {}
    if args.source == "akshare":
        cache_dir = args.cache_dir or (str(PITDataStore(args.pit_root).root / "raw" / "akshare") if args.pit_root else "")
        if cache_dir:
            provider_kwargs["cache_dir"] = cache_dir
    provider = get_provider(args.source, **provider_kwargs)
    val_provider = get_provider(args.val_source) if args.val_source else None

    asof = args.universe_asof or args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    asset_metadata = {}
    strict_eligible_codes: set[str] | None = None
    if args.pit_root:
        pit_universe = PITDataStore(args.pit_root).read_universe_asof(asof, asset_types=("fund", "etf"))
        strict_universe = filter_universe(pit_universe, asof=asof, strict_eligibility=True)
        strict_eligible_codes = set(strict_universe["code"].astype(str))
        asset_metadata = {str(row["code"]): row.to_dict() for _, row in pit_universe.iterrows()}

    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    inferred_target_growth = None
    scored_for_selection: pd.DataFrame | None = None
    if not codes:
        if args.source == "mock":
            codes = list(provider.truth.keys())  # type: ignore[attr-defined]
        elif args.scores_in:
            scored = pd.read_csv(args.scores_in, dtype={"code": "string"})
            if strict_eligible_codes is None:
                p.error("--scores-in 需要 --pit-root，以核验基金的最新开放申购状态。")
            scored = scored.loc[scored["code"].astype(str).isin(strict_eligible_codes)].copy()
            if scored.empty:
                p.error("评分文件中没有同时满足严格范围和开放申购条件的产品。")
            scored_for_selection = scored
            inferred_target_growth = args.target_growth
            if inferred_target_growth is None:
                inferred_target_growth = resolve_target_growth(provider, val_provider, args.start, args.end)
            codes = select_full_universe_candidates(
                scored, config.OPTIMIZER.n_candidates, target_growth=inferred_target_growth,
            )
            print(f"复用全池评分：{len(scored)} 只，选入优化 {len(codes)} 只（主动基金+风格ETF锚点）。")
        elif args.all_eligible and args.pit_root:
            asof = args.universe_asof or args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
            store = PITDataStore(args.pit_root)
            scored = score_universe(
                provider, start=args.start, end=args.end, pit_store=store, universe_asof=asof,
            )
            scored_for_selection = scored
            if args.score_out:
                scored.to_csv(args.score_out, index=False, encoding="utf-8-sig")
                print(f"全池评分已写入：{args.score_out}")
            inferred_target_growth = args.target_growth
            if inferred_target_growth is None:
                inferred_target_growth = resolve_target_growth(provider, val_provider, args.start, args.end)
            codes = select_full_universe_candidates(
                scored, config.OPTIMIZER.n_candidates, target_growth=inferred_target_growth,
            )
            print(f"全池筛选：严格合规池评分 {len(scored)} 只，选入优化 {len(codes)} 只（主动基金+风格ETF锚点）。")
        else:
            print("请用 --codes 指定候选基金，或使用 --pit-root ... --all-eligible 对全池评分后选基。")
            return 2

    if strict_eligible_codes is not None:
        ineligible = [code for code in codes if str(code) not in strict_eligible_codes]
        if ineligible:
            p.error("以下产品不满足严格范围或当前开放申购条件，不能进入目标组合：" + ", ".join(ineligible))

    # 对真实ETF（不是研究用合成ETF）施加可交易容量上限。PIT行情缺失时上限为零，
    # 使优化器选择主动基金或明确报告风格不可达，而不是输出不可执行的ETF仓位。
    max_weight_by_code = None
    if args.portfolio_aum_yi is not None and args.pit_root:
        if args.portfolio_aum_yi <= 0:
            p.error("--portfolio-aum-yi 必须为正数。")
        if not 0 < args.max_order_to_fund_aum <= 1:
            p.error("--max-order-to-fund-aum 必须在 (0, 1] 内。")
        etf_codes = [code for code in codes if str(asset_metadata.get(str(code), {}).get("asset_type", "")).lower() == "etf"
                     or str(asset_metadata.get(str(code), {}).get("is_stock_etf", "")).lower() in ("1", "true", "yes", "y", "是")]
        active_fund_codes = [code for code in codes if code not in set(etf_codes)]
        fund_caps = active_fund_capacity_weight_caps(
            active_fund_codes, asset_metadata, portfolio_aum_yi=args.portfolio_aum_yi,
            max_order_to_fund_aum=args.max_order_to_fund_aum,
        )
        market_asof = args.capacity_asof or asof
        frames = [PITDataStore(args.pit_root).read_market_asof(code, market_asof) for code in etf_codes]
        market = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        etf_caps = etf_capacity_weight_caps(
            market, etf_codes, portfolio_aum_yi=args.portfolio_aum_yi,
            participation_rate=args.etf_participation_rate, lookback=args.etf_adv_lookback,
        )
        max_weight_by_code = pd.concat([fund_caps, etf_caps])
        if active_fund_codes:
            tightest = ", ".join(f"{code}≤{cap:.1%}" for code, cap in fund_caps.sort_values().head(5).items())
            print(f"主动基金容量上限（订单/基金规模≤{args.max_order_to_fund_aum:.0%}）：最紧约束 {tightest}")
        if etf_codes:
            caps = ", ".join(f"{code}≤{cap:.1%}" for code, cap in etf_caps.items())
            print(f"ETF单日容量上限（{args.portfolio_aum_yi:.2f}亿元、参与率{args.etf_participation_rate:.0%}、{args.etf_adv_lookback}日ADV）：{caps}")
    elif args.portfolio_aum_yi is not None and args.source != "mock":
        if not args.pit_root:
            p.error("--portfolio-aum-yi 需要与--pit-root联用，才能审计ETF成交额。")

    # 目标成长暴露
    if args.target_growth is not None:
        target_g = args.target_growth
        src = "手动指定"
    elif inferred_target_growth is not None:
        target_g = inferred_target_growth
        src = "第③层自动"
    else:
        target_g = resolve_target_growth(provider, val_provider, args.start, args.end)
        src = "第③层自动"
    print(f"\n目标成长暴露 W*_growth = {target_g:.1%}（{src}）"
          f"  → 价值暴露 {1-target_g:.1%}")

    res, detail = build_portfolio(provider, codes, target_g, args.start, args.end,
                                  asset_metadata=asset_metadata,
                                  max_weight_by_code=max_weight_by_code)

    print("\n=== 第④层 · 组合优化结果 ===")
    print(f"  求解状态：{'成功' if res.success else '失败'}  {res.message}")
    print(f"  组合实际成长暴露：{res.realized_growth_exposure:.1%}"
          f"（目标 {res.target_growth_exposure:.1%}）")
    print(f"  期望选基alpha(年化)：{res.exp_active_return:+.2%}")
    print(f"  特质跟踪误差(年化)：{res.idio_te:.2%}")
    print(f"  ETF合计：{res.etf_weight:.1%} | 持有主动基金 {res.n_holdings} 只")
    print(f"  有效持仓数：{res.diagnostics['effective_holdings']:.1f}"
          f" | HHI {res.diagnostics['hhi']:.3f}")

    risk_contrib = res.diagnostics["risk_contributions"]
    print("\n  特质TE风险贡献（前五）：")
    for code, contribution in risk_contrib.head(5).items():
        print(f"    {code:<14}{contribution:+.2%}")
    for alert in res.diagnostics["alerts"]:
        print(f"  ⚠ {alert}")
    if not res.diagnostics["style_feasible"]:
        print("  ⚠ 当前候选池与权重上限无法命中风格目标："
              f"可达成长区间 {res.diagnostics['growth_reachable_min']:.1%}"
              f"~{res.diagnostics['growth_reachable_max']:.1%}。")

    print("\n  目标权重：")
    show = detail.copy()
    show["weight"] = (show["weight"] * 100).round(1).astype(str) + "%"
    print(show.to_string(index=False))

    if args.portfolio_out:
        detail.to_csv(args.portfolio_out, index=False, encoding="utf-8-sig")
        print(f"\n组合明细已写入：{args.portfolio_out}")
    if args.summary_out:
        summary = {
            "universe_asof": asof if args.pit_root else None,
            "return_end": args.end or pd.Timestamp.today().strftime("%Y-%m-%d"),
            "target_growth_exposure": res.target_growth_exposure,
            "realized_growth_exposure": res.realized_growth_exposure,
            "exp_active_return": res.exp_active_return,
            "idio_te": res.idio_te,
            "etf_weight": res.etf_weight,
            "n_holdings": res.n_holdings,
            "success": res.success,
            "message": res.message,
            "implementation": {
                "starting_weights": {},
                "initial_build": True,
                "note": "本次运行按当前无持仓生成首次建仓目标权重。",
            },
            "capacity": {
                "portfolio_aum_yi": args.portfolio_aum_yi,
                "max_order_to_fund_aum": args.max_order_to_fund_aum if args.portfolio_aum_yi is not None else None,
                "participation_rate": args.etf_participation_rate if args.portfolio_aum_yi is not None else None,
                "adv_lookback": args.etf_adv_lookback if args.portfolio_aum_yi is not None else None,
                "asof": (args.capacity_asof or asof) if args.portfolio_aum_yi is not None else None,
                "max_weight_by_code": ({str(code): float(cap) for code, cap in max_weight_by_code.items()}
                                       if max_weight_by_code is not None else {}),
            },
            "diagnostics": {
                key: value for key, value in res.diagnostics.items()
                if key != "risk_contributions"
            },
        }
        with open(args.summary_out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
        print(f"组合摘要已写入：{args.summary_out}")

    if args.backup_count < 0:
        p.error("--backup-count 不能为负数。")
    if args.backup_count and scored_for_selection is not None:
        held_codes = set(detail.loc[detail["weight"] > 1e-4, "code"].astype(str))
        backup_codes = select_backup_candidates(
            scored_for_selection, held_codes, n_backups=args.backup_count, target_growth=target_g,
        )
        backup = scored_for_selection.set_index(scored_for_selection["code"].astype(str)).loc[backup_codes].copy()
        if "size_yi" in backup.columns:
            backup = backup.rename(columns={"size_yi": "aum_yi"})
        backup["subscription_status"] = [asset_metadata.get(code, {}).get("subscription_status", "") for code in backup_codes]
        backup["redemption_status"] = [asset_metadata.get(code, {}).get("redemption_status", "") for code in backup_codes]
        columns = [column for column in (
            "code", "name", "composite_score", "growth_load", "style_r2", "style_alpha_ann",
            "info_ratio", "aum_yi", "subscription_status", "redemption_status",
        ) if column in backup.columns]
        backup = backup.loc[:, columns].reset_index(drop=True)
        print("\n  开放申购备选（未持有）：")
        print(backup.to_string(index=False))
        if args.backup_out:
            backup.to_csv(args.backup_out, index=False, encoding="utf-8-sig")
            print(f"备选基金已写入：{args.backup_out}")
    elif args.backup_out:
        p.error("--backup-out 需要配合 --scores-in 或 --all-eligible。")

    print("\n  说明：风格目标以强惩罚逼近，且会单独报告是否在约束下可达；"
          "\n        优化在此基础上最大化alpha×R²并向0收缩后的选基alpha，同时压低特质风险；"
          "\n        PIT实盘路径只使用候选池中的真实ETF，绝不输出占位ETF。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
