"""端到端打分流程：数据 -> 风格基准 -> RBSA -> 指标 -> 综合分。

对外主入口 score_universe()，CLI 与脚本都调它。
"""
from __future__ import annotations
import logging
import pandas as pd

from . import config
from .data import get_provider, DataProvider
from .data.pit import PITDataStore
from .engine.rbsa import run_rbsa
from .engine.metrics import compute_metrics, FundMetrics
from .engine.scoring import score_funds, explain_top

log = logging.getLogger("fof")


def _years_between(a, b) -> float:
    try:
        return max(0.0, (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25)
    except Exception:  # noqa: BLE001
        return float("nan")


def build_benchmark(provider: DataProvider, start: str, end: str) -> tuple[pd.DataFrame, pd.Series]:
    """构造风格因子收益表和配置的考核基准收益序列。"""
    freq = config.RETURN_FREQ
    factor_rets = {}
    for fac, code in config.STYLE_FACTORS.items():
        close = provider.get_index_close(code, start, end)
        factor_rets[fac] = provider.to_returns(close, freq)
    factor_df = pd.DataFrame(factor_rets).dropna()

    w = config.BENCHMARK_WEIGHTS
    bench = sum(factor_df[f] * w[f] for f in factor_df.columns)
    bench.name = "benchmark"
    return factor_df, bench


def score_one_fund(
    provider: DataProvider,
    code: str,
    name: str,
    factor_df: pd.DataFrame,
    bench: pd.Series,
    start: str,
    end: str,
    pit_meta: dict | None = None,
) -> FundMetrics | None:
    freq = config.RETURN_FREQ
    ppy = config.PERIODS_PER_YEAR[freq]
    rf_per_period = config.RISK_FREE_ANNUAL / ppy

    try:
        nav = provider.get_fund_nav(code, start, end)
    except Exception as e:  # noqa: BLE001
        log.warning("跳过 %s：净值获取失败 %s", code, e)
        return None
    fund_ret = provider.to_returns(nav, freq)

    # 对齐到风格因子样本，取末端 EVAL_WINDOW 期
    aligned = pd.concat([fund_ret, factor_df], axis=1, join="inner").dropna()
    if len(aligned) < config.MIN_OBS:
        log.info("跳过 %s：有效期数 %d < %d", code, len(aligned), config.MIN_OBS)
        return None
    aligned = aligned.iloc[-config.EVAL_WINDOW:]
    fr = aligned.iloc[:, 0]
    fac = aligned[factor_df.columns]
    bench_win = bench.reindex(aligned.index).dropna()

    try:
        rbsa = run_rbsa(fr, fac, rf_per_period=rf_per_period)
    except Exception as e:  # noqa: BLE001
        log.warning("跳过 %s：RBSA 失败 %s", code, e)
        return None

    # 元数据必须与研究时点对齐：提供PIT记录时宁可留空，也不回退到今天的供应商快照。
    if pit_meta is None:
        meta = provider.get_fund_meta(code)
        metric_name = name or meta.name
        size_yi = meta.size_yi
        tenure = _years_between(meta.manager_start, end) if meta.manager_start else float("nan")
    else:
        metric_name = name or str(pit_meta.get("name", ""))
        size_raw = pd.to_numeric(pit_meta.get("aum_yi"), errors="coerce")
        size_yi = float(size_raw) if pd.notna(size_raw) else float("nan")
        manager_start = pit_meta.get("manager_start")
        tenure = _years_between(manager_start, end) if pd.notna(manager_start) else float("nan")

    return compute_metrics(
        code=code, name=metric_name,
        fund_ret=fr, factor_rets=fac, rbsa=rbsa, bench_ret=bench_win,
        ppy=ppy, rolling_window=config.ROLLING_WINDOW,
        bench_growth_weight=config.BENCHMARK_WEIGHTS["growth"],
        size_yi=size_yi, size_sweet=config.SIZE_SWEET_SPOT_YI,
        tenure_years=tenure,
    )


def score_universe(
    provider: DataProvider,
    codes: list[str] | None = None,
    start: str = "2019-01-01",
    end: str = "",
    group_by_type: bool = True,
    limit: int | None = None,
    pit_store: PITDataStore | None = None,
    universe_asof: str | None = None,
) -> pd.DataFrame:
    """对一批基金打分并排名。

    codes : 指定基金代码列表；None 时用 PIT 快照（若提供）或 provider.list_funds() 初筛。
    返回：打分明细表（已按综合分降序），含 composite_score / rank。
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    factor_df, bench = build_benchmark(provider, start, end)

    if codes is None:
        from .engine.universe import filter_universe
        if pit_store is not None:
            pit_asof = universe_asof or end
            uni = pit_store.read_universe_asof(pit_asof, asset_types=("fund", "etf"))
            if uni.empty:
                raise RuntimeError(f"PIT 数据仓在 {pit_asof} 没有可用的基金/ETF快照。")
            if "is_qdii" in uni:
                qdii = uni["is_qdii"].astype(str).str.lower().isin(("1", "true", "yes"))
                uni = uni[~qdii]
        else:
            uni = provider.list_funds()
        uni = filter_universe(
            uni, asof=pit_asof if pit_store is not None else None,
            strict_eligibility=pit_store is not None,
        )
        records = uni.to_dict("records")
        records_are_pit = pit_store is not None
    else:
        # 显式代码同样优先使用PIT元数据：否则ETF会在后续优化中被误作主动基金，
        # 且AUM/经理任期会退回到今天的供应商快照。
        pit_records: dict[str, dict] = {}
        if pit_store is not None:
            pit_asof = universe_asof or end
            pit_universe = pit_store.read_universe_asof(pit_asof, asset_types=("fund", "etf"))
            pit_records = {
                str(row["code"]): row.to_dict()
                for _, row in pit_universe.iterrows()
            }
        # 尝试用清单补名字
        try:
            name_map = provider.list_funds().set_index("code")["name"].to_dict()
        except Exception:  # noqa: BLE001
            name_map = {}
        records = []
        for code in codes:
            rec = dict(pit_records.get(str(code), {}))
            rec["code"] = str(code)
            rec.setdefault("name", name_map.get(code, ""))
            rec["_pit_meta"] = str(code) in pit_records
            records.append(rec)
        records_are_pit = False

    if limit:
        records = records[:limit]

    rows: list[dict] = []
    for i, rec in enumerate(records, 1):
        use_pit_meta = records_are_pit or bool(rec.get("_pit_meta"))
        m = score_one_fund(provider, rec["code"], rec.get("name", ""),
                           factor_df, bench, start, end,
                           pit_meta=rec if use_pit_meta else None)
        if m is not None:
            d = m.as_dict()
            d["fund_type"] = rec.get("fund_type", "")
            d["asset_type"] = rec.get("asset_type", "fund")
            d["is_stock_etf"] = rec.get("is_stock_etf", False)
            d["is_qdii"] = rec.get("is_qdii", False)
            rows.append(d)
        if i % 25 == 0:
            log.info("已处理 %d/%d", i, len(records))

    if not rows:
        raise RuntimeError("没有任何基金通过评价（检查数据源/代码/窗口）。")

    metrics_df = pd.DataFrame(rows)
    group_col = "fund_type" if (group_by_type and metrics_df["fund_type"].nunique() > 1) else None
    scored = score_funds(metrics_df, group_col=group_col)
    return scored
