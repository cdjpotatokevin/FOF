"""端到端打分流程：数据 -> 风格基准 -> RBSA -> 指标 -> 综合分。

对外主入口 score_universe()，CLI 与脚本都调它。
"""
from __future__ import annotations
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .data import get_provider, DataProvider
from .data.pit import PITDataStore
from .engine.rbsa import run_rbsa
from .engine.metrics import compute_metrics, FundMetrics
from .engine.scoring import score_funds

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
    skip_rolling: bool = False,
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
        skip_rolling=skip_rolling,
    )


def _lite_prescreen_score(row: dict) -> float:
    """粗筛排序：风格调整后 alpha × R² + IR，用于缩小全池精评范围。"""
    alpha = pd.to_numeric(row.get("style_alpha_ann"), errors="coerce")
    r2 = pd.to_numeric(row.get("style_r2"), errors="coerce")
    ir = pd.to_numeric(row.get("info_ratio"), errors="coerce")
    if not np.isfinite(alpha):
        alpha = 0.0
    if not np.isfinite(r2):
        r2 = 0.0
    if not np.isfinite(ir):
        ir = 0.0
    return float(alpha * np.clip(r2, 0.0, 1.0) * 0.65 + ir * 0.35)


def _score_record_task(task: dict[str, Any]) -> dict | None:
    """进程池 worker：在子进程内重建 provider 并评价单只基金。"""
    provider = get_provider(task["provider_name"], **task.get("provider_kwargs", {}))
    m = score_one_fund(
        provider,
        task["code"],
        task.get("name", ""),
        task["factor_df"],
        task["bench"],
        task["start"],
        task["end"],
        pit_meta=task.get("pit_meta"),
        skip_rolling=task.get("skip_rolling", False),
    )
    if m is None:
        return None
    d = m.as_dict()
    for key in ("fund_type", "asset_type", "is_stock_etf", "is_qdii"):
        if key in task:
            d[key] = task[key]
    return d


def _score_records(
    provider: DataProvider,
    records: list[dict],
    *,
    records_are_pit: bool,
    factor_df: pd.DataFrame,
    bench: pd.Series,
    start: str,
    end: str,
    skip_rolling: bool,
    workers: int,
    provider_name: str,
    provider_kwargs: dict | None,
) -> list[dict]:
    tasks: list[dict[str, Any]] = []
    for rec in records:
        use_pit_meta = records_are_pit or bool(rec.get("_pit_meta"))
        tasks.append({
            "provider_name": provider_name,
            "provider_kwargs": provider_kwargs or {},
            "code": rec["code"],
            "name": rec.get("name", ""),
            "factor_df": factor_df,
            "bench": bench,
            "start": start,
            "end": end,
            "pit_meta": rec if use_pit_meta else None,
            "skip_rolling": skip_rolling,
            "fund_type": rec.get("fund_type", ""),
            "asset_type": rec.get("asset_type", "fund"),
            "is_stock_etf": rec.get("is_stock_etf", False),
            "is_qdii": rec.get("is_qdii", False),
        })

    rows: list[dict] = []
    total = len(tasks)
    if workers > 1 and total > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_score_record_task, task): i for i, task in enumerate(tasks, 1)}
            done = 0
            for future in as_completed(futures):
                done += 1
                result = future.result()
                if result is not None:
                    rows.append(result)
                if done % 25 == 0 or done == total:
                    log.info("已处理 %d/%d", done, total)
    else:
        for i, task in enumerate(tasks, 1):
            result = _score_record_task(task)
            if result is not None:
                rows.append(result)
            if i % 25 == 0:
                log.info("已处理 %d/%d", i, total)
    return rows


def score_universe(
    provider: DataProvider,
    codes: list[str] | None = None,
    start: str = "2019-01-01",
    end: str = "",
    group_by_type: bool = True,
    limit: int | None = None,
    pit_store: PITDataStore | None = None,
    universe_asof: str | None = None,
    strict_eligibility: bool | None = None,
    skip_rolling: bool = False,
    preselect_pool: int | None = None,
    workers: int = 1,
    provider_name: str | None = None,
    provider_kwargs: dict | None = None,
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
        use_strict = pit_store is not None if strict_eligibility is None else strict_eligibility
        if use_strict and pit_store is not None:
            from .engine.manager_tenure import filter_strict_universe
            uni = filter_strict_universe(
                uni, asof=pit_asof,
                cache_dir=pit_store.root / "raw" / "eastmoney",
            )
        else:
            uni = filter_universe(
                uni, asof=pit_asof if pit_store is not None else None,
                strict_eligibility=use_strict,
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

    pname = provider_name or _provider_label(provider)
    rows = _score_records(
        provider, records, records_are_pit=records_are_pit,
        factor_df=factor_df, bench=bench, start=start, end=end,
        skip_rolling=skip_rolling, workers=workers,
        provider_name=pname, provider_kwargs=provider_kwargs,
    )

    if not rows:
        raise RuntimeError("没有任何基金通过评价（检查数据源/代码/窗口）。")

    metrics_df = pd.DataFrame(rows)
    if preselect_pool and len(metrics_df) > preselect_pool:
        metrics_df["_lite_score"] = metrics_df.apply(_lite_prescreen_score, axis=1)
        metrics_df = metrics_df.sort_values(
            ["_lite_score", "style_alpha_ann", "info_ratio", "code"],
            ascending=[False, False, False, True],
        ).head(preselect_pool).drop(columns="_lite_score")
        log.info("粗筛保留 %d / %d 只进入综合打分", len(metrics_df), len(rows))

    group_col = "fund_type" if (group_by_type and metrics_df["fund_type"].nunique() > 1) else None
    scored = score_funds(metrics_df, group_col=group_col)
    return scored


def _provider_label(provider: DataProvider) -> str:
    mapping = {
        "MockProvider": "mock",
        "AkshareProvider": "akshare",
        "IFinDProvider": "ifind",
        "IFindHTTPProvider": "ifind_http",
    }
    return mapping.get(type(provider).__name__, "akshare")
