"""生产与回测共享的单期调仓编排。

``rebalance_at`` 与 ``run_portfolio --all-eligible`` 使用同一选基、容量与优化路径，
避免回测与实盘逻辑漂移。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from . import config
from .data.base import DataProvider
from .data.pit import PITDataStore
from .engine.capacity import active_fund_capacity_weight_caps, etf_capacity_weight_caps
from .engine.optimizer import OptResult
from .engine.universe import filter_universe
from .pipeline import score_universe
from .portfolio import (
    build_portfolio,
    resolve_target_growth,
    select_full_universe_candidates,
)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "y", "是")


def _is_etf(meta: Mapping[str, Any]) -> bool:
    return str(meta.get("asset_type", "")).lower() == "etf" or _truthy(meta.get("is_stock_etf"))


def pit_asset_metadata(pit_store: PITDataStore, asof: str | pd.Timestamp) -> dict[str, dict]:
    universe = pit_store.read_universe_asof(asof, asset_types=("fund", "etf"))
    return {str(row["code"]): row.to_dict() for _, row in universe.iterrows()}


def pit_snapshot_label(pit_store: PITDataStore | None, asof: str | pd.Timestamp) -> str:
    """PIT 主数据快照的 available_at 标签，用于评分缓存键。"""
    if pit_store is None:
        return pd.Timestamp(asof).strftime("%Y-%m-%d")
    universe = pit_store.read_universe_asof(asof, asset_types=("fund", "etf"))
    if universe.empty or "available_at" not in universe.columns:
        return pd.Timestamp(asof).strftime("%Y-%m-%d")
    return pd.to_datetime(universe["available_at"]).max().strftime("%Y-%m-%d")


def collect_pit_codes(
    pit_store: PITDataStore,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    strict_eligibility: bool = False,
    freq: str = "QE",
) -> list[str]:
    """回测预取：在区间内各调仓日 PIT 可见代码的并集（默认季度末）。"""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    period_ends = pd.date_range(start_ts, end_ts, freq=freq)
    codes: set[str] = set()
    for point in period_ends:
        universe = pit_store.read_universe_asof(point, asset_types=("fund", "etf"))
        if universe.empty:
            continue
        if "is_qdii" in universe.columns:
            qdii = universe["is_qdii"].astype(str).str.lower().isin(("1", "true", "yes"))
            universe = universe[~qdii]
        if strict_eligibility:
            from .engine.manager_tenure import filter_strict_universe
            filtered = filter_strict_universe(
                universe, asof=point, cache_dir=pit_store.root / "raw" / "eastmoney",
            )
        else:
            filtered = filter_universe(universe, asof=point, strict_eligibility=False)
        codes.update(filtered["code"].astype(str).tolist())
    return sorted(codes)


def compute_capacity_caps(
    codes: list[str],
    asset_metadata: Mapping[str, Mapping[str, Any]],
    *,
    pit_store: PITDataStore | None,
    capacity_asof: str,
    portfolio_aum_yi: float,
    max_order_to_fund_aum: float,
    etf_participation_rate: float,
    etf_adv_lookback: int,
    enforce_active_fund_capacity: bool,
    enforce_etf_capacity: bool,
) -> pd.Series | None:
    """计算逐代码权重上限；未启用的容量维度返回 None。"""
    if portfolio_aum_yi <= 0:
        raise ValueError("portfolio_aum_yi 必须为正数")
    if not enforce_active_fund_capacity and not enforce_etf_capacity:
        return None

    etf_codes = [code for code in codes if _is_etf(asset_metadata.get(str(code), {}))]
    active_codes = [code for code in codes if code not in set(etf_codes)]
    pieces: list[pd.Series] = []

    if enforce_active_fund_capacity and active_codes:
        pieces.append(active_fund_capacity_weight_caps(
            active_codes, asset_metadata,
            portfolio_aum_yi=portfolio_aum_yi,
            max_order_to_fund_aum=max_order_to_fund_aum,
        ))
    if enforce_etf_capacity and etf_codes and pit_store is not None:
        frames = [pit_store.read_market_asof(code, capacity_asof) for code in etf_codes]
        market = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        pieces.append(etf_capacity_weight_caps(
            market, etf_codes, portfolio_aum_yi=portfolio_aum_yi,
            participation_rate=etf_participation_rate, lookback=etf_adv_lookback,
        ))
    if not pieces:
        return None
    return pd.concat(pieces)


def score_cache_path(
    cache_dir: str | Path | None,
    asof: str | pd.Timestamp,
    pit_store: PITDataStore | None = None,
    *,
    skip_rolling: bool = False,
    preselect_pool: int | None = None,
) -> Path | None:
    if not cache_dir:
        return None
    pit_label = pit_snapshot_label(pit_store, asof)
    end = pd.Timestamp(asof).date()
    mode = "full"
    if skip_rolling or preselect_pool:
        parts = []
        if skip_rolling:
            parts.append("noroll")
        if preselect_pool:
            parts.append(f"p{preselect_pool}")
        mode = "_".join(parts)
    return Path(cache_dir).expanduser() / f"pit={pit_label}" / f"end={end}" / f"scores_{mode}.csv"


def load_or_score_universe(
    provider: DataProvider,
    *,
    eval_start: str,
    asof: str | pd.Timestamp,
    pit_store: PITDataStore | None,
    strict_eligibility: bool,
    score_cache_dir: str | Path | None,
    scores_in: str | Path | None = None,
    skip_rolling: bool = False,
    preselect_pool: int | None = None,
    workers: int = 1,
    provider_kwargs: dict | None = None,
) -> pd.DataFrame:
    if scores_in:
        frame = pd.read_csv(scores_in, dtype={"code": "string"})
        if pit_store is not None:
            from .engine.universe import filter_universe
            pit_asof = pd.Timestamp(asof).strftime("%Y-%m-%d")
            uni = pit_store.read_universe_asof(pit_asof, asset_types=("fund", "etf"))
            if not uni.empty:
                if "is_qdii" in uni.columns:
                    qdii = uni["is_qdii"].astype(str).str.lower().isin(("1", "true", "yes"))
                    uni = uni[~qdii]
                if strict_eligibility:
                    from .engine.manager_tenure import filter_strict_universe
                    uni = filter_strict_universe(
                        uni, asof=pit_asof, cache_dir=pit_store.root / "raw" / "eastmoney",
                    )
                else:
                    uni = filter_universe(uni, asof=pit_asof, strict_eligibility=False)
                allowed = set(uni["code"].astype(str))
                frame = frame.loc[frame["code"].astype(str).isin(allowed)].copy()
        return frame
    cache_file = score_cache_path(
        score_cache_dir, asof, pit_store,
        skip_rolling=skip_rolling, preselect_pool=preselect_pool,
    )
    if cache_file is not None and cache_file.exists():
        return pd.read_csv(cache_file, dtype={"code": "string"})
    asof_str = pd.Timestamp(asof).strftime("%Y-%m-%d")
    scored = score_universe(
        provider,
        codes=None,
        start=eval_start,
        end=asof_str,
        pit_store=pit_store,
        universe_asof=asof_str,
        strict_eligibility=strict_eligibility,
        skip_rolling=skip_rolling,
        preselect_pool=preselect_pool,
        workers=workers,
        provider_kwargs=provider_kwargs,
    )
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(cache_file, index=False, encoding="utf-8-sig")
    return scored


@dataclass
class RebalanceResult:
    asof: pd.Timestamp
    success: bool
    weights: pd.Series | None = None
    target_growth: float | None = None
    detail: pd.DataFrame | None = None
    opt_result: OptResult | None = None
    scored_count: int = 0
    candidate_count: int = 0
    skip_reason: str = ""
    etf_codes: set[str] = field(default_factory=set)
    synthetic_etf_codes: set[str] = field(default_factory=set)
    scored: pd.DataFrame | None = None


def rebalance_at(
    provider: DataProvider,
    *,
    asof: str | pd.Timestamp,
    eval_start: str,
    pit_store: PITDataStore | None = None,
    val_provider: DataProvider | None = None,
    target_growth: float | None = None,
    opt_cfg: config.OptimizerConfig = config.OPTIMIZER,
    strict_eligibility: bool = False,
    portfolio_aum_yi: float | None = config.PORTFOLIO_AUM_YI,
    max_order_to_fund_aum: float = config.OPTIMIZER.max_order_to_fund_aum,
    etf_participation_rate: float = 0.10,
    etf_adv_lookback: int = 20,
    enforce_active_fund_capacity: bool = False,
    enforce_etf_capacity: bool = False,
    score_cache_dir: str | Path | None = None,
    capacity_asof: str | None = None,
    scores_in: str | Path | None = None,
    skip_rolling: bool = False,
    preselect_pool: int | None = None,
    workers: int = 1,
    provider_kwargs: dict | None = None,
) -> RebalanceResult:
    """在 ``asof`` 时点按生产路径构建目标组合（仅用 ≤asof 的数据）。"""
    point = pd.Timestamp(asof)
    asof_str = point.strftime("%Y-%m-%d")
    asset_metadata: dict[str, dict] | None = (
        pit_asset_metadata(pit_store, point) if pit_store is not None else None
    )

    try:
        scored = load_or_score_universe(
            provider,
            eval_start=eval_start,
            asof=point,
            pit_store=pit_store,
            strict_eligibility=strict_eligibility,
            score_cache_dir=score_cache_dir,
            scores_in=scores_in,
            skip_rolling=skip_rolling,
            preselect_pool=preselect_pool,
            workers=workers,
            provider_kwargs=provider_kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        return RebalanceResult(asof=point, success=False, skip_reason=f"score_failed:{exc}")

    if scored.empty:
        return RebalanceResult(asof=point, success=False, skip_reason="empty_score")

    target_g = target_growth
    if target_g is None:
        target_g = resolve_target_growth(provider, val_provider, eval_start, asof_str)

    try:
        codes = select_full_universe_candidates(
            scored, n_active=opt_cfg.n_candidates, target_growth=target_g,
        )
    except Exception as exc:  # noqa: BLE001
        return RebalanceResult(
            asof=point, success=False, scored_count=len(scored), skip_reason=f"candidate_failed:{exc}",
        )

    if not codes:
        return RebalanceResult(
            asof=point, success=False, scored_count=len(scored), skip_reason="no_candidates",
        )

    meta_for_caps: Mapping[str, Mapping[str, Any]] = asset_metadata or {}
    max_weight_by_code = None
    if portfolio_aum_yi is not None and (
        enforce_active_fund_capacity or enforce_etf_capacity
    ):
        try:
            max_weight_by_code = compute_capacity_caps(
                codes, meta_for_caps,
                pit_store=pit_store,
                capacity_asof=capacity_asof or asof_str,
                portfolio_aum_yi=portfolio_aum_yi,
                max_order_to_fund_aum=max_order_to_fund_aum,
                etf_participation_rate=etf_participation_rate,
                etf_adv_lookback=etf_adv_lookback,
                enforce_active_fund_capacity=enforce_active_fund_capacity,
                enforce_etf_capacity=enforce_etf_capacity,
            )
        except Exception as exc:  # noqa: BLE001
            return RebalanceResult(
                asof=point, success=False, scored_count=len(scored),
                candidate_count=len(codes), skip_reason=f"capacity_failed:{exc}",
            )

    try:
        opt_result, detail = build_portfolio(
            provider, codes, target_g, eval_start, asof_str,
            cfg=opt_cfg, asset_metadata=asset_metadata,
            max_weight_by_code=max_weight_by_code,
        )
    except Exception as exc:  # noqa: BLE001
        return RebalanceResult(
            asof=point, success=False, scored_count=len(scored),
            candidate_count=len(codes), skip_reason=f"optimize_failed:{exc}",
        )

    if not opt_result.success:
        return RebalanceResult(
            asof=point, success=False, scored_count=len(scored),
            candidate_count=len(codes), skip_reason=f"optimizer:{opt_result.message}",
            opt_result=opt_result, detail=detail, scored=scored,
        )

    etf_codes = set(detail.loc[detail["type"].isin(("ETF", "ETF补全")), "code"].astype(str))
    synthetic_etf_codes = set(detail.loc[detail["type"] == "ETF补全", "code"].astype(str))
    return RebalanceResult(
        asof=point,
        success=True,
        weights=opt_result.weights,
        target_growth=target_g,
        detail=detail,
        opt_result=opt_result,
        scored_count=len(scored),
        candidate_count=len(codes),
        etf_codes=etf_codes - synthetic_etf_codes,
        synthetic_etf_codes=synthetic_etf_codes,
        scored=scored,
    )
