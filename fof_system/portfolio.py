"""第④层编排：把 ②选基 + ③风格目标 + ④优化 串成一套组合构建流程。

流程：
  1. 候选基金（外部传入，或第②层综合分 top-N）
  2. 对每只候选跑 RBSA → 风格载荷 + 风格调整后 alpha + R² + 主动收益序列
  3. 构建期望 alpha 向量（alpha×R²后收缩）与主动收益协方差（收缩）
  4. 取第③层当期风格目标 W*_growth（或外部指定）
  5. 加入风格 ETF 补全资产，求解优化器 → 基金/ETF 目标权重
"""
from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Mapping, Any
import numpy as np
import pandas as pd

from . import config
from .data import get_provider, DataProvider
from .engine.rbsa import run_rbsa
from .engine import risk_model
from .engine.optimizer import optimize_portfolio, OptResult
from .pipeline import build_benchmark

log = logging.getLogger("fof.portfolio")


@dataclass
class FundInput:
    code: str
    name: str
    growth_load: float
    value_load: float
    alpha_ann: float
    style_r2: float
    n_obs: int
    asset_type: str = "fund"


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "y", "是")


def _asset_type(meta: Mapping[str, Any]) -> str:
    if str(meta.get("asset_type", "")).lower() == "etf" or _truthy(meta.get("is_stock_etf")):
        return "etf"
    return "fund"


def r2_adjusted_alpha(alpha_ann: pd.Series, style_r2: pd.Series) -> pd.Series:
    """按RBSA解释度折减风格调整后alpha。

    受约束RBSA的R²可能为负；投资含义上不能让负R²反向放大alpha，因此先截断到
    [0, 1]，再计算 ``alpha × R²``。
    """
    alpha = pd.to_numeric(alpha_ann, errors="coerce").fillna(0.0)
    confidence = pd.to_numeric(style_r2, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return alpha * confidence


def select_full_universe_candidates(scored: pd.DataFrame,
                                    n_active: int = config.OPTIMIZER.n_candidates,
                                    target_growth: float | None = None,
                                    style_complement_candidates: int = config.OPTIMIZER.style_complement_candidates,
                                    style_complement_gap: float = config.OPTIMIZER.style_complement_gap) -> list[str]:
    """从全体已评分的严格合规池中选优化候选，不按AUM预先截断。

    主动基金按综合评分选取；若提供当期成长目标，则至少加入若干高分、成长载荷显著
    低于目标的主动基金，避免候选风格拥挤并被迫用低流动性ETF补全。ETF不以历史
    alpha排名，而从全体合格ETF里选出最接近纯价值和纯成长的两个风格锚点。所有步骤
    都基于全池评分，不把规模作为预先筛选门槛。
    """
    if scored.empty:
        raise ValueError("全池评分为空，无法生成优化候选。")
    asset_type = scored.get("asset_type", pd.Series("fund", index=scored.index)).astype(str).str.lower()
    active = scored.loc[asset_type.ne("etf")].copy()
    if active.empty:
        raise ValueError("全池评分中没有主动基金候选。")
    required = ["composite_score", "style_alpha_ann", "info_ratio", "code"]
    missing = [column for column in required if column not in active]
    if missing:
        raise ValueError(f"全池评分缺少主动基金筛选字段: {missing}")
    ranked_active = active.sort_values(["composite_score", "style_alpha_ann", "info_ratio", "code"],
                                       ascending=[False, False, False, True])
    active_codes: list[str] = []
    if target_growth is not None and "growth_load" in ranked_active:
        growth = pd.to_numeric(ranked_active["growth_load"], errors="coerce")
        threshold = max(0.0, float(target_growth) - style_complement_gap)
        complements = ranked_active.loc[growth <= threshold].head(min(style_complement_candidates, n_active))
        active_codes.extend(complements["code"].astype(str).tolist())
    for code in ranked_active["code"].astype(str):
        if code not in active_codes:
            active_codes.append(code)
        if len(active_codes) >= n_active:
            break

    etfs = scored.loc[asset_type.eq("etf")].copy()
    if etfs.empty or "growth_load" not in etfs:
        return active_codes
    etfs["_r2"] = pd.to_numeric(etfs.get("style_r2", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    etfs["_growth"] = pd.to_numeric(etfs["growth_load"], errors="coerce")
    anchors: list[str] = []
    for target in (0.0, 1.0):
        ranked = etfs.assign(_distance=(etfs["_growth"] - target).abs()).sort_values(
            ["_distance", "_r2", "code"], ascending=[True, False, True],
        )
        if not ranked.empty:
            code = str(ranked.iloc[0]["code"])
            if code not in anchors:
                anchors.append(code)
    return active_codes + anchors


def select_backup_candidates(
    scored: pd.DataFrame,
    excluded_codes: set[str] | list[str],
    n_backups: int = 5,
    target_growth: float | None = None,
    style_complement_gap: float = config.OPTIMIZER.style_complement_gap,
) -> list[str]:
    """从同一严格合规评分池给出未持有的主动基金备选。

    备选优先保留目标成长载荷另一侧的产品，避免主组合中低成长/价值暴露基金临时
    不可申购时，只剩同风格基金可以替换。调用方须先按最新PIT开放申购状态过滤。
    """
    if n_backups <= 0:
        return []
    asset_type = scored.get("asset_type", pd.Series("fund", index=scored.index)).astype(str).str.lower()
    active = scored.loc[asset_type.ne("etf")].copy()
    excluded = {str(code) for code in excluded_codes}
    active = active.loc[~active["code"].astype(str).isin(excluded)]
    required = ["composite_score", "style_alpha_ann", "info_ratio", "code"]
    missing = [column for column in required if column not in active]
    if missing:
        raise ValueError(f"备选池缺少字段: {missing}")
    ranked = active.sort_values(["composite_score", "style_alpha_ann", "info_ratio", "code"],
                                ascending=[False, False, False, True])
    codes: list[str] = []
    if target_growth is not None and "growth_load" in ranked:
        growth = pd.to_numeric(ranked["growth_load"], errors="coerce")
        low_side = ranked.loc[growth <= max(0.0, float(target_growth) - style_complement_gap)]
        high_side = ranked.loc[growth > max(0.0, float(target_growth) - style_complement_gap)]
        low_target = (n_backups + 1) // 2
        for frame, take in ((low_side, low_target), (high_side, n_backups - low_target)):
            for code in frame["code"].astype(str).head(take):
                if code not in codes:
                    codes.append(code)
    for code in ranked["code"].astype(str):
        if code not in codes:
            codes.append(code)
        if len(codes) >= n_backups:
            break
    return codes[:n_backups]


def compute_fund_inputs(
    provider: DataProvider,
    codes: list[str],
    factor_df: pd.DataFrame,
    start: str,
    end: str,
    asset_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[FundInput], pd.DataFrame]:
    """对候选基金逐个 RBSA，返回 (输入列表, 主动收益对齐表)。"""
    freq = config.RETURN_FREQ
    ppy = config.PERIODS_PER_YEAR[freq]
    rf = config.RISK_FREE_ANNUAL / ppy

    asset_metadata = asset_metadata or {}
    inputs: list[FundInput] = []
    active_map: dict[str, pd.Series] = {}
    # PIT全池路径已有名称，避免为了补名称额外拉取一次供应商全市场清单。
    if all(str(asset_metadata.get(str(code), {}).get("name", "")).strip() for code in codes):
        name_map: dict[str, str] = {}
    else:
        try:
            name_map = provider.list_funds().set_index("code")["name"].to_dict()
        except Exception:  # noqa: BLE001
            name_map = {}

    for code in codes:
        meta = asset_metadata.get(str(code), {})
        if _truthy(meta.get("is_qdii")):
            log.warning("跳过QDII产品 %s", code)
            continue
        try:
            nav = provider.get_fund_nav(code, start, end)
            fr = provider.to_returns(nav, freq)
            aligned = pd.concat([fr, factor_df], axis=1, join="inner").dropna()
            if len(aligned) < config.MIN_OBS:
                log.info("跳过 %s：样本不足", code)
                continue
            aligned = aligned.iloc[-config.EVAL_WINDOW:]
            rb = run_rbsa(aligned.iloc[:, 0], aligned[factor_df.columns], rf_per_period=rf)
            inputs.append(FundInput(
                code=code, name=str(meta.get("name") or name_map.get(code, "")),
                growth_load=rb.weights.get("growth", float("nan")),
                value_load=rb.weights.get("value", float("nan")),
                alpha_ann=rb.alpha_annual(ppy), style_r2=rb.style_r2, n_obs=rb.n_obs,
                asset_type=_asset_type(meta),
            ))
            active_map[code] = rb.active_returns
        except Exception as e:  # noqa: BLE001
            log.warning("跳过 %s：%s", code, e)

    active_df = pd.DataFrame(active_map)
    return inputs, active_df


def build_portfolio(
    provider: DataProvider,
    codes: list[str],
    target_growth: float,
    start: str = "2019-01-01",
    end: str = "",
    cfg: config.OptimizerConfig = config.OPTIMIZER,
    asset_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    max_weight_by_code: Mapping[str, float] | None = None,
) -> tuple[OptResult, pd.DataFrame]:
    """构建组合。target_growth 为第③层目标成长暴露（0~1）。

    返回 (优化结果, 资产输入明细表)。
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    factor_df, _ = build_benchmark(provider, start, end)

    inputs, active_df = compute_fund_inputs(
        provider, codes, factor_df, start, end, asset_metadata=asset_metadata,
    )
    if len(inputs) < 2:
        raise RuntimeError("有效候选基金不足 2 只，无法优化。")

    fund_codes = [fi.code for fi in inputs]
    ppy = config.PERIODS_PER_YEAR[config.RETURN_FREQ]

    # 优化器的期望alpha = 风格调整后alpha × RBSA R²，再向0收缩。
    # 这会降低低解释度风格回归中“偶然残差”对权重的影响。
    alpha_raw = pd.Series({fi.code: fi.alpha_ann for fi in inputs})
    style_r2 = pd.Series({fi.code: fi.style_r2 for fi in inputs})
    alpha_r2 = r2_adjusted_alpha(alpha_raw, style_r2)
    alpha = risk_model.shrink_alpha(alpha_r2, cfg.alpha_shrink)

    # 主动收益协方差（收缩，年化）
    cov = risk_model.active_cov(active_df[fund_codes], shrink=cfg.cov_shrink, ppy=ppy)

    growth_load = pd.Series({fi.code: fi.growth_load for fi in inputs})
    actual_etf_codes = [fi.code for fi in inputs if fi.asset_type == "etf"]
    # 实物ETF是风格暴露工具，不把历史RBSA残差解释为可复制的选基alpha；但保留其
    # 实际主动收益协方差，用于估计相对成长/价值基准的跟踪风险。
    alpha.loc[actual_etf_codes] = 0.0

    # 合成风格ETF仅用于mock/研究验证。实盘路径必须使用候选池中的真实ETF，不能把
    # ETF_GROWTH/ETF_VALUE 这类占位资产输出为调仓指令。
    etf_codes: list[str] = list(actual_etf_codes)
    synthetic_etf_codes: list[str] = []
    use_synthetic_style_etf = cfg.use_style_etf and asset_metadata is None
    if use_synthetic_style_etf:
        for ecode, meta in config.STYLE_ETF_ASSETS.items():
            synthetic_etf_codes.append(ecode)
            etf_codes.append(ecode)
            alpha[ecode] = 0.0
            growth_load[ecode] = meta["growth_load"]
        cov = risk_model.add_etf_assets(cov, synthetic_etf_codes)

    res = optimize_portfolio(
        alpha=alpha, cov=cov, growth_load=growth_load,
        target_growth=target_growth, etf_codes=etf_codes,
        max_weight_fund=cfg.max_weight_fund, min_weight_fund=cfg.min_weight_fund,
        risk_aversion=cfg.risk_aversion, etf_total_cap=cfg.etf_total_cap,
        te_budget_annual=cfg.te_budget_annual,
        max_weight_by_code=max_weight_by_code,
    )
    report = risk_model.ex_ante_risk_report(
        weights=res.weights, alpha=alpha, cov=cov, growth_load=growth_load,
        etf_codes=etf_codes, target_growth=target_growth,
        benchmark_growth=config.BENCHMARK_WEIGHTS["growth"],
        te_budget=cfg.te_budget_annual,
    )
    res.diagnostics.update(report.summary())
    res.diagnostics["risk_contributions"] = report.risk_contributions
    res.diagnostics["alerts"] = report.alerts

    # 明细表
    name_map = {fi.code: fi.name for fi in inputs}
    name_map.update({e: config.STYLE_ETF_ASSETS[e]["name"] for e in synthetic_etf_codes})
    metadata = asset_metadata or {}
    aum_map = {str(code): meta.get("aum_yi", float("nan")) for code, meta in metadata.items()}
    subscription_map = {str(code): meta.get("subscription_status", "") for code, meta in metadata.items()}
    redemption_map = {str(code): meta.get("redemption_status", "") for code, meta in metadata.items()}

    def _aum(value: Any) -> float:
        try:
            return round(float(value), 4)
        except (TypeError, ValueError):
            return float("nan")
    gl_map = growth_load.to_dict()
    al_map = alpha.to_dict()
    raw_alpha_map = alpha_raw.to_dict()
    r2_map = style_r2.clip(0.0, 1.0).to_dict()
    r2_alpha_map = alpha_r2.to_dict()
    rows = []
    for code, wt in res.weights.items():
        rows.append({
            "code": code, "name": name_map.get(code, ""), "weight": wt,
            "growth_load": round(gl_map.get(code, float("nan")), 3),
            "style_r2": round(r2_map.get(code, 0.0), 4),
            "alpha_ann(原始)": round(raw_alpha_map.get(code, 0.0), 4),
            "alpha_ann(R²调整后)": round(r2_alpha_map.get(code, 0.0), 4),
            "alpha_ann(收缩后)": round(al_map.get(code, 0.0), 4),
            "aum_yi": _aum(aum_map.get(code, float("nan"))),
            "subscription_status": subscription_map.get(code, ""),
            "redemption_status": redemption_map.get(code, ""),
            "type": (
                "ETF补全" if code in set(synthetic_etf_codes)
                else "ETF" if code in set(actual_etf_codes)
                else "主动基金"
            ),
        })
    detail = pd.DataFrame(rows)
    return res, detail


def resolve_target_growth(
    provider: DataProvider,
    val_provider: DataProvider | None,
    start: str,
    end: str,
) -> float:
    """调用第③层得到当期目标成长暴露。"""
    from .engine.style_timing import StyleTimer
    timer = StyleTimer(provider, valuation_provider=val_provider)
    view = timer.current_view(start, end)
    return view.w_growth
