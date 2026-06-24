"""单基金评价指标。

绝大多数指标都建立在 RBSA 的"主动收益"序列 active_t = 基金 - 风格复制 之上，
因为 FOF 要的是"剥离成长/价值 beta 后"的经理能力，而不是风格踩对的运气。
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from .rbsa import RBSAResult, rolling_style_weights


@dataclass
class FundMetrics:
    code: str
    name: str = ""
    # 风格
    growth_load: float = float("nan")
    value_load: float = float("nan")
    style_tilt: float = float("nan")       # 成长-价值，>0 偏成长
    style_vs_bench: float = float("nan")   # 相对考核基准的成长净偏离
    style_r2: float = float("nan")
    # 主动收益（核心）
    style_alpha_ann: float = float("nan")  # 风格调整后年化 alpha
    info_ratio: float = float("nan")       # IR = alpha / 主动波动
    excess_win_rate: float = float("nan")  # 主动收益为正的期数占比
    alpha_consistency: float = float("nan")# 滚动窗口 alpha>0 占比
    # 风险
    max_drawdown: float = float("nan")
    calmar: float = float("nan")
    ann_return: float = float("nan")
    ann_vol: float = float("nan")
    # 相对考核基准
    excess_vs_bench_ann: float = float("nan")  # 基金 - 考核基准 年化超额
    te_vs_bench_ann: float = float("nan")      # 相对考核基准的年化跟踪误差
    # 元数据派生
    size_yi: float = float("nan")
    size_score: float = float("nan")
    tenure_years: float = float("nan")
    n_obs: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def max_drawdown_from_returns(ret: pd.Series) -> float:
    """由收益序列算最大回撤（正数，0.2 = 回撤20%）。"""
    nav = (1 + ret).cumprod()
    peak = nav.cummax()
    dd = (nav - peak) / peak
    return float(-dd.min()) if len(dd) else float("nan")


def ann_stats(ret: pd.Series, ppy: int) -> tuple[float, float]:
    """年化收益与年化波动。"""
    mu = ret.mean() * ppy
    vol = ret.std(ddof=1) * np.sqrt(ppy)
    return float(mu), float(vol)


def compute_metrics(
    code: str,
    name: str,
    fund_ret: pd.Series,
    factor_rets: pd.DataFrame,
    rbsa: RBSAResult,
    bench_ret: pd.Series,
    ppy: int,
    rolling_window: int,
    bench_growth_weight: float = 0.7,
    size_yi: float = float("nan"),
    size_sweet: tuple[float, float] = (2.0, 80.0),
    tenure_years: float = float("nan"),
) -> FundMetrics:
    """汇总单只基金全部指标。"""
    active = rbsa.active_returns

    m = FundMetrics(code=code, name=name, n_obs=rbsa.n_obs)

    # --- 风格 ---
    m.growth_load = rbsa.weights.get("growth", float("nan"))
    m.value_load = rbsa.weights.get("value", float("nan"))
    m.style_tilt = m.growth_load - m.value_load
    m.style_vs_bench = m.growth_load - bench_growth_weight
    m.style_r2 = rbsa.style_r2

    # --- 主动收益核心 ---
    m.style_alpha_ann = rbsa.alpha_annual(ppy)
    active_vol = active.std(ddof=1) * np.sqrt(ppy)
    m.info_ratio = m.style_alpha_ann / active_vol if active_vol > 0 else float("nan")
    m.excess_win_rate = float((active > 0).mean())

    # alpha 一致性：滚动窗口里 alpha 为正的比例（稳定性，不是一次性运气）
    try:
        roll = rolling_style_weights(fund_ret, factor_rets, rolling_window)
        if len(roll):
            m.alpha_consistency = float((roll["alpha_per_period"] > 0).mean())
    except Exception:  # noqa: BLE001
        m.alpha_consistency = float("nan")

    # --- 风险 ---
    m.max_drawdown = max_drawdown_from_returns(fund_ret)
    m.ann_return, m.ann_vol = ann_stats(fund_ret, ppy)
    m.calmar = m.ann_return / m.max_drawdown if m.max_drawdown and m.max_drawdown > 0 else float("nan")

    # --- 相对考核基准 ---
    aligned = pd.concat([fund_ret.rename("f"), bench_ret.rename("b")], axis=1, join="inner").dropna()
    diff = aligned["f"] - aligned["b"]
    m.excess_vs_bench_ann = float(diff.mean() * ppy)
    m.te_vs_bench_ann = float(diff.std(ddof=1) * np.sqrt(ppy))

    # --- 规模适中度（落在甜点区间得分高，过大过小衰减）---
    m.size_yi = size_yi
    m.size_score = _size_score(size_yi, size_sweet)

    # --- 任期 ---
    m.tenure_years = tenure_years

    return m


def _size_score(size_yi: float, sweet: tuple[float, float]) -> float:
    """规模适中度 0~1：区间内=1，越偏离衰减。"""
    if not np.isfinite(size_yi):
        return float("nan")
    lo, hi = sweet
    if lo <= size_yi <= hi:
        return 1.0
    if size_yi < lo:
        return max(0.0, size_yi / lo)            # 太小线性衰减
    return max(0.0, hi / size_yi)                # 太大按反比衰减（容量约束）
