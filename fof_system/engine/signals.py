"""风格择时信号库。

约定：每个信号返回一个 daily 的"观点" Series，view ∈ [-1, 1]：
    view > 0  → 偏成长（超配 国证成长）
    view < 0  → 偏价值（超配 国证价值）
所有信号**因果计算**（t 时刻只用 ≤t 的数据），避免前视偏差。
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _causal_z(s: pd.Series, window: int) -> pd.Series:
    """滚动 z-score（因果）：用过去 window 期的均值/标准差标准化当前值。"""
    mu = s.rolling(window, min_periods=max(20, window // 4)).mean()
    sd = s.rolling(window, min_periods=max(20, window // 4)).std(ddof=0)
    z = (s - mu) / sd.replace(0, np.nan)
    return z


def momentum_view(
    growth_close: pd.Series,
    value_close: pd.Series,
    lookback: int,
    z_window: int,
) -> pd.Series:
    """风格动量（趋势跟随）：近 lookback 期成长相对价值的累计超额。

    成长近期更强 → 顺势偏成长（view>0）。用 tanh 把 z 压到 [-1,1]。
    """
    df = pd.concat([growth_close.rename("g"), value_close.rename("v")],
                   axis=1, join="inner").sort_index().dropna()
    rel = df["g"] / df["g"].shift(lookback) - df["v"] / df["v"].shift(lookback)
    z = _causal_z(rel, z_window)
    return np.tanh(z).rename("momentum")


def vol_regime_view(
    growth_close: pd.Series,
    value_close: pd.Series,
    vol_window: int,
    z_window: int,
    growth_weight: float = 0.5,
) -> pd.Series:
    """波动 regime（逆向）：混合指数已实现波动偏高=风险厌恶，偏防御的价值（view<0）。"""
    df = pd.concat([growth_close.rename("g"), value_close.rename("v")],
                   axis=1, join="inner").sort_index().dropna()
    blended_ret = (growth_weight * df["g"].pct_change()
                   + (1 - growth_weight) * df["v"].pct_change())
    vol = blended_ret.rolling(vol_window, min_periods=vol_window // 2).std(ddof=0)
    z = _causal_z(vol, z_window)
    return (-np.tanh(z)).rename("vol_regime")


def valuation_spread_view(
    growth_val: pd.Series,
    value_val: pd.Series,
    z_window: int,
) -> pd.Series:
    """估值价差（均值回归，逆向）：成长相对价值的估值比 = PE_g / PE_v。

    该比值相对自身历史偏高 → 成长太贵 → 逆向偏价值（view<0）；反之偏成长。
    """
    df = pd.concat([growth_val.rename("g"), value_val.rename("v")],
                   axis=1, join="inner").sort_index().dropna()
    df = df[(df["g"] > 0) & (df["v"] > 0)]
    ratio = np.log(df["g"] / df["v"])     # 取对数更对称
    z = _causal_z(ratio, z_window)
    return (-np.tanh(z)).rename("valuation")
