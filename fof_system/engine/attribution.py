"""第⑤层：风险与归因（收益法风格/选股归因）。

为什么不用经典 Brinson 分桶：FOF 持有的是基金（成长/价值的混合体），无法干净地
把持仓归入"成长桶/价值桶"。改用 RBSA 载荷做收益法分解，每期超额**恒等**地拆成：

    R_p,t − R_b,t = 风格择时_t + 选股_t
    风格择时_t = (W^p_g − W^b_g)·R_g,t + (W^p_v − W^b_v)·R_v,t    （第③层的功劳）
    选股_t     = Σ_i x_i · (r_i,t − [g_i·R_g,t + v_i·R_v,t])       （第②层的功劳）

其中 W^p_g = Σ_i x_i·g_i 为组合的成长暴露。该式由构造恒等，无"说不清"的残差。
（按固定权重/逐期再平衡假设；权重漂移与载荷时变的影响体现在跨期几何链接残差里。）

进一步可按行业归因——需 iFinD 持仓穿透到个股 + 行业收益，作为后续扩展，本层先给
风格/选股这层最关键的拆解。
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .. import config


@dataclass
class AttributionResult:
    periods: pd.DataFrame            # 逐期：style_timing / selection / excess / Rp / Rb
    fund_selection: pd.Series        # 各基金累计选股贡献（按窗口求和）
    # 累计（几何）
    cum_excess: float                # 组合相对基准的累计超额（几何）
    cum_port: float
    cum_bench: float
    # 拆解（算术求和）与年化
    total_style_timing: float
    total_selection: float
    linking_residual: float          # 几何累计 − (风格择时+选股算术和)
    years: float
    ann_excess: float
    ann_style_timing: float
    ann_selection: float
    avg_growth_exposure: float       # 窗口内组合平均成长暴露
    diagnostics: dict = field(default_factory=dict)


def style_selection_attribution(
    weights: pd.Series,              # 资产权重（基金+ETF），index=代码
    returns: pd.DataFrame,           # 各资产期收益，列=代码，行=时间
    growth_load: pd.Series,          # 各资产成长载荷
    value_load: pd.Series,           # 各资产价值载荷
    style_returns: pd.DataFrame,     # 含 'growth','value' 两列的期收益
    bench_growth_weight: float | None = None,
    ppy: int = 52,
) -> AttributionResult:
    if bench_growth_weight is None:
        bench_growth_weight = config.BENCHMARK_WEIGHTS["growth"]
    codes = [c for c in weights.index if c in returns.columns]
    w = weights.reindex(codes).fillna(0.0)
    R = returns[codes].dropna(how="all")
    Rg = style_returns["growth"].reindex(R.index)
    Rv = style_returns["value"].reindex(R.index)
    df = pd.concat([R, Rg.rename("_g"), Rv.rename("_v")], axis=1).dropna()
    R = df[codes]
    Rg, Rv = df["_g"], df["_v"]

    gl = growth_load.reindex(codes).fillna(0.5)
    vl = value_load.reindex(codes).fillna(0.5)

    # 每资产风格复制收益 与 主动收益
    style_rep = pd.DataFrame({c: gl[c] * Rg + vl[c] * Rv for c in codes})
    active = R - style_rep

    # 组合层
    Wp_g = float((w * gl).sum())
    Wp_v = float((w * vl).sum())
    Rp = (R * w).sum(axis=1)
    Rb = bench_growth_weight * Rg + (1 - bench_growth_weight) * Rv
    style_timing = (Wp_g - bench_growth_weight) * Rg + (Wp_v - (1 - bench_growth_weight)) * Rv
    selection = (active * w).sum(axis=1)
    excess = Rp - Rb

    periods = pd.DataFrame({
        "Rp": Rp, "Rb": Rb, "excess": excess,
        "style_timing": style_timing, "selection": selection,
    })

    # 逐基金累计选股贡献
    fund_sel = (active * w).sum(axis=0).reindex(codes)

    # 累计（几何）与链接
    cum_port = float((1 + Rp).prod() - 1)
    cum_bench = float((1 + Rb).prod() - 1)
    cum_excess = cum_port - cum_bench
    tot_st = float(style_timing.sum())
    tot_sel = float(selection.sum())
    linking = cum_excess - (tot_st + tot_sel)

    years = max(len(periods) / ppy, 1e-9)
    return AttributionResult(
        periods=periods, fund_selection=fund_sel.sort_values(ascending=False),
        cum_excess=cum_excess, cum_port=cum_port, cum_bench=cum_bench,
        total_style_timing=tot_st, total_selection=tot_sel, linking_residual=linking,
        years=years, ann_excess=cum_excess / years,
        ann_style_timing=tot_st / years, ann_selection=tot_sel / years,
        avg_growth_exposure=Wp_g,
        diagnostics={"n_periods": len(periods), "n_assets": len(codes)},
    )


def synth_etf_returns(style_returns: pd.DataFrame, etf_growth_load: dict[str, float]) -> pd.DataFrame:
    """为风格 ETF 补全资产合成收益：纯成长ETF=成长指数收益，纯价值ETF=价值指数收益。"""
    cols = {}
    for code, gload in etf_growth_load.items():
        cols[code] = gload * style_returns["growth"] + (1 - gload) * style_returns["value"]
    return pd.DataFrame(cols)
