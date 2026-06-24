"""风格择时回测：tilt 组合 vs 配置的成长/价值基准。

只回答一个问题：相对考核基准，按信号做有界成长/价值偏离，到底有没有
稳定的超额？输出年化超额、信息比率、月度胜率、换手、主动回撤。

防前视：月末 t 定的权重，从 t 的**下一交易日**起生效。
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from .. import config


@dataclass
class BacktestResult:
    ann_return_tilt: float
    ann_return_bench: float
    ann_excess: float
    tracking_error: float
    info_ratio: float
    hit_rate: float           # 调仓周期内跑赢基准的比例
    avg_active_growth: float  # 平均成长净偏离
    turnover_annual: float    # 年化单边换手
    active_max_drawdown: float
    n_periods: int

    def as_dict(self) -> dict:
        return asdict(self)


def run_style_backtest(
    growth_close: pd.Series,
    value_close: pd.Series,
    weight_df: pd.DataFrame,
    ppy: int = 252,
    rebalance: str = "M",
    bench_growth_weight: float | None = None,
) -> tuple[BacktestResult, pd.DataFrame]:
    """
    growth_close/value_close : 指数收盘价（日频）
    weight_df                : index=调仓日, 列含 w_growth/w_value
    返回：(汇总指标, 日频明细 DataFrame)
    """
    if bench_growth_weight is None:
        bench_growth_weight = config.BENCHMARK_WEIGHTS["growth"]
    px = pd.concat([growth_close.rename("g"), value_close.rename("v")],
                   axis=1, join="inner").sort_index().dropna()
    ret = px.pct_change().dropna()

    # 权重前视防护：t 日定的权重，t 之后生效 → shift 到下一交易日并前向填充
    wg = weight_df["w_growth"].reindex(ret.index, method=None)
    wg = wg.shift(1).reindex(ret.index).ffill()
    wg = wg.dropna()
    ret = ret.loc[wg.index]
    wv = 1 - wg

    tilt_ret = wg * ret["g"] + wv * ret["v"]
    bench_ret = bench_growth_weight * ret["g"] + (1 - bench_growth_weight) * ret["v"]
    active = tilt_ret - bench_ret

    # 年化
    ann_t = (1 + tilt_ret).prod() ** (ppy / len(tilt_ret)) - 1
    ann_b = (1 + bench_ret).prod() ** (ppy / len(bench_ret)) - 1
    te = active.std(ddof=1) * np.sqrt(ppy)
    ann_excess = active.mean() * ppy
    ir = ann_excess / te if te > 0 else float("nan")

    # 调仓周期胜率
    rule = {"M": "ME", "W": "W-FRI"}.get(rebalance, "ME")
    per_active = (1 + active).resample(rule).prod() - 1
    hit = float((per_active > 0).mean()) if len(per_active) else float("nan")

    # 换手（单边）：相邻调仓日 w_growth 变化绝对值之和，年化
    w_at_rebal = weight_df["w_growth"].dropna()
    turns = w_at_rebal.diff().abs().sum()
    years = max((ret.index[-1] - ret.index[0]).days / 365.25, 1e-9)
    turnover_annual = float(turns / years)

    # 主动回撤
    active_nav = (1 + active).cumprod()
    active_dd = float(-((active_nav - active_nav.cummax()) / active_nav.cummax()).min())

    detail = pd.DataFrame({
        "w_growth": wg, "tilt_ret": tilt_ret, "bench_ret": bench_ret,
        "active": active, "active_nav": active_nav,
    })

    res = BacktestResult(
        ann_return_tilt=float(ann_t), ann_return_bench=float(ann_b),
        ann_excess=float(ann_excess), tracking_error=float(te), info_ratio=float(ir),
        hit_rate=hit, avg_active_growth=float((wg - bench_growth_weight).mean()),
        turnover_annual=turnover_annual, active_max_drawdown=active_dd,
        n_periods=len(per_active),
    )
    return res, detail
