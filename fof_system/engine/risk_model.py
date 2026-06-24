"""第④层风险模型：基金"主动收益"协方差 + 期望 alpha 向量。

为什么用主动收益而不是总收益：组合的风格暴露由优化约束钉到第③层目标，
风格 beta 带来的风险是"有意为之"的；优化器真正要控制的是**剥离风格后的特质风险**
（选基选错的风险）。所以协方差建立在 RBSA 残差 active_t = 基金 - 风格复制 上。
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .rbsa import rolling_style_weights


@dataclass
class ExAnteRiskReport:
    """调仓前的特质风险、风格偏离和集中度快照。"""

    expected_alpha: float
    idio_te: float
    expected_ir: float
    growth_exposure: float
    active_growth_vs_benchmark: float
    active_growth_vs_target: float
    etf_weight: float
    hhi: float
    effective_holdings: float
    risk_contributions: pd.Series
    alerts: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        return {
            "ex_ante_alpha": self.expected_alpha,
            "ex_ante_idio_te": self.idio_te,
            "ex_ante_ir": self.expected_ir,
            "growth_exposure": self.growth_exposure,
            "active_growth_vs_benchmark": self.active_growth_vs_benchmark,
            "active_growth_vs_target": self.active_growth_vs_target,
            "etf_weight": self.etf_weight,
            "hhi": self.hhi,
            "effective_holdings": self.effective_holdings,
        }


def shrink_alpha(alpha: pd.Series, shrink: float) -> pd.Series:
    """期望 alpha 向 0 收缩：α_adj = (1-shrink)·α。

    历史 alpha 点估计噪声大（见第②层说明），收缩降低优化器对其过度反应。
    """
    return (1.0 - shrink) * alpha


def active_cov(active_returns: pd.DataFrame, shrink: float = 0.2,
               ppy: int = 52) -> pd.DataFrame:
    """由各基金主动收益序列构建年化协方差，并向对角阵收缩。

    active_returns : DataFrame，列=基金代码，行=对齐后的期主动收益
    shrink         : 0~1，Σ_shrunk = (1-s)·Σ_sample + s·diag(Σ_sample)
    返回：年化协方差 DataFrame（基金×基金）
    """
    df = active_returns.dropna(how="all")
    sample = df.cov()
    # 对角收缩（Ledoit-Wolf 的简化版：把非对角元朝 0 收缩）
    diag = np.diag(np.diag(sample.values))
    shrunk = (1.0 - shrink) * sample.values + shrink * diag
    cov = pd.DataFrame(shrunk, index=sample.index, columns=sample.columns) * ppy
    return cov


def add_etf_assets(cov: pd.DataFrame, etf_codes: list[str]) -> pd.DataFrame:
    """为风格 ETF 补全资产扩展协方差矩阵。

    纯风格 ETF 的"主动收益"（相对其自身风格）≈0，故特质方差≈0、与基金不相关。
    给一个极小对角值保证正定。
    """
    if not etf_codes:
        return cov
    all_codes = list(cov.columns) + etf_codes
    n = len(all_codes)
    big = np.zeros((n, n))
    k = len(cov.columns)
    big[:k, :k] = cov.values
    eps = 1e-8
    for i in range(k, n):
        big[i, i] = eps
    return pd.DataFrame(big, index=all_codes, columns=all_codes)


def ex_ante_risk_report(
    weights: pd.Series,
    alpha: pd.Series,
    cov: pd.DataFrame,
    growth_load: pd.Series,
    etf_codes: list[str],
    target_growth: float,
    benchmark_growth: float,
    te_budget: float | None = None,
    style_tolerance: float = 0.01,
) -> ExAnteRiskReport:
    """生成调仓前可用的特质TE、风格偏离与集中度报告。

    ``cov`` 必须由 RBSA 残差（而非基金总收益）估计；否则成长/价值 beta 会被
    重复计入主动风险。风险贡献的和等于组合年化特质 TE。
    """
    codes = weights.index
    w = weights.reindex(codes).fillna(0.0).astype(float)
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights 必须有正的合计权重")
    w = w / total

    a = alpha.reindex(codes).fillna(0.0)
    S = cov.reindex(index=codes, columns=codes).fillna(0.0).to_numpy()
    g = growth_load.reindex(codes).fillna(benchmark_growth)
    wv = w.to_numpy()
    marginal = S @ wv
    variance = float(wv @ marginal)
    te = float(np.sqrt(max(variance, 0.0)))
    contribution = pd.Series(
        wv * marginal / te if te > 0 else np.zeros(len(w)), index=codes,
    )

    growth = float(w @ g)
    etf_weight = float(w[w.index.isin(etf_codes)].sum())
    hhi = float((w ** 2).sum())
    effective = float(1 / hhi) if hhi > 0 else float("nan")
    expected_alpha = float(w @ a)
    expected_ir = expected_alpha / te if te > 0 else float("nan")
    active_target = growth - target_growth

    alerts: list[str] = []
    if abs(active_target) > style_tolerance:
        alerts.append(f"成长暴露偏离当期目标 {active_target:+.1%}，超过容忍度 {style_tolerance:.1%}。")
    if te_budget is not None and te > te_budget:
        alerts.append(f"事前特质TE {te:.2%} 超过预算 {te_budget:.2%}。")
    if hhi > 0.20:
        alerts.append(f"权重集中度 HHI={hhi:.3f}，有效持仓仅 {effective:.1f} 只。")

    return ExAnteRiskReport(
        expected_alpha=expected_alpha,
        idio_te=te,
        expected_ir=expected_ir,
        growth_exposure=growth,
        active_growth_vs_benchmark=growth - benchmark_growth,
        active_growth_vs_target=active_target,
        etf_weight=etf_weight,
        hhi=hhi,
        effective_holdings=effective,
        risk_contributions=contribution.sort_values(ascending=False),
        alerts=alerts,
    )


def style_drift_report(
    fund_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    window: int,
    threshold: float = 0.15,
    rf_per_period: float = 0.0,
) -> pd.DataFrame:
    """以滚动 RBSA 检测基金成长载荷相对自身历史中枢的漂移。

    调用方应先把输入截断至监控时点；函数本身不会读取未来数据。漂移阈值的
    默认值是成长载荷 15 个百分点，适合作为人工复核触发器而非自动卖出指令。
    """
    rows: list[dict] = []
    for code in fund_returns.columns:
        try:
            rolling = rolling_style_weights(
                fund_returns[code].dropna(), factor_returns, window, rf_per_period,
            )
        except Exception:  # noqa: BLE001
            continue
        if rolling.empty or "growth" not in rolling:
            continue
        latest = float(rolling["growth"].iloc[-1])
        median = float(rolling["growth"].median())
        drift = latest - median
        rows.append({
            "code": code,
            "latest_growth_load": latest,
            "historical_growth_median": median,
            "growth_drift": drift,
            "style_drift_alert": abs(drift) >= threshold,
            "n_rolling_obs": len(rolling),
        })
    if not rows:
        return pd.DataFrame(columns=[
            "latest_growth_load", "historical_growth_median", "growth_drift",
            "style_drift_alert", "n_rolling_obs",
        ])
    return pd.DataFrame(rows).set_index("code").sort_values(
        "growth_drift", key=np.abs, ascending=False,
    )
