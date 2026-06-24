"""ETF流动性与组合容量压力：基于成交额和参与率的保守上限。"""
from __future__ import annotations

import pandas as pd


def etf_capacity_report(
    market: pd.DataFrame,
    weights: pd.Series,
    *,
    participation_rate: float = 0.10,
    lookback: int = 20,
    portfolio_aum_yi: float | None = None,
) -> pd.DataFrame:
    """计算ETF交易容量。

    ``amount`` 以元计，``weight`` 与 ``portfolio_aum_yi`` 以组合权重/亿元计。
    未提供组合规模时，输出每只ETF约束下可承载的最大组合规模（亿元）。
    """
    if not 0 < participation_rate <= 1:
        raise ValueError("participation_rate 必须在 (0, 1] 内")
    if lookback < 1:
        raise ValueError("lookback 必须至少为1")
    frame = market.copy()
    frame["code"] = frame["code"].astype(str)
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    frame["trading_date"] = pd.to_datetime(frame["trading_date"], errors="coerce")
    rows = []
    for code, weight in weights.items():
        weight = float(weight)
        if weight <= 0:
            continue
        series = frame[(frame["code"] == str(code)) & frame["amount"].gt(0)].sort_values("trading_date").tail(lookback)
        if series.empty:
            rows.append({"code": str(code), "weight": weight, "status": "missing_amount"})
            continue
        adv_yi = float(series["amount"].mean() / 1e8)
        max_order_yi = adv_yi * participation_rate
        max_portfolio_yi = max_order_yi / weight
        row = {
            "code": str(code), "weight": weight, "adv_yi": adv_yi,
            "participation_rate": participation_rate,
            "max_order_yi": max_order_yi,
            "max_portfolio_yi": max_portfolio_yi,
            "observations": len(series), "status": "ok",
        }
        if portfolio_aum_yi is not None:
            order_yi = portfolio_aum_yi * weight
            row.update({
                "portfolio_aum_yi": portfolio_aum_yi,
                "order_yi": order_yi,
                "order_participation": order_yi / adv_yi,
                "within_limit": order_yi <= max_order_yi,
            })
        rows.append(row)
    return pd.DataFrame(rows).set_index("code") if rows else pd.DataFrame()


def etf_capacity_weight_caps(
    market: pd.DataFrame,
    etf_codes: list[str],
    *,
    portfolio_aum_yi: float,
    participation_rate: float = 0.10,
    lookback: int = 20,
) -> pd.Series:
    """按成交额给出 ETF 的单日可交易权重上限。

    上限为 ``参与率 × 近20日平均成交额 / FOF规模``。缺少可用成交额的 ETF
    返回 0：在没有流动性证据时，组合构建不能把它当作可交易的风格补全工具。
    """
    if portfolio_aum_yi <= 0:
        raise ValueError("portfolio_aum_yi 必须为正数")
    codes = [str(code) for code in etf_codes]
    if not codes:
        return pd.Series(dtype=float)
    caps = pd.Series(0.0, index=codes, dtype=float)
    if market.empty or not {"code", "trading_date", "amount"}.issubset(market.columns):
        return caps
    # 以 100% 假设权重复用统一的 ADV / 参与率计算；随后把单日允许订单额
    # 除以组合规模，得到与优化器同量纲的权重上限。
    report = etf_capacity_report(
        market, pd.Series(1.0, index=codes),
        participation_rate=participation_rate, lookback=lookback,
    )
    if report.empty:
        return caps
    available = report[report["status"].eq("ok")]
    caps.loc[available.index.astype(str)] = (
        pd.to_numeric(available["max_order_yi"], errors="coerce").fillna(0.0)
        / float(portfolio_aum_yi)
    )
    return caps.clip(lower=0.0, upper=1.0)
