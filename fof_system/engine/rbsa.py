"""RBSA —— Sharpe 收益法风格分析 (Returns-Based Style Analysis)。

思想（Sharpe 1992）：用基金自身的收益序列，反推它“等价于”多少成长 + 多少价值。
求解带约束的回归：

    min_w  Var( r_fund - Σ_i w_i * r_style_i )
    s.t.   w_i >= 0 ,  Σ_i w_i = 1

w 即风格载荷（如 成长 0.7 / 价值 0.3）。约束保证结果像“真实可投的风格组合”，
不会出现负权重或杠杆，解释性强。

风格调整后 alpha（“选股超额”）：
    style_replica_t = Σ_i w_i * r_style_i,t          # 用纯风格指数复制出的影子组合
    active_t        = r_fund_t - style_replica_t      # 扣掉风格后的主动收益
    alpha_ann       = mean(active_t) * periods_per_year

这才是 FOF 真正想要的——剥离了成长/价值 beta 之后，基金经理还剩多少超额。
绝对收益排名混入了风格运气，会把“风格踩对”误判成“经理厉害”。
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class RBSAResult:
    weights: dict[str, float]          # 风格载荷，和≈1
    style_r2: float                    # 拟合优度（风格能解释多少基金波动）
    active_returns: pd.Series          # 逐期主动收益 = 基金 - 风格复制
    alpha_per_period: float            # 每期平均主动收益
    n_obs: int
    factor_names: list[str] = field(default_factory=list)

    def alpha_annual(self, periods_per_year: int) -> float:
        return self.alpha_per_period * periods_per_year


def solve_style_weights(
    fund_ret: np.ndarray,
    factor_ret: np.ndarray,
) -> np.ndarray:
    """求解约束风格权重。fund_ret: (T,)  factor_ret: (T, K)  返回 w: (K,)。

    目标：最小化主动收益方差（等价于 Sharpe 的最小化跟踪方差）。
    约束：w>=0，Σw=1。用 SLSQP；并给一个解析初值（等权）。
    """
    T, K = factor_ret.shape

    def obj(w):
        active = fund_ret - factor_ret @ w
        return np.var(active, ddof=0)

    def grad(w):
        active = fund_ret - factor_ret @ w
        # d/dw Var = -2/T * Xᵀ(active - mean(active))
        ac = active - active.mean()
        return -2.0 / T * (factor_ret.T @ ac)

    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0,
             "jac": lambda w: np.ones(K)},)
    bounds = [(0.0, 1.0)] * K
    w0 = np.full(K, 1.0 / K)
    res = minimize(obj, w0, jac=grad, bounds=bounds, constraints=cons,
                   method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else w0


def run_rbsa(
    fund_ret: pd.Series,
    factor_rets: pd.DataFrame,
    rf_per_period: float = 0.0,
) -> RBSAResult:
    """对单只基金做一次（全窗口）RBSA。

    fund_ret    : 基金期收益 Series
    factor_rets : 各风格因子期收益 DataFrame（列为因子名，如 growth/value）
    rf_per_period: 每期无风险利率（默认 0，即用原始收益做风格分解）
    """
    df = pd.concat([fund_ret.rename("fund"), factor_rets], axis=1, join="inner").dropna()
    if len(df) < factor_rets.shape[1] + 2:
        raise ValueError("有效样本过少，无法做 RBSA")

    factor_names = list(factor_rets.columns)
    y = (df["fund"] - rf_per_period).to_numpy()
    X = (df[factor_names] - rf_per_period).to_numpy()

    w = solve_style_weights(y, X)
    replica = X @ w
    active = y - replica

    # 风格 R²：1 - Var(主动) / Var(基金)
    var_fund = np.var(df["fund"].to_numpy(), ddof=0)
    style_r2 = 1.0 - np.var(active, ddof=0) / var_fund if var_fund > 0 else 0.0

    active_s = pd.Series(active, index=df.index, name="active")
    return RBSAResult(
        weights={n: float(wi) for n, wi in zip(factor_names, w)},
        style_r2=float(style_r2),
        active_returns=active_s,
        alpha_per_period=float(active_s.mean()),
        n_obs=len(df),
        factor_names=factor_names,
    )


def rolling_style_weights(
    fund_ret: pd.Series,
    factor_rets: pd.DataFrame,
    window: int,
    rf_per_period: float = 0.0,
) -> pd.DataFrame:
    """滚动 RBSA：观察风格漂移。返回 index=日期, 列=各因子载荷 + alpha_per_period。"""
    df = pd.concat([fund_ret.rename("fund"), factor_rets], axis=1, join="inner").dropna()
    factor_names = list(factor_rets.columns)
    rows = {}
    for end in range(window, len(df) + 1):
        sub = df.iloc[end - window:end]
        y = (sub["fund"] - rf_per_period).to_numpy()
        X = (sub[factor_names] - rf_per_period).to_numpy()
        w = solve_style_weights(y, X)
        active = y - X @ w
        rec = {n: wi for n, wi in zip(factor_names, w)}
        rec["alpha_per_period"] = active.mean()
        rows[df.index[end - 1]] = rec
    return pd.DataFrame.from_dict(rows, orient="index")
