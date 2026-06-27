"""第④层组合优化器：特征 targeting 的均值-方差优化。

把第②层的"选基 alpha"和第③层的"风格目标"落到真实的基金/ETF 权重：

    max_x   αᵀx − γ · xᵀ Σ x
    s.t.    Σ x = 1                      （满仓）
            x ≥ 0                        （long-only，公募 FOF 不做空）
            x_i ≤ w_max                  （单票上限，分散）
            Σ x_i · growth_load_i ≈ W*g  （以强惩罚逼近第③层目标）
            Σ x_etf ≤ etf_cap            （限制 ETF 占比，优先用主动基金赚 alpha）
            [可选] xᵀΣx ≤ TE_budget²     （特质跟踪误差预算）

其中 α 只有主动基金有（ETF≈0），Σ 是**主动收益**协方差（特质风险）。
风格 beta 风险以目标暴露为中心管理；优化器控制的是特质风险，并显式报告目标是否可达。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize


@dataclass
class OptResult:
    weights: pd.Series                  # 最终权重（基金 + ETF，>0 的）
    exp_active_return: float            # 期望选基 alpha = αᵀx（年化）
    idio_te: float                      # 特质跟踪误差 = sqrt(xᵀΣx)（年化）
    realized_growth_exposure: float     # 组合实际成长暴露
    target_growth_exposure: float
    etf_weight: float                   # ETF 合计
    n_holdings: int
    success: bool
    message: str = ""
    diagnostics: dict = field(default_factory=dict)


def optimize_portfolio(
    alpha: pd.Series,                   # index=资产代码（基金+ETF），ETF 处≈0
    cov: pd.DataFrame,                  # 主动收益年化协方差（同 index）
    growth_load: pd.Series,             # 各资产成长载荷（ETF: 1或0）
    target_growth: float,               # 第③层目标成长暴露 W*_growth
    etf_codes: list[str],
    max_weight_fund: float = 0.15,
    min_weight_fund: float = 0.0,
    risk_aversion: float = 8.0,
    etf_total_cap: float = 0.40,
    te_budget_annual: float | None = None,
    max_weight_by_code: Mapping[str, float] | None = None,
    min_weight_by_code: Mapping[str, float] | None = None,
    weight_budget: float = 1.0,
    style_penalty: float = 5e3,
    style_tolerance: float = 1e-3,
) -> OptResult:
    codes = list(alpha.index)
    n = len(codes)
    a = alpha.reindex(codes).fillna(0.0).to_numpy()
    S = cov.reindex(index=codes, columns=codes).fillna(0.0).to_numpy()
    gl = growth_load.reindex(codes).fillna(0.5).to_numpy()
    is_etf = np.array([c in set(etf_codes) for c in codes])

    # 边界：基金 [min,max]，ETF [0, etf_cap]；容量或合规产生的逐代码上限可进一步收紧。
    cap_source = max_weight_by_code if max_weight_by_code is not None else {}
    floor_source = min_weight_by_code if min_weight_by_code is not None else {}
    code_caps = {str(code): float(cap) for code, cap in cap_source.items()}
    code_floors = {str(code): float(floor) for code, floor in floor_source.items()}
    bounds = []
    for i, code in enumerate(codes):
        lower, upper = (0.0, etf_total_cap) if is_etf[i] else (min_weight_fund, max_weight_fund)
        if str(code) in code_floors:
            lower = max(lower, max(0.0, code_floors[str(code)]))
        if str(code) in code_caps:
            upper = min(upper, max(0.0, code_caps[str(code)]))
        if upper < lower - 1e-12:
            raise ValueError(f"{code} 的逐代码上限低于最小持仓约束。")
        bounds.append((lower, upper))

    # 先用线性规划检查“约束本身”是否允许命中风格目标。风格目标采用软惩罚是为
    # 避免等式约束共线导致 SLSQP 失败，但不能因此把不可达的目标伪装成已命中。
    A_ub = [is_etf.astype(float)] if etf_codes else None
    b_ub = [etf_total_cap] if etf_codes else None
    budget = float(weight_budget)
    if not (0.0 < budget <= 1.0 + 1e-9):
        raise ValueError(f"weight_budget 须在 (0, 1] 内，收到 {budget}")
    lp_args = dict(A_ub=A_ub, b_ub=b_ub, A_eq=[np.ones(n)], b_eq=[budget], bounds=bounds, method="highs")
    # 先确认“满仓”本身可行。若主动基金数量不足以填满非ETF仓位，且ETF总上限又较低，
    # SLSQP 可能返回违反约束的失败解；此时绝不能把该权重展示为可交易组合。
    base_lp = linprog(np.zeros(n), **lp_args)
    if not base_lp.success:
        raise ValueError(
            "组合权重约束无法满仓：请增加主动基金候选数、提高单只基金上限或提高ETF总上限。"
        )
    min_lp = linprog(gl, **lp_args)
    max_lp = linprog(-gl, **lp_args)
    growth_min = float(min_lp.fun) if min_lp.success else float("nan")
    growth_max = float(-max_lp.fun) if max_lp.success else float("nan")
    style_feasible = bool(min_lp.success and max_lp.success
                          and growth_min - style_tolerance <= target_growth <= growth_max + style_tolerance)

    # 风格暴露用"二次惩罚"而非硬等式：避免当载荷近似相等时该约束与"和为1"
    # 共线导致约束 Jacobian 秩亏、SLSQP 求解失败；同时对不可达目标可优雅地尽量逼近。
    # 目标：min  -αᵀx + γ·xᵀΣx + ρ·(glᵀx - target)²
    def neg_util(x):
        gap = gl @ x - target_growth
        return -(a @ x) + risk_aversion * (x @ S @ x) + style_penalty * gap * gap

    def neg_util_grad(x):
        gap = gl @ x - target_growth
        return -a + 2.0 * risk_aversion * (S @ x) + 2.0 * style_penalty * gap * gl

    # 仅保留"和为1"硬等式，杜绝共线
    cons = [
        {"type": "eq", "fun": lambda x: x.sum() - budget, "jac": lambda x: np.ones(n)},
    ]
    if etf_codes:
        cons.append({"type": "ineq",  # etf_cap - Σx_etf ≥ 0
                     "fun": lambda x: etf_total_cap - x[is_etf].sum(),
                     "jac": lambda x: -is_etf.astype(float)})
    if te_budget_annual is not None:
        te2 = te_budget_annual ** 2
        cons.append({"type": "ineq",  # te² - xᵀΣx ≥ 0
                     "fun": lambda x: te2 - x @ S @ x,
                     "jac": lambda x: -2.0 * (S @ x)})

    # 用线性规划找到的可行满仓权重做初值，而不是可能违反ETF总上限的等权初值。
    x0 = base_lp.x
    res = minimize(neg_util, x0, jac=neg_util_grad, bounds=bounds, constraints=cons,
                   method="SLSQP", options={"maxiter": 1000, "ftol": 1e-10})

    # 数值求解失败时回退到显式可行的线性规划解；不把越界的失败迭代点伪装成投资组合。
    x = np.clip(res.x, 0, None) if res.success else x0
    x = x / x.sum() * budget if x.sum() > 0 else x0
    w = pd.Series(x, index=codes)
    w = w[w > 1e-4].sort_values(ascending=False)  # 去掉极小权重

    exp_a = float(a @ x)
    idio_te = float(np.sqrt(max(x @ S @ x, 0.0)))
    realized_g = float(gl @ x)
    etf_w = float(x[is_etf].sum())
    style_gap = realized_g - target_growth
    message = str(res.message)
    if not style_feasible:
        message += f"；目标成长暴露 {target_growth:.1%} 不可达，可达区间 {growth_min:.1%}~{growth_max:.1%}"
    elif abs(style_gap) > style_tolerance:
        message += f"；风格目标偏离 {style_gap:+.2%}"

    return OptResult(
        # 保留全精度给回测/换手计算；展示层再格式化，避免四舍五入制造虚假换手。
        weights=w,
        exp_active_return=exp_a,
        idio_te=idio_te,
        realized_growth_exposure=realized_g,
        target_growth_exposure=float(target_growth),
        etf_weight=etf_w,
        n_holdings=int((w.index.isin([c for c in codes if c not in set(etf_codes)])).sum()),
        success=bool(res.success),
        message=message,
        diagnostics={
            "obj": float(res.fun), "iters": int(res.get("nit", -1)),
            "style_target_gap": style_gap, "style_feasible": style_feasible,
            "growth_reachable_min": growth_min, "growth_reachable_max": growth_max,
        },
    )
