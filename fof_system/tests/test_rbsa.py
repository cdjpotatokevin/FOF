"""RBSA 正确性测试：用 mock 数据的已知真值校验风格载荷与 alpha 的还原。"""
import numpy as np
import pandas as pd
import pytest

from fof_system.data import get_provider
from fof_system.engine.rbsa import run_rbsa, solve_style_weights


def _weekly(provider, code):
    return provider.to_returns(provider.get_fund_nav(code, "", ""), "W")


def _factor_weekly(provider):
    g = provider.to_returns(provider.get_index_close("399370", "", ""), "W")
    v = provider.to_returns(provider.get_index_close("399371", "", ""), "W")
    return pd.DataFrame({"growth": g, "value": v}).dropna()


def test_weights_sum_to_one_and_nonneg():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 0.02, (200, 2))
    true_w = np.array([0.7, 0.3])
    y = X @ true_w + rng.normal(0, 0.001, 200)
    w = solve_style_weights(y, X)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all()
    assert np.allclose(w, true_w, atol=0.05)


def test_recover_growth_fund_style():
    """成长型基金应被还原为高成长载荷，且 alpha 接近设定真值。"""
    mp = get_provider("mock")
    factors = _factor_weekly(mp)
    fr = _weekly(mp, "F_GROWTH_STAR")
    res = run_rbsa(fr, factors)
    assert res.weights["growth"] > res.weights["value"]
    assert res.weights["growth"] > 0.6
    # 真值 alpha=0.06，周频还原年化应同号且量级接近
    assert res.alpha_annual(52) > 0.02


def test_recover_value_fund_style():
    mp = get_provider("mock")
    factors = _factor_weekly(mp)
    fr = _weekly(mp, "F_VALUE_STAR")
    res = run_rbsa(fr, factors)
    assert res.weights["value"] > res.weights["growth"]
    assert res.weights["value"] > 0.6


def test_lagging_fund_has_negative_alpha():
    mp = get_provider("mock")
    factors = _factor_weekly(mp)
    fr = _weekly(mp, "F_GROWTH_LAG")  # 真值 alpha=-0.02
    res = run_rbsa(fr, factors)
    assert res.alpha_annual(52) < 0.0


def test_closet_index_high_r2():
    """影子指数基金特质噪声极小，风格 R² 应很高。"""
    mp = get_provider("mock")
    factors = _factor_weekly(mp)
    fr = _weekly(mp, "F_CLOSET_IDX")
    res = run_rbsa(fr, factors)
    assert res.style_r2 > 0.9
