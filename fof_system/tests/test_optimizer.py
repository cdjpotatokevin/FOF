"""第④层优化器与编排测试：验证约束满足与端到端链路。"""
import numpy as np
import pandas as pd
from dataclasses import replace
import pytest

from fof_system.data import get_provider
from fof_system.engine.optimizer import optimize_portfolio
from fof_system.engine import risk_model
from fof_system.portfolio import (
    build_portfolio, r2_adjusted_alpha, select_backup_candidates,
    select_full_universe_candidates,
)
from fof_system import config


def _toy_problem():
    codes = ["A", "B", "C"]
    alpha = pd.Series({"A": 0.06, "B": 0.03, "C": -0.01})
    cov = pd.DataFrame(np.diag([0.04, 0.03, 0.05]), index=codes, columns=codes)
    gl = pd.Series({"A": 0.8, "B": 0.5, "C": 0.2})
    return alpha, cov, gl, codes


def test_constraints_satisfied():
    alpha, cov, gl, codes = _toy_problem()
    res = optimize_portfolio(alpha, cov, gl, target_growth=0.55, etf_codes=[],
                             max_weight_fund=0.6, risk_aversion=5.0)
    w = res.weights.reindex(codes).fillna(0)
    assert abs(w.sum() - 1.0) < 1e-4              # 满仓
    assert (w >= -1e-6).all()                      # long-only
    assert w.max() <= 0.6 + 1e-4                    # 单票上限
    assert abs(res.realized_growth_exposure - 0.55) < 1e-3  # 风格暴露命中目标


def test_higher_alpha_gets_more_weight():
    # 用相同风格载荷，使风格约束不干预，纯看 alpha 偏好
    codes = ["A", "B", "C"]
    alpha = pd.Series({"A": 0.06, "B": 0.03, "C": -0.01})
    cov = pd.DataFrame(np.diag([0.04, 0.04, 0.04]), index=codes, columns=codes)
    gl = pd.Series({"A": 0.5, "B": 0.5, "C": 0.5})
    res = optimize_portfolio(alpha, cov, gl, target_growth=0.5, etf_codes=[],
                             max_weight_fund=1.0, risk_aversion=1.0)
    w = res.weights.reindex(codes).fillna(0)
    assert w["A"] > w["B"] > w["C"]   # 高alpha资产权重更高（载荷相同时）


def test_etf_completion_reaches_extreme_target():
    """当目标暴露超出基金载荷范围时，ETF 补全应让暴露可达。"""
    alpha, cov, gl, codes = _toy_problem()  # 基金成长载荷 0.2~0.8
    cov2 = risk_model.add_etf_assets(cov, ["ETF_GROWTH", "ETF_VALUE"])
    alpha2 = alpha.copy(); alpha2["ETF_GROWTH"] = 0; alpha2["ETF_VALUE"] = 0
    gl2 = gl.copy(); gl2["ETF_GROWTH"] = 1.0; gl2["ETF_VALUE"] = 0.0
    res = optimize_portfolio(alpha2, cov2, gl2, target_growth=0.92,
                             etf_codes=["ETF_GROWTH", "ETF_VALUE"],
                             max_weight_fund=0.6, etf_total_cap=0.6)
    assert abs(res.realized_growth_exposure - 0.92) < 1e-2
    assert res.etf_weight > 0   # 必须用到 ETF 才能到 0.92


def test_risk_aversion_reduces_te():
    alpha, cov, gl, codes = _toy_problem()
    lo = optimize_portfolio(alpha, cov, gl, 0.5, [], max_weight_fund=1.0, risk_aversion=0.5)
    hi = optimize_portfolio(alpha, cov, gl, 0.5, [], max_weight_fund=1.0, risk_aversion=50.0)
    assert hi.idio_te <= lo.idio_te + 1e-9   # 更厌恶风险 → 特质TE更低


def test_infeasible_style_target_is_explicitly_reported():
    """风格目标超出候选资产可达范围时，不能把数值求解成功误报为已命中目标。"""
    alpha, cov, gl, _ = _toy_problem()  # 成长载荷最高仅 0.8
    res = optimize_portfolio(alpha, cov, gl, target_growth=0.95, etf_codes=[],
                             max_weight_fund=0.6, risk_aversion=5.0)
    assert not res.diagnostics["style_feasible"]
    assert res.diagnostics["growth_reachable_max"] < 0.95
    assert "不可达" in res.message


def test_end_to_end_mock_portfolio():
    mp = get_provider("mock")
    codes = list(mp.truth.keys())
    res, detail = build_portfolio(mp, codes, target_growth=0.55,
                                  start="2019-01-01", end="2023-12-31")
    assert res.success
    assert abs(res.realized_growth_exposure - 0.55) < 0.02
    assert abs(detail["weight"].sum() - 1.0) < 0.02
    # 高 alpha 的成长之星应获得正权重
    assert "F_GROWTH_STAR" in set(detail["code"])


def test_actual_etf_from_pit_metadata_has_zero_alpha_and_etf_type():
    mp = get_provider("mock")
    cfg = replace(config.OPTIMIZER, use_style_etf=False, etf_total_cap=0.5, max_weight_fund=0.6)
    res, detail = build_portfolio(
        mp, list(mp.truth.keys()), target_growth=0.6, start="2019-01-01", end="2023-12-31", cfg=cfg,
        asset_metadata={"F_GROWTH_STAR": {"asset_type": "etf", "is_stock_etf": True}},
    )
    row = detail.set_index("code").loc["F_GROWTH_STAR"]
    assert row["type"] == "ETF"
    assert row["alpha_ann(收缩后)"] == 0.0
    assert res.etf_weight <= cfg.etf_total_cap + 1e-8


def test_optimizer_alpha_is_adjusted_by_clipped_style_r2():
    raw = pd.Series({"high": 0.10, "low": 0.10, "negative": 0.10})
    r2 = pd.Series({"high": 0.90, "low": 0.20, "negative": -0.30})

    adjusted = r2_adjusted_alpha(raw, r2)

    assert adjusted["high"] == pytest.approx(0.09)
    assert adjusted["low"] == pytest.approx(0.02)
    assert adjusted["negative"] == 0.0


def test_full_universe_selection_uses_score_not_aum_prefilter():
    scored = pd.DataFrame([
        {"code": "small-high", "asset_type": "fund", "aum_yi": 3, "composite_score": 90,
         "style_alpha_ann": 0.10, "info_ratio": 1.0},
        {"code": "large-low", "asset_type": "fund", "aum_yi": 300, "composite_score": 40,
         "style_alpha_ann": 0.02, "info_ratio": 0.2},
        {"code": "value-etf", "asset_type": "etf", "composite_score": 1,
         "growth_load": 0.02, "style_r2": 0.9},
        {"code": "growth-etf", "asset_type": "etf", "composite_score": 1,
         "growth_load": 0.98, "style_r2": 0.9},
    ])

    selected = select_full_universe_candidates(scored, n_active=1)

    assert selected == ["small-high", "value-etf", "growth-etf"]


def test_full_universe_selection_adds_low_growth_active_complements():
    scored = pd.DataFrame([
        {"code": "growth-1", "asset_type": "fund", "composite_score": 99, "style_alpha_ann": 0.1, "info_ratio": 1.0, "growth_load": 1.0},
        {"code": "growth-2", "asset_type": "fund", "composite_score": 98, "style_alpha_ann": 0.1, "info_ratio": 1.0, "growth_load": 0.9},
        {"code": "value-high", "asset_type": "fund", "composite_score": 80, "style_alpha_ann": 0.1, "info_ratio": 1.0, "growth_load": 0.2},
        {"code": "value-etf", "asset_type": "etf", "composite_score": 1, "growth_load": 0.0, "style_r2": 0.9},
        {"code": "growth-etf", "asset_type": "etf", "composite_score": 1, "growth_load": 1.0, "style_r2": 0.9},
    ])

    selected = select_full_universe_candidates(
        scored, n_active=2, target_growth=0.7, style_complement_candidates=1, style_complement_gap=0.15,
    )

    assert selected == ["value-high", "growth-1", "value-etf", "growth-etf"]


def test_backup_selection_excludes_held_and_retains_both_style_sides():
    scored = pd.DataFrame([
        {"code": "held-low", "asset_type": "fund", "composite_score": 99, "style_alpha_ann": 0.2, "info_ratio": 2.0, "growth_load": 0.1},
        {"code": "low-1", "asset_type": "fund", "composite_score": 90, "style_alpha_ann": 0.2, "info_ratio": 2.0, "growth_load": 0.2},
        {"code": "low-2", "asset_type": "fund", "composite_score": 89, "style_alpha_ann": 0.2, "info_ratio": 2.0, "growth_load": 0.3},
        {"code": "high-1", "asset_type": "fund", "composite_score": 88, "style_alpha_ann": 0.2, "info_ratio": 2.0, "growth_load": 1.0},
        {"code": "high-2", "asset_type": "fund", "composite_score": 87, "style_alpha_ann": 0.2, "info_ratio": 2.0, "growth_load": 0.9},
        {"code": "etf", "asset_type": "etf", "composite_score": 100, "style_alpha_ann": 0.2, "info_ratio": 2.0, "growth_load": 0.0},
    ])
    selected = select_backup_candidates(scored, {"held-low"}, n_backups=4, target_growth=0.7)
    assert selected == ["low-1", "low-2", "high-1", "high-2"]


def test_pit_metadata_disables_non_tradeable_synthetic_style_etfs():
    mp = get_provider("mock")
    cfg = replace(config.OPTIMIZER, max_weight_fund=0.2, etf_total_cap=0.4, use_style_etf=True)
    _, detail = build_portfolio(
        mp, list(mp.truth.keys()), target_growth=0.6, start="2019-01-01", end="2023-12-31", cfg=cfg,
        asset_metadata={"F_GROWTH_STAR": {"asset_type": "etf", "is_stock_etf": True}},
    )
    assert "ETF补全" not in set(detail["type"])


def test_optimizer_rejects_unfillable_candidate_set_instead_of_returning_bad_weights():
    alpha = pd.Series({"F1": 0.03, "F2": 0.02, "ETF": 0.0})
    cov = pd.DataFrame(np.eye(3) * 0.02, index=alpha.index, columns=alpha.index)
    growth = pd.Series({"F1": 0.6, "F2": 0.5, "ETF": 1.0})
    with pytest.raises(ValueError, match="无法满仓"):
        optimize_portfolio(
            alpha, cov, growth, target_growth=0.7, etf_codes=["ETF"],
            max_weight_fund=0.15, etf_total_cap=0.25,
        )


def test_per_code_capacity_cap_limits_real_etf_weight():
    alpha = pd.Series({"F1": 0.03, "F2": 0.02, "ETF": 0.0})
    cov = pd.DataFrame(np.eye(3) * 0.02, index=alpha.index, columns=alpha.index)
    growth = pd.Series({"F1": 0.5, "F2": 0.5, "ETF": 1.0})
    res = optimize_portfolio(
        alpha, cov, growth, target_growth=0.7, etf_codes=["ETF"],
        max_weight_fund=0.6, etf_total_cap=0.4,
        max_weight_by_code=pd.Series({"ETF": 0.05}),
    )
    assert res.weights.reindex(["ETF"]).fillna(0.0).iloc[0] <= 0.05 + 1e-8
    assert not res.diagnostics["style_feasible"]
