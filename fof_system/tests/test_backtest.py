"""第⑥层滚动回测、事前风险和风格漂移监控测试。"""
from dataclasses import replace

import numpy as np
import pandas as pd

from fof_system import config
from fof_system.data import get_provider
from fof_system.data.pit import PITDataStore
from fof_system.engine import risk_model
from fof_system.engine.backtest import ChainBacktester
from fof_system.pipeline import build_benchmark


def test_walk_forward_backtest_reports_net_costs():
    """回测必须同时保留毛收益、净收益和每次调仓成本。"""
    provider = get_provider("mock")
    bt_cfg = replace(config.BACKTEST, fund_one_way_cost=0.002, etf_one_way_cost=0.001)
    output = ChainBacktester(
        provider, list(provider.truth), "2019-01-01", "2024-09-30",
        backtest_cfg=bt_cfg,
    ).run()

    assert output.rebalances > 2
    assert {"gross_port_ret", "transaction_cost", "port_ret", "bench_ret", "active"}.issubset(output.weekly)
    assert np.allclose(
        output.weekly["gross_port_ret"] - output.weekly["transaction_cost"],
        output.weekly["port_ret"],
    )
    assert output.cost_history.sum() > 0
    assert output.metrics["total_transaction_cost"] > 0
    assert output.target_growth_history.between(
        config.BENCHMARK_WEIGHTS["growth"] - config.STYLE_TIMING.max_tilt - 1e-9,
        config.BENCHMARK_WEIGHTS["growth"] + config.STYLE_TIMING.max_tilt + 1e-9,
    ).all()


def test_ex_ante_risk_report_reconciles_te_and_style():
    weights = pd.Series({"A": 0.6, "B": 0.4})
    alpha = pd.Series({"A": 0.04, "B": 0.02})
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.03]], index=weights.index, columns=weights.index)
    growth_load = pd.Series({"A": 0.8, "B": 0.55})

    report = risk_model.ex_ante_risk_report(
        weights, alpha, cov, growth_load, [], target_growth=0.7,
        benchmark_growth=0.7, te_budget=0.1,
    )

    assert report.idio_te > 0
    assert abs(report.growth_exposure - 0.7) < 1e-12
    assert abs(report.risk_contributions.sum() - report.idio_te) < 1e-12
    assert report.effective_holdings > 1


def test_style_drift_report_has_latest_rbsa_loads():
    provider = get_provider("mock")
    factor_df, _ = build_benchmark(provider, "2019-01-01", "2024-09-30")
    returns = pd.DataFrame({
        code: provider.to_returns(provider.get_fund_nav(code, "2019-01-01", "2024-09-30"), "W")
        for code in provider.truth
    })
    report = risk_model.style_drift_report(returns, factor_df, window=52)

    assert set(provider.truth).issubset(report.index)
    assert {"latest_growth_load", "growth_drift", "style_drift_alert"}.issubset(report.columns)
    assert report["latest_growth_load"].between(0, 1).all()


def test_backtester_uses_pit_status_at_each_rebalance(tmp_path):
    provider = get_provider("mock")
    store = PITDataStore(tmp_path / "pit")
    universe = provider.list_funds().assign(asset_type="fund", subscription_status="开放申购")
    store.write_universe_snapshot(universe, source="mock", available_at="2019-01-01")
    updated = universe.copy()
    updated.loc[updated["code"] == "F_GROWTH_STAR", "status"] = "suspended"
    store.write_universe_snapshot(
        updated, source="mock", effective_date="2024-01-01", available_at="2024-01-01",
    )

    from fof_system.rebalance import rebalance_at
    before = rebalance_at(
        provider, asof="2023-12-29", eval_start="2019-01-01", pit_store=store,
        strict_eligibility=False, score_cache_dir=tmp_path / "cache",
    )
    after = rebalance_at(
        provider, asof="2024-01-31", eval_start="2019-01-01", pit_store=store,
        strict_eligibility=False, score_cache_dir=tmp_path / "cache",
    )
    assert before.success and before.weights is not None
    assert "F_GROWTH_STAR" in before.weights.index
    if after.success and after.weights is not None:
        assert "F_GROWTH_STAR" not in after.weights.index


def test_actual_etf_uses_etf_transaction_cost_in_static_backtest():
    provider = get_provider("mock")
    bt_cfg = replace(config.BACKTEST, fund_one_way_cost=0.003, etf_one_way_cost=0.001)
    backtester = ChainBacktester(
        provider, ["F_GROWTH_STAR"], "2019-01-01", "2024-09-30", backtest_cfg=bt_cfg,
        asset_metadata={"F_GROWTH_STAR": {"asset_type": "etf", "is_stock_etf": True}},
    )
    cost = backtester._transaction_cost(
        pd.Series({"F_GROWTH_STAR": 1.0}), None, {"F_GROWTH_STAR"},
    )
    assert cost == bt_cfg.etf_one_way_cost
